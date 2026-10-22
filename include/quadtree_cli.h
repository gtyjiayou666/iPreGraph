#pragma once
#include "quadtree_index.h"
#include <iostream>
#include <map>
#include <sstream>
#include <cctype>

namespace quad_cli {
inline std::map<std::string, std::string> args(int argc, char** argv) {
    std::map<std::string, std::string> out;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--help") { out[key] = "1"; continue; }
        if (key.size() < 3 || key.substr(0, 2) != "--" || i + 1 >= argc)
            throw std::runtime_error("expected --option value");
        if (!out.emplace(key, argv[++i]).second) throw std::runtime_error("duplicate option: " + key);
    }
    return out;
}
inline void known(const std::map<std::string, std::string>& a, const std::vector<std::string>& names) {
    for (const auto& p : a)
        if (std::find(names.begin(), names.end(), p.first) == names.end())
            throw std::runtime_error("unknown option: " + p.first);
}
inline std::string required(const std::map<std::string, std::string>& a, const std::string& key) {
    auto it = a.find(key);
    if (it == a.end() || it->second.empty()) throw std::runtime_error("missing " + key);
    return it->second;
}
inline uint32_t number(const std::map<std::string, std::string>& a, const std::string& key, uint32_t fallback) {
    auto it = a.find(key); if (it == a.end()) return fallback;
    if (it->second.empty() || !std::all_of(it->second.begin(), it->second.end(), [](char c) { return c >= '0' && c <= '9'; }))
        throw std::runtime_error("expected unsigned integer: " + key);
    auto v = std::stoull(it->second);
    if (v > std::numeric_limits<uint32_t>::max()) throw std::runtime_error("integer too large: " + key);
    return static_cast<uint32_t>(v);
}
inline iPreGraph::QuadSearchOptions search_options(const std::map<std::string, std::string>& a) {
    iPreGraph::QuadSearchOptions options;
    options.entry_points = number(a, "--entry_points", 5);
    options.edge_limit = number(a, "--edge_limit", 0);
    options.seed = number(a, "--seed", 42);
    if (!options.entry_points || (a.count("--edge_limit") && !options.edge_limit))
        throw std::runtime_error("entry_points and explicit edge_limit must be positive");
    return options;
}
// Numeric JSON arrays used by the existing multi-attribute benchmark. Flatten
// nested arrays in row-major order, but reject malformed JSON and non-numbers.
class NumericJson {
    std::istream& in;
    std::vector<double> values;
    [[noreturn]] void bad() const { throw std::runtime_error("expected JSON arrays of finite numbers"); }
    int peek() { in >> std::ws; return in.peek(); }
    static bool digit(int c) { return c >= '0' && c <= '9'; }
    void value(unsigned depth) {
        if (depth > 8) bad();
        int c = peek();
        if (c == '[') {
            in.get();
            if (peek() == ']') { in.get(); return; }
            for (;;) {
                value(depth + 1);
                c = peek(); in.get();
                if (c == ']') return;
                if (c != ',') bad();
            }
        }
        std::string token;
        if (c == '-') { token += char(in.get()); c = in.peek(); }
        if (!digit(c)) bad();
        if (c == '0') token += char(in.get());
        else while (digit(in.peek())) token += char(in.get());
        if (in.peek() == '.') {
            token += char(in.get()); if (!digit(in.peek())) bad();
            while (digit(in.peek())) token += char(in.get());
        }
        if (in.peek() == 'e' || in.peek() == 'E') {
            token += char(in.get());
            if (in.peek() == '+' || in.peek() == '-') token += char(in.get());
            if (!digit(in.peek())) bad();
            while (digit(in.peek())) token += char(in.get());
        }
        double v = std::stod(token); if (!std::isfinite(v)) bad(); values.push_back(v);
    }
public:
    explicit NumericJson(std::istream& stream) : in(stream) {}
    std::vector<double> read() {
        if (peek() != '[') bad();
        value(0);
        if (peek() != std::char_traits<char>::eof()) bad();
        return std::move(values);
    }
};
inline std::vector<double> json_numbers(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open JSON: " + path);
    return NumericJson(in).read();
}
inline std::vector<iPreGraph::QuadPoint> json_attributes(const std::string& path, size_t count) {
    auto values = json_numbers(path);
    if (values.size() != count * 2) throw std::runtime_error("attribute JSON must contain exactly two values per vector");
    std::vector<iPreGraph::QuadPoint> out; out.reserve(count);
    for (size_t i = 0; i < count; ++i) out.push_back({values[2*i], values[2*i+1]});
    return out;
}
// Each query contributes xmin, xmax, ymin, ymax after flattening.
// Nesting does not change the inclusive AND semantics of the two intervals.
inline std::vector<iPreGraph::QuadRect> json_ranges(const std::string& path, size_t count) {
    auto values = json_numbers(path);
    if (values.size() != count * 4) throw std::runtime_error("range JSON must contain xmin,xmax,ymin,ymax per query");
    std::vector<iPreGraph::QuadRect> out; out.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        iPreGraph::QuadRect r{values[4*i], values[4*i+1], values[4*i+2], values[4*i+3]};
        if (!r.valid()) throw std::runtime_error("invalid query rectangle at query " + std::to_string(i));
        out.push_back(r);
    }
    return out;
}
inline std::string format_arg(const std::map<std::string, std::string>& a, const std::string& key) {
    auto it = a.find(key); return it == a.end() ? "auto" : it->second;
}

// fvecs repeats [int32 dimension, float32[dimension]] for EVERY vector.
// The legacy bin format has one [int32 count, int32 dimension] file header.
// Both formats are little-endian; no heuristic reinterpretation of a bad fvecs.
class VectorReader {
    std::ifstream in;
    bool fvecs = false;
    uint32_t rows_read = 0;
    uint32_t u32() {
        unsigned char b[4]; in.read(reinterpret_cast<char*>(b), 4);
        if (!in) throw std::runtime_error("truncated vector dimension/header");
        return uint32_t(b[0]) | (uint32_t(b[1]) << 8) | (uint32_t(b[2]) << 16) | (uint32_t(b[3]) << 24);
    }
public:
    uint32_t count = 0, dimension = 0;
    VectorReader(const std::string& path, std::string format = "auto") : in(path, std::ios::binary) {
        if (!in) throw std::runtime_error("cannot open vectors: " + path);
        if (format == "auto") {
            auto ext = std::filesystem::path(path).extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return char(std::tolower(c)); });
            format = ext == ".fvecs" ? "fvecs" : "bin";
        }
        if (format != "fvecs" && format != "bin") throw std::runtime_error("vector format must be auto, fvecs, or bin");
        fvecs = format == "fvecs";
        in.seekg(0, std::ios::end); auto end = in.tellg();
        if (end < std::streamoff(fvecs ? 4 : 8)) throw std::runtime_error("empty or truncated vector file");
        const uint64_t bytes = static_cast<uint64_t>(end); in.seekg(0);
        if (fvecs) dimension = u32();
        else { count = u32(); dimension = u32(); }
        if (!dimension || dimension > uint32_t(std::numeric_limits<int32_t>::max()))
            throw std::runtime_error("invalid vector dimension");
        if (fvecs) {
            const uint64_t row_bytes = 4 + uint64_t(dimension) * 4;
            if (bytes % row_bytes || bytes / row_bytes > uint32_t(std::numeric_limits<int32_t>::max()))
                throw std::runtime_error("fvecs file length mismatch or truncated record");
            count = static_cast<uint32_t>(bytes / row_bytes);
            in.seekg(0);
        } else if (bytes != 8 + uint64_t(count) * dimension * 4) {
            throw std::runtime_error("bin vector file length mismatch");
        }
        if (!count || count > uint32_t(std::numeric_limits<int32_t>::max()))
            throw std::runtime_error("invalid vector count");
    }
    void read_row(std::vector<float>& row) {
        if (rows_read >= count) throw std::runtime_error("read past vector count");
        if (fvecs && u32() != dimension)
            throw std::runtime_error("inconsistent fvecs dimension at row " + std::to_string(rows_read));
        row.resize(dimension);
        in.read(reinterpret_cast<char*>(row.data()), std::streamsize(dimension) * 4);
        if (!in) throw std::runtime_error("truncated vector payload");
        const uint32_t one = 1;
        if (*reinterpret_cast<const unsigned char*>(&one) != 1) {
            for (float& v : row) {
                auto b = reinterpret_cast<unsigned char*>(&v);
                std::swap(b[0], b[3]); std::swap(b[1], b[2]);
            }
        }
        for (float v : row) if (!std::isfinite(v)) throw std::runtime_error("non-finite vector component");
        ++rows_read;
    }
};
inline std::vector<std::vector<float>> vectors(const std::string& path, const std::string& format = "auto") {
    VectorReader reader(path, format);
    std::vector<std::vector<float>> out(reader.count);
    for (auto& row : out) reader.read_row(row);
    return out;
}
template<size_t N>
inline std::vector<std::array<double, N>> rows(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open numeric rows: " + path);
    std::vector<std::array<double, N>> out;
    std::string line; size_t lineno = 0;
    while (std::getline(in, line)) {
        ++lineno;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream ss(line); ss >> std::ws;
        if (ss.eof()) continue;
        std::array<double, N> row;
        for (auto& v : row)
            if (!(ss >> v) || !std::isfinite(v)) throw std::runtime_error("invalid numeric row " + std::to_string(lineno));
        ss >> std::ws;
        if (!ss.eof()) throw std::runtime_error("too many columns at row " + std::to_string(lineno));
        out.push_back(row);
    }
    return out;
}
} // namespace quad_cli
