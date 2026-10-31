// Regression coverage for the migrated iRangeGraph multi-attribute search.
// Run with a writable fixture directory; no external datasets are required.
#include "search_zorder.h"
#include <filesystem>
#include <set>
#include <sstream>

namespace {
using Index = iPreGraph_multi::iPreGraph_Search_Zorder<float>;
using Loader = iPreGraph_multi::DataLoader;
using Node = iPreGraph_multi::TreeNode;
using Bounds = std::vector<std::pair<int, int>>;

void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

template<class Function>
void expect_error(Function function, const char *message) {
    bool threw = false;
    try { function(); } catch (const std::exception &) { threw = true; }
    require(threw, message);
}

std::vector<char> predicate(const Loader &data, const Bounds &bounds) {
    std::vector<char> flags(data.data_nb, 1);
    for (int id = 0; id < data.data_nb; ++id)
        for (int d = 0; d < data.attr_nb; ++d)
            if (data.attributes[id][d] < bounds[d].first ||
                data.attributes[id][d] > bounds[d].second) flags[id] = 0;
    return flags;
}

std::vector<std::pair<float, int>> exact(const Loader &data,
        const std::vector<float> &query, const std::vector<char> &flags, size_t k) {
    std::vector<std::pair<float, int>> result;
    for (int id = 0; id < data.data_nb; ++id) {
        if (!flags[id]) continue;
        float distance = 0;
        for (int d = 0; d < data.Dim; ++d) {
            float delta = query[d] - data.data_points[id][d];
            distance += delta * delta;
        }
        result.emplace_back(distance, id);
    }
    std::sort(result.begin(), result.end());
    if (result.size() > k) result.resize(k);
    return result;
}

std::vector<std::pair<float, int>> unpack(std::priority_queue<PFI> result,
        const std::vector<char> &flags, size_t k) {
    require(result.size() <= k, "search returned more than k neighbors");
    std::set<int> seen;
    std::vector<std::pair<float, int>> rows;
    while (!result.empty()) {
        const auto hit = result.top(); result.pop();
        require(hit.second >= 0 && static_cast<size_t>(hit.second) < flags.size(), "invalid result ID");
        require(flags[hit.second] != 0, "result violates the full attribute predicate");
        require(seen.insert(hit.second).second, "duplicate result ID");
        rows.push_back(hit);
    }
    std::sort(rows.begin(), rows.end());
    return rows;
}

void write_vectors(const std::filesystem::path &path,
                   const std::vector<std::vector<float>> &rows) {
    std::ofstream out(path, std::ios::binary);
    int count = static_cast<int>(rows.size()), dim = static_cast<int>(rows.front().size());
    out.write(reinterpret_cast<const char *>(&count), sizeof(count));
    out.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
    for (const auto &row : rows)
        out.write(reinterpret_cast<const char *>(row.data()), row.size() * sizeof(float));
    require(bool(out), "cannot write vector fixture");
}

// Complete adjacency within each tree node makes the full-range oracle exact.
// The serialized graph follows the existing constructor's point/layer layout.
void write_graph(const std::filesystem::path &path, Loader &data) {
    iPreGraph_multi::SegmentTreeZorder tree(data.data_nb, data.attr_nb);
    tree.BuildTree(tree.root, data.attributes);
    std::ofstream out(path, std::ios::binary);
    for (int id = 0; id < data.data_nb; ++id) {
        for (int depth = 0; depth <= tree.max_depth; ++depth) {
            std::vector<unsigned> neighbors;
            for (auto *node : tree.treenodes) {
                if (node->depth != depth || id < node->lbound || id > node->rbound) continue;
                for (int other = node->lbound; other <= node->rbound; ++other)
                    if (other != id) neighbors.push_back(static_cast<unsigned>(other));
            }
            unsigned count = static_cast<unsigned>(neighbors.size());
            out.write(reinterpret_cast<const char *>(&count), sizeof(count));
            out.write(reinterpret_cast<const char *>(neighbors.data()), neighbors.size() * sizeof(unsigned));
        }
    }
    require(bool(out), "cannot write graph fixture");
}

void run_case(const std::filesystem::path &directory, int dimensions) {
    std::filesystem::create_directories(directory);
    Loader data;
    data.data_nb = 8; data.Dim = 2; data.attr_nb = dimensions;
    data.query_nb = 1; data.query_K = 2;
    const int base = dimensions == 4 ? (1 << 24) : 0;
    for (int id = 0; id < data.data_nb; ++id) {
        // Every coordinate grows with row ID, so these rows have Morton order.
        data.attributes.emplace_back(dimensions, base + id);
        data.data_points.push_back({float(id), float(id * id)});
    }
    data.query_points = {{2.25f, 4.5f}};
    data.CalMultiZorder();
    require(std::is_sorted(data.multi_zorder.begin(), data.multi_zorder.end(),
                          iPreGraph_multi::MultiAttributeZOrderLess), "fixture is not Morton sorted");
    if (dimensions == 4)
        require(data.multi_zorder.front().size() > 1, "four-attribute fixture must exercise more than 64 Morton bits");
    Bounds bounds(dimensions, {base, base + 7});
    data.query_range_multi["mixed"] = {Attr_Constraint{bounds}};
    auto flags = predicate(data, bounds);
    const auto expected = exact(data, data.query_points[0], flags, data.query_K);
    data.ground_truth["mixed"] = {{expected[0].second, expected[1].second}};
    write_graph(directory / "index.bin", data);
    write_vectors(directory / "base.bin", data.data_points);
    write_vectors(directory / "queries.bin", data.query_points);
    {
        std::ofstream attributes(directory / "attributes.json"), ranges(directory / "ranges.json"), gt(directory / "gt.json");
        attributes << '[';
        for (int id = 0; id < data.data_nb; ++id) for (int d = 0; d < dimensions; ++d)
            attributes << (id || d ? "," : "") << data.attributes[id][d];
        attributes << ']';
        ranges << '[';
        for (int d = 0; d < dimensions; ++d)
            ranges << (d ? "," : "") << base << ',' << base + 7;
        ranges << ']'; gt << '[' << expected[0].second << ',' << expected[1].second << ']';
    }
    Index index((directory / "index.bin").string(), &data, 16);
    auto nodes = index.tree->range_filter(index.tree->root, 0, data.data_nb - 1);
    require(index.probability.empty(), "fixture must exercise lazy probability initialization");
    for (unsigned seed : {1u, 7u, 42u}) {
        auto found = index.TopDown_nodeentries_search_mutil(nodes, data.query_points[0].data(),
            0, 7, 16, 2, 16, flags, seed);
        require(unpack(found, flags, 2) == expected, "full-range search disagrees with brute-force L2");
    }
    require(index.probability.size() == static_cast<size_t>(index.MaxStep), "probability table was not initialized");

    // In-range result IDs must satisfy every attribute, even when the position
    // cover deliberately contains nonqualifying points.
    bounds.back() = {base + 2, base + 5};
    auto filtered = predicate(data, bounds);
    auto found = index.TopDown_nodeentries_search_mutil(nodes, data.query_points[0].data(),
        0, 7, 16, 2, 16, filtered, 42);
    require(!unpack(found, filtered, 2).empty(), "filtered complete graph found no qualifying points");
    std::vector<char> none(data.data_nb, 0);
    require(index.TopDown_nodeentries_search_mutil(nodes, data.query_points[0].data(),
        0, 7, 16, 2, 16, none, 42).empty(), "no-match predicate returned neighbors");
    std::vector<Node *> empty;
    require(index.TopDown_nodeentries_search_mutil(empty, data.query_points[0].data(),
        8, 7, 16, 2, 16, flags, 42).empty(), "empty Morton cover returned neighbors");

    // At the transit-depth cap, nonqualifying neighbors cannot be selected.
    index.MaxStep = 3; index.setprob();
    searcher::Bitset<uint64_t> visited(data.data_nb);
    visited.set(2);
    auto edges = index.SelectEdgeMutil(0, 0, 7, 2, 2, visited, filtered);
    require(edges.size() <= 2, "edge selection exceeded its budget");
    for (auto edge : edges)
        require(edge.second && filtered[edge.first] && edge.first != 2, "transit cap or visited filtering failed");

    auto invalid = [&](int ef, int k, int limit, const std::vector<char> &mask) {
        index.TopDown_nodeentries_search_mutil(nodes, data.query_points[0].data(), 0, 7, ef, k, limit, mask, 42);
    };
    expect_error([&] { invalid(0, 2, 16, flags); }, "zero EF accepted");
    expect_error([&] { invalid(16, 0, 16, flags); }, "zero k accepted");
    expect_error([&] { invalid(16, 2, 0, flags); }, "zero edge limit accepted");
    expect_error([&] { invalid(16, 2, 16, std::vector<char>(1, 1)); }, "wrong predicate size accepted");
    index.MaxStep = 0;
    expect_error([&] { invalid(16, 2, 16, flags); }, "invalid transit depth accepted");
    index.MaxStep = 20;

    const auto saved_keys = data.multi_zorder;
    std::vector<int> ef{16};
    const auto prefix = (directory / "result_").string();
    index.search_irange(ef, prefix, 16);
    require(data.multi_zorder == saved_keys, "benchmark changed caller-owned Morton keys");
    std::ifstream csv(prefix + "mixed-irange.csv");
    std::string line;
    require(bool(std::getline(csv, line)), "benchmark did not write a CSV row");
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream row(line);
    int actual_ef; double recall, qps, distances, hops;
    require(bool(row >> actual_ef >> recall >> qps >> distances >> hops), "invalid benchmark CSV layout");
    require(actual_ef == 16 && recall == 1.0 && distances > 0 && hops > 0,
            "benchmark results disagree with the full-range oracle");
    std::cout << "PASS: " << dimensions << " attributes, predicate/ID checks, invalid inputs, and benchmark CSV\n";
}
} // namespace

int main(int argc, char **argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: irange_multi_test <fixture-directory>");
        // Allocator copies must free the requested block, including after a
        // second allocation. This also exercises native Windows aligned frees.
        memory::align_alloc<int> allocator;
        int *small = allocator.allocate(32);
        int *large = allocator.allocate(20000);
        require(reinterpret_cast<uintptr_t>(small) % 64 == 0, "small block alignment");
        require(reinterpret_cast<uintptr_t>(large) % (1u << 21) == 0, "large block alignment");
        small[0] = 17; large[19999] = 23;
        memory::align_alloc<int> copy = allocator;
        copy.deallocate(small, 32);
        require(large[19999] == 23, "deallocated the wrong block");
        allocator.deallocate(large, 20000);
        searcher::Bitset<uint32_t> bits(65);
        bits.set(64);
        require(bits.get(64) && !bits.get(63), "32-bit bitset block layout");
        const std::filesystem::path directory(argv[1]);
        run_case(directory / "2d", 2);
        run_case(directory / "4d", 4);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
