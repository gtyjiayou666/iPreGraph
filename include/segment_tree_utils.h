#pragma once

#include "space_l2.h"
#include <filesystem>
#include <string>
#include <cstring>
#include <vector>
#include <fstream>
#include <sys/time.h>
#include <map>
#include <thread>
class Exception : public std::runtime_error
{
public:
    Exception(const std::string &msg) : std::runtime_error(msg) {}
};

void CheckPath(std::string filename)
{
    std::filesystem::path pathObj(filename);
    std::filesystem::path dirPath = pathObj.parent_path();
    if (!std::filesystem::exists(dirPath))
    {
        try
        {
            if (std::filesystem::create_directories(dirPath))
            {
                std::cout << "Directory created: " << dirPath << std::endl;
            }
            else
            {
                std::cerr << "Failed to create directory: " << dirPath << std::endl;
            }
        }
        catch (std::filesystem::filesystem_error &e)
        {
            throw Exception(e.what());
        }
    }
}

float GetTime(timeval &begin, timeval &end)
{
    return end.tv_sec - begin.tv_sec + (end.tv_usec - begin.tv_usec) * 1.0 / 1000000.0;
}

namespace iPreGraph
{
    struct points
    {
        int left;
        int right;
        int num;
        points(int left, int right, int num)
            : left(left), right(right), num(num) {}
    };
    typedef std::pair<float, std::pair<int, int>> PFII;

    typedef std::pair<float, int> PFI;
    struct PFIORPRO
    {
        float dis;
        int id;
        int range_id;
        PFIORPRO(float d, int i, int range_id)
            : dis(d), id(i), range_id(range_id) {}
    };
    struct PFIORULTRA
    {
        float dis;
        int id;
        int range_id;
        bool out;
        PFIORULTRA(float d, int i, int range_id, bool out)
            : dis(d), id(i), range_id(range_id), out(out) {}
    };
    struct CompareULTRA
    {
        bool operator()(const PFIORULTRA &a, const PFIORULTRA &b)
        {
            return a.dis > b.dis; // 小顶堆：dis 小的优先（在堆顶）
        }
    };
    struct Compare
    {
        bool operator()(const PFIORPRO &a, const PFIORPRO &b)
        {
            return a.dis > b.dis; // 小顶堆：dis 小的优先（在堆顶）
        }
    };

    typedef unsigned int tableint;
    typedef unsigned int linklistsizeint;

    class DataLoader
    {
    public:
        int Dim, query_nb, query_K;
        std::vector<std::vector<float>> query_points;
        int data_nb;
        std::vector<std::vector<float>> data_points;
        std::unordered_map<int, std::vector<std::pair<int, int>>> query_range;
        std::unordered_map<int, std::vector<std::vector<int>>> groundtruth;

        DataLoader() {}
        ~DataLoader() {}

        // query vector filename format: 4 bytes: query number; 4 bytes: dimension; query_nb*Dim vectors
        void LoadQuery(std::string filename)
        {
            std::ifstream infile(filename, std::ios::in | std::ios::binary);
            if (!infile.is_open())
                throw Exception("cannot open " + filename);
            infile.read((char *)&query_nb, sizeof(int));
            infile.read((char *)&Dim, sizeof(int));
            query_points.resize(query_nb);
            for (int i = 0; i < query_nb; i++)
            {
                query_points[i].resize(Dim);
                infile.read((char *)query_points[i].data(), Dim * sizeof(float));
            }
            infile.close();
        }

        // Used only when computing groundtruth and constructing index. Do not use this to load data for search process
        void LoadData(std::string filename)
        {
            std::ifstream infile(filename, std::ios::in | std::ios::binary);
            if (!infile.is_open())
                throw Exception("cannot open " + filename);
            infile.read((char *)&data_nb, sizeof(int));
            infile.read((char *)&Dim, sizeof(int));
            data_points.resize(data_nb);
            for (int i = 0; i < data_nb; i++)
            {
                data_points[i].resize(Dim);
                infile.read((char *)data_points[i].data(), Dim * sizeof(float));
            }
            infile.close();
        }

        // By default generation, 0.bin~9.bin denotes 2^0~2^-9 range fractions, 17.bin denotes mixed range fraction.
        // Before reading the query ranges, make sure query vectors have been read.
        void LoadQueryRange(std::string fileprefix)
        {
            std::vector<int> s;
            for (int i = 0; i < 10; i++)
                s.emplace_back(i);
            s.emplace_back(17);
            for (auto suffix : s)
            {
                std::string filename = fileprefix + std::to_string(suffix) + ".bin";
                std::ifstream infile(filename, std::ios::in | std::ios::binary);
                if (!infile.is_open())
                    throw Exception("cannot open " + filename);
                for (int i = 0; i < query_nb; i++)
                {
                    int ql, qr;
                    infile.read((char *)&ql, sizeof(int));
                    infile.read((char *)&qr, sizeof(int));
                    query_range[suffix].emplace_back(ql, qr);
                }
                infile.close();
            }
        }

        // 0.bin~9.bin correspond to groundtruth for 2^0~2^-9 range fractions, 17.bin for mixed fraction
        void LoadGroundtruth(std::string fileprefix)
        {
            for (auto t : query_range)
            {
                int suffix = t.first;
                std::string filename = fileprefix + std::to_string(suffix) + ".bin";
                std::ifstream infile(filename, std::ios::in | std::ios::binary);
                if (!infile.is_open())
                    throw Exception("cannot open " + filename);
                groundtruth[suffix].resize(query_nb);
                for (int i = 0; i < query_nb; i++)
                {
                    groundtruth[suffix][i].resize(query_K);
                    infile.read((char *)groundtruth[suffix][i].data(), query_K * sizeof(int));
                }
                infile.close();
            }
        }
    };

    // ---- helpers for label_or: read integer payloads from JSON/text files ----
    // 读取整个文件为字符串
    inline std::string fan_read_whole_file(const std::string &filename)
    {
        std::ifstream in(filename);
        if (!in.is_open())
            throw Exception("cannot open " + filename);
        std::stringstream buf;
        buf << in.rdbuf();
        return buf.str();
    }

    // 从文本里顺序提取所有整数（用于 label range 文件：label l r 三元组）
    inline std::vector<int> fan_parse_all_ints(const std::string &text, const std::string &who)
    {
        std::vector<int> out;
        size_t i = 0;
        while (i < text.size())
        {
            char c = text[i];
            bool is_num_start = (c == '-' || c == '+' || std::isdigit(static_cast<unsigned char>(c)));
            if (!is_num_start)
            {
                ++i;
                continue;
            }
            size_t start = i;
            if (c == '-' || c == '+')
                ++i;
            bool has_digit = false;
            while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i])))
            {
                ++i;
                has_digit = true;
            }
            if (!has_digit)
            {
                // 单独的 +/- 号，跳过
                ++i;
                continue;
            }
            // 若数字后紧跟小数点或指数，说明不是整数，明确报错
            if (i < text.size() && (text[i] == '.' || text[i] == 'e' || text[i] == 'E'))
                throw Exception("non-integer number in " + who);
            std::string token = text.substr(start, i - start);
            errno = 0;
            char *end = nullptr;
            long long v = std::strtoll(token.c_str(), &end, 10);
            if (errno == ERANGE || v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max())
                throw Exception("integer out of int32 range in " + who + ": " + token);
            out.push_back(static_cast<int>(v));
        }
        return out;
    }

    class DataLoaderOr
    {
    public:
        int Dim, query_nb, query_K;
        std::vector<std::vector<float>> query_points;
        int data_nb;
        std::vector<std::vector<float>> data_points;
        std::vector<int> label_arr;
        std::vector<std::unordered_map<int, std::vector<std::pair<int, int>>>> query_range;
        std::unordered_map<int, std::vector<std::vector<int>>> groundtruth;
        int range_num;

        DataLoaderOr(int range_num) : range_num(range_num)
        {
            query_range.resize(range_num);
        }
        DataLoaderOr(const std::string &filename,
                     int attr_num = 3,
                     int label_col = 0)
        {
            if (attr_num <= 0 || label_col < 0 || label_col >= attr_num)
                throw Exception("load_label_array: need 0 <= label_col < attr_num (got attr_num=" +
                                std::to_string(attr_num) + ", label_col=" + std::to_string(label_col) + ")");

            std::vector<int> v = fan_parse_all_ints(fan_read_whole_file(filename), filename);
            if ((long long)v.size() % attr_num != 0)
                throw Exception("attribute file " + filename + ": integer count (" + std::to_string(v.size()) +
                                ") is not a multiple of attr_num (" + std::to_string(attr_num) + ")");

            int npoints = (int)(v.size() / attr_num);

            label_arr.resize(npoints);
            for (int pid = 0; pid < npoints; pid++)
                label_arr[pid] = v[pid * attr_num + label_col];

            int distinct = npoints > 0 ? 1 : 0;
            for (int pid = 1; pid < npoints; pid++)
            {
                if (label_arr[pid] < label_arr[pid - 1])
                    throw Exception("attribute file " + filename + ": label decreases at pid " +
                                    std::to_string(pid) + " (" + std::to_string(label_arr[pid - 1]) + " -> " +
                                    std::to_string(label_arr[pid]) + ") -- base must be sorted by label ascending");
                if (label_arr[pid] != label_arr[pid - 1])
                    distinct++;
            }

            std::cout << "loaded " << npoints << " labels (" << distinct << " distinct) from " << filename << std::endl;
        }
        ~DataLoaderOr() {}

        // query vector filename format: 4 bytes: query number; 4 bytes: dimension; query_nb*Dim vectors
        void LoadQuery(std::string filename)
        {
            std::ifstream infile(filename, std::ios::in | std::ios::binary);
            if (!infile.is_open())
                throw Exception("cannot open " + filename);
            infile.read((char *)&query_nb, sizeof(int));
            infile.read((char *)&Dim, sizeof(int));
            std::cout << "dim=" << Dim << std::endl;
            query_points.resize(query_nb);
            for (int i = 0; i < query_nb; i++)
            {
                query_points[i].resize(Dim);
                infile.read((char *)query_points[i].data(), Dim * sizeof(float));
            }
            infile.close();
        }

        // Used only when computing groundtruth and constructing index. Do not use this to load data for search process
        void LoadData(std::string filename)
        {
            std::ifstream infile(filename, std::ios::in | std::ios::binary);
            if (!infile.is_open())
                throw Exception("cannot open " + filename);
            infile.read((char *)&data_nb, sizeof(int));
            infile.read((char *)&Dim, sizeof(int));
            data_points.resize(data_nb);
            for (int i = 0; i < data_nb; i++)
            {
                data_points[i].resize(Dim);
                infile.read((char *)data_points[i].data(), Dim * sizeof(float));
            }
            infile.close();
        }

        // By default generation, 0.bin~9.bin denotes 2^0~2^-9 range fractions, 17.bin denotes mixed range fraction.
        // Before reading the query ranges, make sure query vectors have been read.
        void LoadQueryRange(std::string fileprefix)
        {
            std::vector<int> s;
            for (int i = 0; i < 5; i++)
                s.emplace_back(i);
            for (auto suffix : s)
            {
                std::string filename = fileprefix + std::to_string(suffix) + ".bin";
                std::ifstream infile(filename, std::ios::in | std::ios::binary);
                if (!infile.is_open())
                    throw Exception("cannot open " + filename);
                int range_num;
                infile.read(reinterpret_cast<char *>(&range_num), sizeof(range_num));
                for (int i = 0; i < query_nb; i++)
                {
                    std::vector<std::pair<int, int>> ranges;
                    ranges.reserve(range_num);

                    for (int j = 0; j < range_num; ++j)
                    {
                        int ql, qr;
                        infile.read(reinterpret_cast<char *>(&ql), sizeof(ql));
                        infile.read(reinterpret_cast<char *>(&qr), sizeof(qr));
                        // if (suffix == 0)
                        // {
                        //     std::cout << "(" << ql << "," << qr << ")";
                        // }

                        if (infile.fail())
                        {
                            throw std::runtime_error("File " + filename + " corrupted at query " + std::to_string(i));
                        }
                        query_range[j][suffix].emplace_back(ql, qr);
                    }
                }
                infile.close();
            }
        }

        // 0.bin~9.bin correspond to groundtruth for 2^0~2^-9 range fractions, 17.bin for mixed fraction
        void LoadGroundtruth(std::string fileprefix)
        {
            for (auto t : query_range[0])
            {
                int suffix = t.first;
                std::string filename = fileprefix + std::to_string(suffix) + ".bin";
                std::ifstream infile(filename, std::ios::in | std::ios::binary);
                if (!infile.is_open())
                    throw Exception("cannot open " + filename);
                groundtruth[suffix].resize(query_nb);
                for (int i = 0; i < query_nb; i++)
                {
                    groundtruth[suffix][i].resize(query_K);
                    infile.read((char *)groundtruth[suffix][i].data(), query_K * sizeof(int));
                }
                infile.close();
            }
        }

    };

    class QueryGeneratorOr
    {
    public:
        int data_nb, query_nb, range_num;
        hnswlib::L2Space *space;

        QueryGeneratorOr(int data_num, int query_num, int range_num) : data_nb(data_num), query_nb(query_num), range_num(range_num) {}
        ~QueryGeneratorOr() {}

        // 辅助函数：将 total 拆分为 count 个正整数（每个 >=1）
        std::vector<int> split_positive(int total, int count, std::default_random_engine &e)
        {
            if (total < count)
            {
                throw Exception("Cannot split " + std::to_string(total) + " into " + std::to_string(count) + " positive integers.");
            }
            std::vector<int> cuts(count - 1);
            std::uniform_int_distribution<int> dist(1, total - 1);
            for (int i = 0; i < count - 1; ++i)
            {
                cuts[i] = dist(e);
            }
            std::sort(cuts.begin(), cuts.end());
            cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end()); // 去重（概率低但安全）

            // 如果去重后不足，补足（极小概率）
            while (static_cast<int>(cuts.size()) < count - 1)
            {
                int x = dist(e);
                if (std::find(cuts.begin(), cuts.end(), x) == cuts.end())
                {
                    cuts.push_back(x);
                    std::sort(cuts.begin(), cuts.end());
                }
            }

            std::vector<int> parts;
            int last = 0;
            for (int cut : cuts)
            {
                parts.push_back(cut - last);
                last = cut;
            }
            parts.push_back(total - last);
            return parts;
        }

        // 辅助函数：将 total 拆分为 count 个非负整数（≥0）
        std::vector<int> split_nonnegative(int total, int count, std::default_random_engine &e)
        {
            if (total < 0)
            {
                throw Exception("Total gap cannot be negative!");
            }
            if (count == 1)
            {
                return {total};
            }
            // 使用“星与条”方法：在 total + count - 1 个位置选 count - 1 个分隔点
            std::vector<int> cuts(count - 1);
            std::uniform_int_distribution<int> dist(0, total + count - 2);
            for (int i = 0; i < count - 1; ++i)
            {
                cuts[i] = dist(e);
            }
            std::sort(cuts.begin(), cuts.end());
            cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());

            while (static_cast<int>(cuts.size()) < count - 1)
            {
                int x = dist(e);
                if (std::find(cuts.begin(), cuts.end(), x) == cuts.end())
                {
                    cuts.push_back(x);
                    std::sort(cuts.begin(), cuts.end());
                }
            }

            std::vector<int> parts;
            int last = -1;
            for (int cut : cuts)
            {
                parts.push_back(cut - last - 1);
                last = cut;
            }
            parts.push_back(total + count - 1 - last - 1);
            return parts;
        }

        void GenerateRange(std::string saveprefix)
        {
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine e(seed);

            std::vector<float> fenshu = {0.1, 0.05, 0.01, 0.005, 0.001};
            std::vector<std::pair<int, int>> rs;
            // int current_len = data_nb;
            // current_len /= 2;
            for (int i = 0; i < fenshu.size(); i++)
            {
                rs.emplace_back(this->data_nb * fenshu[i], i);
                // current_len /= 2;
            }
            for (auto t : rs)
            {
                int total_len = t.first;
                int suffix = t.second;
                std::string savepath = saveprefix + std::to_string(suffix) + ".bin";
                CheckPath(savepath);
                std::cout << "save query range to " << savepath << std::endl;
                std::ofstream outfile(savepath, std::ios::out | std::ios::binary);
                if (!outfile.is_open())
                    throw Exception("cannot open " + savepath);
                outfile.write((char *)(&range_num), sizeof(int));

                // 拆分长度分布
                // std::uniform_int_distribution<int> u_split(1, total_len - 1);
                // 分隔点分布：必须留出至少 len + 1 的空间
                if (data_nb < total_len + 1)
                {
                    throw Exception("data_nb too small to fit two non-overlapping ranges with total length " + std::to_string(total_len));
                }

                for (int i = 0; i < query_nb; i++)
                {
                    std::vector<int> lengths = split_positive(total_len, range_num, e);
                    int gap_total = (this->data_nb - total_len);
                    // int gap_total = total_len;
                    std::vector<int> gaps = split_nonnegative(gap_total, range_num - 1, e);
                    // if (data_nb - total_len - total_len - 1 > 1)
                    // {
                    //     std::uniform_int_distribution<int> dist(1, data_nb - total_len - total_len - 1);
                    //     x = dist(e);
                    // }
                    gaps.push_back(0);
                    std::vector<std::pair<int, int>> ranges;
                    // std::uniform_int_distribution<int> dist(0, data_nb - 1000000);
                    // int pos = dist(e);
                    int pos = 0;
                    for (int j = 0; j < range_num; ++j)
                    {
                        int start = pos;
                        int end = start + lengths[j] - 1;
                        ranges.emplace_back(start, end);
                        pos = end + 1 + gaps[j]; // 跳过区间 + 下一个间隙
                    }

                    for (const auto &r : ranges)
                    {
                        // std::cout << r.first << ", " << r.second << std::endl;
                        outfile.write(reinterpret_cast<const char *>(&r.first), sizeof(int));
                        outfile.write(reinterpret_cast<const char *>(&r.second), sizeof(int));
                    }
                }

                outfile.close();
            }
        }

        float dis_compute(std::vector<float> &v1, std::vector<float> &v2)
        {
            hnswlib::DISTFUNC<float> fstdistfunc_ = space->get_dist_func();
            float dis = fstdistfunc_((char *)v1.data(), (char *)v2.data(), space->get_dist_func_param());
            return dis;
        }

        void GenerateGroundtruth(std::string saveprefix, DataLoaderOr &storage)
        {
            space = new hnswlib::L2Space(storage.Dim);
            for (int suffix = 0; suffix < 14; ++suffix)
            {
                std::string savepath = saveprefix + std::to_string(suffix) + ".bin";
                CheckPath(savepath);
                std::ofstream outfile(savepath, std::ios::out | std::ios::binary);
                if (!outfile.is_open())
                    throw Exception("cannot open " + savepath);

                std::cout << "Generating ground truth for suffix " << suffix << std::endl;

                for (int i = 0; i < storage.query_nb; ++i)
                {
                    std::priority_queue<std::pair<float, int>> ans; // max-heap

                    // 遍历所有 range_num 个区间
                    for (int k = 0; k < range_num; ++k)
                    {
                        // 获取第 k 个区间中，suffix 对应的第 i 个查询范围
                        auto [ql, qr] = storage.query_range[k].at(suffix)[i];

                        // 遍历该区间内所有数据点
                        for (int j = ql; j <= qr; ++j)
                        {
                            float dis = dis_compute(storage.query_points[i], storage.data_points[j]);
                            ans.emplace(dis, j);
                            if ((int)ans.size() > storage.query_K)
                            {
                                ans.pop(); // 保持堆大小为 K
                            }
                        }
                    }

                    // 收集结果（此时堆顶是最大距离，顺序是从大到小）
                    std::vector<int> results;
                    while (!ans.empty())
                    {
                        results.push_back(ans.top().second);
                        ans.pop();
                    }
                    // 反转得到距离从小到大（通常 ground truth 按距离升序排列）
                    std::reverse(results.begin(), results.end());

                    // 写入文件：K 个 int（点 ID）
                    for (int id : results)
                    {
                        outfile.write(reinterpret_cast<const char *>(&id), sizeof(int));
                    }
                }

                outfile.close();
            }
        }

    };

    class QueryGenerator
    {
    public:
        int data_nb, query_nb;
        hnswlib::L2Space *space;

        QueryGenerator(int data_num, int query_num) : data_nb(data_num), query_nb(query_num) {}
        ~QueryGenerator() {}

        void GenerateRange(std::string saveprefix)
        {
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine e(seed);

            std::vector<std::pair<int, int>> rs;
            int current_len = data_nb;
            for (int i = 0; i < 10; i++)
            {
                if (current_len < 10)
                    throw Exception("dataset size is too small, increase the amount of data objects!");
                rs.emplace_back(current_len, i);
                current_len /= 2;
            }
            for (auto t : rs)
            {
                int len = t.first, suffix = t.second;
                std::string savepath = saveprefix + std::to_string(suffix) + ".bin";
                CheckPath(savepath);
                std::cout << "save query range to" << savepath << std::endl;
                std::ofstream outfile(savepath, std::ios::out | std::ios::binary);
                if (!outfile.is_open())
                    throw Exception("cannot open " + savepath);
                std::uniform_int_distribution<int> u_start(0, data_nb - len);
                for (int i = 0; i < query_nb; i++)
                {
                    int ql = u_start(e);
                    int qr = ql + len - 1;
                    if (ql >= data_nb || qr >= data_nb)
                        throw Exception("Query range out of bound");
                    outfile.write((char *)&ql, sizeof(int));
                    outfile.write((char *)&qr, sizeof(int));
                }
                outfile.close();
            }

            rs.clear();
            current_len = data_nb;
            for (int i = 0; i < 10; i++)
            {
                rs.emplace_back(current_len, i);
                current_len /= 2;
            }
            std::string savepath = saveprefix + "17.bin";
            CheckPath(savepath);
            std::cout << "save query range to" << savepath << std::endl;
            std::ofstream outfile(savepath, std::ios::out | std::ios::binary);
            if (!outfile.is_open())
                throw Exception("cannot open " + savepath);

            for (auto t : rs)
            {
                int len = t.first;
                std::uniform_int_distribution<int> u_start(0, data_nb - len);

                for (int i = 0; i < query_nb / 10; i++)
                {
                    int ql = u_start(e);
                    int qr = ql + len - 1;
                    if (ql >= data_nb || qr >= data_nb)
                        throw Exception("Query range out of bound");
                    outfile.write((char *)&ql, sizeof(int));
                    outfile.write((char *)&qr, sizeof(int));
                }
            }
            outfile.close();
        }

        float dis_compute(std::vector<float> &v1, std::vector<float> &v2)
        {
            hnswlib::DISTFUNC<float> fstdistfunc_ = space->get_dist_func();
            float dis = fstdistfunc_((char *)v1.data(), (char *)v2.data(), space->get_dist_func_param());
            return dis;
        }

        void GenerateGroundtruth(std::string saveprefix, DataLoader &storage)
        {
            space = new hnswlib::L2Space(storage.Dim);
            for (auto t : storage.query_range)
            {
                int suffix = t.first;
                std::string savepath = saveprefix + std::to_string(suffix) + ".bin";
                CheckPath(savepath);
                std::ofstream outfile(savepath, std::ios::out | std::ios::binary);
                if (!outfile.is_open())
                    throw Exception("cannot open " + savepath);
                std::cout << "generating for " << t.first << std::endl;
                for (int i = 0; i < query_nb; i++)
                {
                    auto rp = t.second[i];
                    int ql = rp.first, qr = rp.second;
                    std::priority_queue<std::pair<float, int>> ans;
                    for (int j = ql; j <= qr; j++)
                    {
                        float dis = dis_compute(storage.query_points[i], storage.data_points[j]);
                        ans.emplace(dis, j);
                        if (ans.size() > storage.query_K)
                            ans.pop();
                    }
                    while (ans.size())
                    {
                        auto id = ans.top().second;
                        ans.pop();
                        outfile.write((char *)&id, sizeof(int));
                    }
                }
                outfile.close();
            }
        }
    };

    struct MBR
    {
        int dim = 0;
        std::vector<double> min_vals; // 各维下界
        std::vector<double> max_vals; // 各维上界

        MBR() = default;

        // 指定维度，初始化为"空"（无效）MBR
        explicit MBR(int d) : dim(d)
        {
            min_vals.assign(dim, std::numeric_limits<double>::max());
            max_vals.assign(dim, std::numeric_limits<double>::lowest());
        }

        // 从一对上下界构造（方便从 Attr_Constraint 直接转）
        MBR(const std::vector<std::pair<double, double>> &bounds)
        {
            dim = static_cast<int>(bounds.size());
            min_vals.reserve(dim);
            max_vals.reserve(dim);
            for (auto &b : bounds)
            {
                min_vals.push_back(b.first);
                max_vals.push_back(b.second);
            }
        }

        // 判空：尚未合并过任何有效区域
        bool IsEmpty() const
        {
            if (dim == 0)
                return true;
            for (int d = 0; d < dim; d++)
                if (min_vals[d] > max_vals[d]) // 初始化态必满足
                    return true;
            return false;
        }

        void Merge(const MBR &other)
        {
            if (other.dim == 0)
                return;
            if (dim != other.dim)
                throw Exception("MBR::Merge: dimension mismatch");
            if (dim == 0)
            {
                *this = other;
                return;
            }
            for (int d = 0; d < dim; d++)
            {
                min_vals[d] = std::min(min_vals[d], other.min_vals[d]);
                max_vals[d] = std::max(max_vals[d], other.max_vals[d]);
            }
        }

        bool Intersects(const MBR &other) const
        {
            if (dim == 0 || other.dim == 0)
                return false;
            if (dim != other.dim)
                return false;
            for (int d = 0; d < dim; d++)
            {
                if (min_vals[d] > other.max_vals[d] || other.min_vals[d] > max_vals[d])
                    return false;
            }
            return true;
        }

        // 点是否落在当前 MBR 内（用于属性过滤）
        bool Contains(const std::vector<double> &p) const
        {
            if (dim == 0 || static_cast<int>(p.size()) != dim)
                return false;
            for (int d = 0; d < dim; d++)
            {
                if (p[d] < min_vals[d] || p[d] > max_vals[d])
                    return false;
            }
            return true;
        }

        // 当前 MBR 是否完全包含 other（剪枝用：节点 MBR ⊆ 查询 MBR 时可跳过逐点过滤）
        bool FullyContains(const MBR &other) const
        {
            if (dim == 0 || other.dim == 0)
                return false;
            if (dim != other.dim)
                return false;
            for (int d = 0; d < dim; d++)
            {
                if (min_vals[d] > other.min_vals[d] || max_vals[d] < other.max_vals[d])
                    return false;
            }
            return true;
        }
    };

    class TreeNode
    {
    public:
        int node_id;
        int lbound, rbound;
        int depth;
        int range_id;
        std::vector<TreeNode *> childs;
        TreeNode(int l, int r, int d) : lbound(l), rbound(r), depth(d)
        {
        }
    };

    class SegmentTree
    {
    public:
        int ways_ = 2;
        TreeNode *root{nullptr};
        int max_depth{-1};
        std::vector<TreeNode *> treenodes;

        SegmentTree(int data_nb)
        {
            root = new TreeNode(0, data_nb - 1, 0);
        }

        void BuildTree(TreeNode *u)
        {
            if (u == nullptr)
                throw Exception("Tree node is a nullptr");
            treenodes.emplace_back(u);
            max_depth = std::max(max_depth, u->depth);
            int L = u->lbound, R = u->rbound;
            size_t Len = R - L + 1;
            if (L == R)
            {
                return;
            }
            // if (L == R)
            //     return;
            int gap = (R - L + 1) / ways_;
            int res = (R - L + 1) % ways_;

            for (int l = L; l <= R;)
            {
                int r = l + gap - 1;
                if (res > 0)
                {
                    r++;
                    res--;
                }
                r = std::min(r, R);
                TreeNode *childnode = new TreeNode(l, r, u->depth + 1);
                u->childs.emplace_back(childnode);
                BuildTree(childnode);
                l = r + 1;
            }
        }

        std::vector<TreeNode *> range_filter(TreeNode *u, int ql, int qr)
        {
            if (u->lbound >= ql && u->rbound <= qr)
            {
                return {u};
            }
            std::vector<TreeNode *> res;
            if (u->lbound > qr)
                return res;
            if (u->rbound < ql)
                return res;
            for (auto child : u->childs)
            {
                auto t = range_filter(child, ql, qr);
                while (t.size())
                {
                    res.emplace_back(t.back());
                    t.pop_back();
                }
            }
            return res;
        }
    };

    // 稀疏块位图：只保存非零 64bit 字 (block_idx, word)，按 block_idx 升序。
    // "全都满足"交集语义下位图极稀疏（叶子恰 1 bit、内部节点交集后通常 ≤1 块），
    // 若按值域全宽 dense 存储（如标签 0..100000），仅叶子就需 2.2M×3 张 ~12KB 位图，内存不可行。
    // API（set/test/intersect_with）与 dense 版一致。
    class Bitmap
    {
    public:
        std::vector<std::pair<uint32_t, uint64_t>> blocks; // sorted by first, second != 0

        Bitmap() {}
        explicit Bitmap(int nbits) {} // 兼容原构造调用，稀疏存储无需预分配

        void set(int i)
        {
            uint32_t b = (uint32_t)(i >> 6);
            uint64_t m = 1ULL << (i & 63);
            for (size_t k = 0; k < blocks.size(); k++)
            {
                if (blocks[k].first == b)
                {
                    blocks[k].second |= m;
                    return;
                }
                if (blocks[k].first > b)
                {
                    blocks.insert(blocks.begin() + k, std::make_pair(b, m));
                    return;
                }
            }
            blocks.emplace_back(b, m);
        }

        bool test(int i) const
        {
            if (i < 0)
                return false;
            uint32_t b = (uint32_t)(i >> 6);
            uint64_t m = 1ULL << (i & 63);
            for (auto &kv : blocks)
            {
                if (kv.first == b)
                    return (kv.second & m) != 0ULL;
                if (kv.first > b)
                    return false;
            }
            return false;
        }

        // 判断 this 和 other 是否有交集（任意一个 bit 同时在两个 bitmap 中为 1）
        bool has_intersection(const Bitmap &other) const
        {
            size_t p = 0, q = 0;
            while (p < blocks.size() && q < other.blocks.size())
            {
                if (blocks[p].first < other.blocks[q].first)
                    p++;
                else if (blocks[p].first > other.blocks[q].first)
                    q++;
                else
                {
                    if ((blocks[p].second & other.blocks[q].second) != 0ULL)
                        return true;
                    p++;
                    q++;
                }
            }
            return false;
        }
        // this |= other（并集，双指针归并）
        void union_with(const Bitmap &other)
        {
            std::vector<std::pair<uint32_t, uint64_t>> out;
            out.reserve(blocks.size() + other.blocks.size());
            size_t p = 0, q = 0;
            while (p < blocks.size() && q < other.blocks.size())
            {
                if (blocks[p].first < other.blocks[q].first)
                {
                    out.push_back(blocks[p]);
                    p++;
                }
                else if (blocks[p].first > other.blocks[q].first)
                {
                    out.push_back(other.blocks[q]);
                    q++;
                }
                else
                {
                    uint64_t w = blocks[p].second | other.blocks[q].second;
                    out.emplace_back(blocks[p].first, w);
                    p++;
                    q++;
                }
            }
            while (p < blocks.size())
            {
                out.push_back(blocks[p]);
                p++;
            }
            while (q < other.blocks.size())
            {
                out.push_back(other.blocks[q]);
                q++;
            }
            blocks.swap(out);
        }

        Bitmap &operator|=(const Bitmap &other)
        {
            union_with(other);
            return *this;
        }
        void reset() { blocks.clear(); }
    };

    // 带标签位图的树节点：继承 TreeNode，把 Bitmap 数组挂在派生节点上
    // （模仿 SegmentTreeZorder 在 TreeNode::mbr 上挂 MBR 的思路）
    class TreeNodeLabel
    {
    public:
        int node_id;
        int lbound, rbound;
        int depth;
        int range_id;
        std::vector<TreeNodeLabel *> childs;
        std::vector<Bitmap> bmps;
        static int next_id; // 声明静态成员
        TreeNodeLabel(int l, int r, int d) : lbound(l), rbound(r), depth(d)
        {
            node_id = next_id++;
        }
    };

    int TreeNodeLabel::next_id = 0;
    // 模仿 SegmentTreeZorder（本文件 1276 行）的 SegmentTreeLabel：
    // 树形与 ways_ 切分规则同 SegmentTree（ways_=10），保证层级与图的 layer 导航对齐；
    // 区别是每个节点带 dim 个 Bitmap，记录"当前节点范围的点全都满足哪些标签"。
    class SegmentTreeLabel
    {
    public:
        int ways_ = 2;
        int dim_ = 1;              // 属性列数（标签在第几列由查询侧决定）
        std::vector<int> dim_max_; // 每列属性值域 [0, dim_max_[d])，位图宽度
        TreeNodeLabel *root{nullptr};
        int max_depth{-1};
        std::vector<TreeNodeLabel *> treenodes;

        // attributes: data_nb × dim，attributes[pid][d] = 第 pid 点第 d 列属性（pid 即排序后底库顺序）
        SegmentTreeLabel(int data_nb, const std::vector<int> &attribute, int dim)
            : dim_(dim)
        {
            if (dim_ <= 0)
                throw Exception("SegmentTreeLabel: dim must be positive");
            if ((int)attribute.size() != data_nb)
                throw Exception("SegmentTreeLabel: attribute size " + std::to_string(attribute.size()) +
                                " != data_nb " + std::to_string(data_nb));
            dim_max_.assign(dim_, 0);
            for (int i = 0; i < data_nb; i++)
            {
                dim_max_[0] = std::max(dim_max_[0], attribute[i] + 1);
            }
            root = new TreeNodeLabel(0, data_nb - 1, 0);
            init_node(root);
            BuildTree(root, attribute);
        }

        void init_node(TreeNodeLabel *u)
        {
            u->bmps.clear();
            for (int d = 0; d < dim_; d++)
                u->bmps.emplace_back(dim_max_[d]);
        }

        void BuildTree(TreeNodeLabel *u, const std::vector<int> &attributes)
        {
            if (u == nullptr)
                throw Exception("Tree node is a nullptr");

            treenodes.emplace_back(u);
            max_depth = std::max(max_depth, u->depth);
            int L = u->lbound, R = u->rbound;

            if (L == R)
            {
                // 叶子：范围内只有该点，它每列的属性值就是"全都满足"的标签
                for (int d = 0; d < dim_; d++)
                    u->bmps[d].set(attributes[L]);
                return;
            }

            int gap = (R - L + 1) / ways_;
            int res = (R - L + 1) % ways_;

            for (int l = L; l <= R;)
            {
                int r = l + gap - 1;
                if (res > 0)
                {
                    r++;
                    res--;
                }
                r = std::min(r, R);
                TreeNodeLabel *childnode = new TreeNodeLabel(l, r, u->depth + 1);
                init_node(childnode);
                u->childs.emplace_back(childnode);
                BuildTree(childnode, attributes);
                l = r + 1;
            }

            // 内部节点：位图 = 各子节点位图的逐位与（交集）
            for (int d = 0; d < dim_; d++)
            {
                u->bmps[d].reset();
                for (const auto *child : u->childs)
                {
                    const TreeNodeLabel *cl = static_cast<const TreeNodeLabel *>(child);
                    u->bmps[d] |= cl->bmps[d];
                }
            }
        }

        // 与 SegmentTree / SegmentTreeZorder 同语义：按 pid 区间 [ql,qr] 返回覆盖它的最少节点集合
        std::vector<TreeNodeLabel *> range_filter(TreeNodeLabel *u, int ql, int qr)
        {
            std::vector<TreeNodeLabel *> res;
            if (u == nullptr)
                return res;
            if (u->lbound > qr || u->rbound < ql)
                return res;
            if (u->lbound >= ql && u->rbound <= qr)
            {
                res.push_back(u);
                return res;
            }
            for (auto child : u->childs)
            {
                auto t = range_filter(static_cast<TreeNodeLabel *>(child), ql, qr);
                res.insert(res.end(), t.begin(), t.end());
            }
            return res;
        }


        ~SegmentTreeLabel()
        {
            for (auto *p : treenodes)
                delete p;
            treenodes.clear();
            root = nullptr;
        }
    };
};