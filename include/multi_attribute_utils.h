#pragma once

#include "segment_tree_utils.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

typedef struct Attr_Constraint
{
    std::vector<std::pair<int, int>> attr_constraints;
} Attr_Constraint;
typedef std::pair<float, int> PFI;
namespace iPreGraph_multi
{
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
            if (dim != other.dim)
            {
                throw Exception("MBR::Merge: dimension mismatch");
            }
            for (int d = 0; d < dim; d++)
            {
                min_vals[d] = std::min(min_vals[d], other.min_vals[d]);
                max_vals[d] = std::max(max_vals[d], other.max_vals[d]);
            }
        }

        bool Intersects(const MBR &other) const
        {
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
        MBR mbr;
        int node_id;
        int lbound, rbound;
        int depth;
        int range_id;
        std::vector<TreeNode *> childs;
        TreeNode(int l, int r, int d, int dim) : lbound(l), rbound(r), depth(d), mbr(dim)
        {
        }
    };

    struct Point
    {
        int x;
        int y;
    };

    class SegmentTreeZorder
    {
    public:
        int ways_ = 2;
        int dim_ = 2; // 新增：属性维度，必须与 DataLoader 一致
        TreeNode *root{nullptr};
        int max_depth{-1};
        std::vector<TreeNode *> treenodes;

        // SegmentTreeZorder(int data_nb) { root = new TreeNode(0, data_nb - 1, 0); }
        SegmentTreeZorder(int data_nb, int dim) : dim_(dim)
        {
            root = new TreeNode(0, data_nb - 1, 0, dim);
        }

        // attributes: N×dim 的二维数组（z-order 重排后的顺序），即 attributes[i][d] = 第 i 个点的第 d 维属性
        void BuildTree(TreeNode *u, const std::vector<std::vector<int>> &attributes)
        {
            if (u == nullptr)
                throw Exception("Tree node is a nullptr");
            if (dim_ <= 0)
                throw Exception("SegmentTreeZorder: dim_ not set");

            treenodes.emplace_back(u);
            max_depth = std::max(max_depth, u->depth);
            int L = u->lbound, R = u->rbound;

            if (L == R)
            {
                // 叶子：用该点的 dim 个属性值初始化 MBR（退化为一个点）
                for (int d = 0; d < dim_; d++)
                {
                    double v = static_cast<double>(attributes[L][d]);
                    u->mbr.min_vals[d] = v;
                    u->mbr.max_vals[d] = v;
                }
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
                TreeNode *childnode = new TreeNode(l, r, u->depth + 1, dim_);
                u->childs.emplace_back(childnode);
                BuildTree(childnode, attributes);
                l = r + 1;
            }
            if (L != R)
            {
                for (const auto *child : u->childs)
                    u->mbr.Merge(child->mbr);
            }
        }

        // 按 z-order 区间 [ql, qr] 筛出覆盖该区间的最少树节点集合
        std::vector<TreeNode *> range_filter(TreeNode *u, int ql, int qr)
        {
            std::vector<TreeNode *> res;
            if (u == nullptr)
                return res;
            if (u->lbound > qr || u->rbound < ql)
                return res; // 区间不相交
            if (u->lbound >= ql && u->rbound <= qr)
            {
                res.push_back(u);
                return res;
            } // 完全包含，整段命中

            for (auto child : u->childs)
            {
                auto t = range_filter(child, ql, qr);
                res.insert(res.end(), t.begin(), t.end());
            }
            return res;
        }

        // 可选增强版：在区间过滤基础上再用 MBR 做一次空间剪枝（结果不变，只是返回的节点更少）
        std::vector<TreeNode *> range_filter(TreeNode *u, int ql, int qr, const MBR &qmb)
        {
            std::vector<TreeNode *> res;
            if (u == nullptr)
                return res;
            if (u->lbound > qr || u->rbound < ql)
                return res;
            if (!u->mbr.Intersects(qmb))
                return res; // 新增：空间剪枝
            if (u->lbound >= ql && u->rbound <= qr)
            {
                res.push_back(u);
                return res;
            }

            for (auto child : u->childs)
            {
                auto t = range_filter(child, ql, qr, qmb);
                res.insert(res.end(), t.begin(), t.end());
            }
            return res;
        }

        ~SegmentTreeZorder()
        {
            for (auto *p : treenodes)
                delete p;
            treenodes.clear();
            root = nullptr;
        }
    };
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
    using MultiAttributeZCode = std::vector<std::uint64_t>;

    /**
     * Read all integer values from a flat or nested JSON array.
     *
     * The attribute and range generators both write integer-only JSON arrays.
     * Ignoring array nesting keeps this reader compatible with both the
     * historical flat representation and nested debug files.
     */
    inline std::vector<int> ReadJsonIntegers(const std::string &filename)
    {
        std::ifstream infile(filename);
        if (!infile.is_open())
        {
            throw std::runtime_error("cannot open JSON file: " + filename);
        }

        std::ostringstream buffer;
        buffer << infile.rdbuf();
        const std::string contents = buffer.str();
        const char *cursor = contents.c_str();
        const char *end = cursor + contents.size();
        std::vector<int> values;

        while (cursor < end)
        {
            if (*cursor != '-' &&
                !std::isdigit(static_cast<unsigned char>(*cursor)))
            {
                ++cursor;
                continue;
            }

            errno = 0;
            char *parsed_end = nullptr;
            const long long value = std::strtoll(cursor, &parsed_end, 10);
            if (parsed_end == cursor)
            {
                ++cursor;
                continue;
            }
            if (errno == ERANGE ||
                value < std::numeric_limits<int>::min() ||
                value > std::numeric_limits<int>::max())
            {
                throw std::runtime_error(
                    "integer outside int range in JSON file: " + filename);
            }
            if (parsed_end < end &&
                (*parsed_end == '.' || *parsed_end == 'e' ||
                 *parsed_end == 'E'))
            {
                throw std::runtime_error(
                    "non-integer number in JSON file: " + filename);
            }

            values.push_back(static_cast<int>(value));
            cursor = parsed_end;
        }

        if (values.empty())
        {
            throw std::runtime_error(
                "JSON file contains no integer values: " + filename);
        }
        return values;
    }

    /**
     * Compute an arbitrary-dimensional Morton/Z-order code.
     *
     * Word 0 contains the least-significant 64 bits. Within every coordinate
     * bit plane, attribute 0 is the least-significant bit, matching
     * attrGenerator.py. Multiple uint64 words avoid truncation when
     * attribute_count * bit_width is greater than 64.
     */
    // Interleave non-negative attribute bits into a multiword Morton key.
    // Use this same encoding when sorting base rows and computing query bounds.
    inline MultiAttributeZCode CalculateMultiAttributeZOrder(
        const std::vector<int> &point)
    {
        if (point.empty())
        {
            throw std::invalid_argument(
                "cannot calculate a Z-order code for an empty point");
        }

        std::uint32_t max_value = 0;
        for (const int value : point)
        {
            if (value < 0)
            {
                throw std::invalid_argument(
                    "Z-order attributes must be non-negative");
            }
            max_value = std::max(
                max_value, static_cast<std::uint32_t>(value));
        }

        std::size_t bit_width = 1;
        for (std::uint32_t remaining = max_value; remaining > 1;
             remaining >>= 1)
        {
            ++bit_width;
        }

        const std::size_t dimension = point.size();
        const std::size_t total_bits = bit_width * dimension;
        MultiAttributeZCode code((total_bits + 63) / 64, 0);

        for (std::size_t bit = 0; bit < bit_width; ++bit)
        {
            for (std::size_t attribute = 0; attribute < dimension;
                 ++attribute)
            {
                const std::size_t morton_bit = bit * dimension + attribute;
                const std::uint64_t coordinate_bit =
                    (static_cast<std::uint32_t>(point[attribute]) >> bit) & 1U;
                code[morton_bit / 64] |=
                    coordinate_bit << (morton_bit % 64);
            }
        }
        return code;
    }

    /** Compare arbitrary-width Morton codes as unsigned integers. */
    inline bool MultiAttributeZOrderLess(
        const MultiAttributeZCode &left,
        const MultiAttributeZCode &right)
    {
        const std::size_t word_count = std::max(left.size(), right.size());
        for (std::size_t word = word_count; word-- > 0;)
        {
            const std::uint64_t left_word =
                word < left.size() ? left[word] : 0;
            const std::uint64_t right_word =
                word < right.size() ? right[word] : 0;
            if (left_word != right_word)
            {
                return left_word < right_word;
            }
        }
        return false;
    }

    struct TwoRangeQuery
    {
        int l1, r1, l2, r2;
    };

    class DataLoader
    {
    public:
        int Dim{0}, query_nb{0}, query_K{0};
        std::vector<std::vector<float>> query_points;
        int data_nb{0};
        std::vector<std::vector<float>> data_points;
        std::vector<int> original_id;

        int attr_nb{0};
        std::vector<std::vector<int>> attributes;
        std::vector<int> keys;
        std::vector<int> vids;
        std::vector<std::vector<int>> maxmin;
        std::vector<int> zorder;
        std::vector<int> lexorder;
        std::vector<MultiAttributeZCode> multi_zorder;

        hnswlib::L2Space *space;

        std::unordered_map<std::string, std::vector<Attr_Constraint>> query_range_or;
        std::unordered_map<std::string, std::vector<Attr_Constraint>> query_range_multi;
        std::unordered_map<std::string, std::vector<TwoRangeQuery>> query_range_2d;
        std::unordered_map<std::string, std::vector<std::pair<int, int>>> query_range;
        std::unordered_map<std::string, std::vector<std::vector<int>>> ground_truth;

        DataLoader() {}
        ~DataLoader() {}

        /**
         * Load the flat JSON attributes produced by attrGenerator.py.
         * Layout: [p0_a0, ..., p0_aD, p1_a0, ...].
         */
        // Read row-major integer attributes; verify the requested column count
        // and, when supplied, the expected number of base-vector rows.
        void LoadMultiAttributesJson(
            const std::string &filename,
            std::size_t attribute_count,
            std::size_t expected_data_count = 0)
        {
            if (attribute_count == 0)
            {
                throw std::invalid_argument(
                    "attribute_count must be greater than zero");
            }

            const std::vector<int> values = ReadJsonIntegers(filename);
            if (values.size() % attribute_count != 0)
            {
                throw std::runtime_error(
                    "attribute value count (" +
                    std::to_string(values.size()) +
                    ") is not divisible by attribute_count (" +
                    std::to_string(attribute_count) + ")");
            }

            const std::size_t file_data_count =
                values.size() / attribute_count;
            if (expected_data_count != 0 &&
                file_data_count != expected_data_count)
            {
                throw std::runtime_error(
                    "attribute file contains " +
                    std::to_string(file_data_count) +
                    " points, expected " +
                    std::to_string(expected_data_count));
            }
            if (data_nb > 0 &&
                file_data_count != static_cast<std::size_t>(data_nb))
            {
                throw std::runtime_error(
                    "attribute point count does not match loaded data count");
            }
            if (file_data_count >
                static_cast<std::size_t>(std::numeric_limits<int>::max()))
            {
                throw std::runtime_error("attribute point count exceeds int range");
            }

            data_nb = static_cast<int>(file_data_count);
            attr_nb = static_cast<int>(attribute_count);
            attributes.assign(
                file_data_count, std::vector<int>(attribute_count));
            for (std::size_t point = 0; point < file_data_count; ++point)
            {
                const std::size_t offset = point * attribute_count;
                std::copy_n(
                    values.begin() + offset,
                    attribute_count,
                    attributes[point].begin());
            }
            multi_zorder.clear();
            std::cout << "Loaded " << file_data_count << " points x "
                      << attribute_count << " attributes from " << filename
                      << std::endl;
        }

        /**
         * Load query ranges produced by qrangeGenerator.py.
         * Layout: [q0_a0_low, q0_a0_high, ..., q1_a0_low, ...].
         * Constraints are ANDed across attributes.
         */
        // Read one inclusive [low, high] interval per query and attribute.
        // Attribute constraints are combined with AND by the search driver.
        void LoadMultiRangesJson(
            const std::string &filename,
            std::size_t attribute_count,
            std::size_t expected_query_count = 0,
            const std::string &query_key = "mixed")
        {
            if (attribute_count == 0)
            {
                throw std::invalid_argument(
                    "attribute_count must be greater than zero");
            }
            const std::vector<int> values = ReadJsonIntegers(filename);
            const std::size_t values_per_query = attribute_count * 2;
            if (values.size() % values_per_query != 0)
            {
                throw std::runtime_error(
                    "range value count (" + std::to_string(values.size()) +
                    ") is not divisible by attribute_count * 2 (" +
                    std::to_string(values_per_query) + ")");
            }

            const std::size_t file_query_count =
                values.size() / values_per_query;
            if (expected_query_count != 0 &&
                file_query_count != expected_query_count)
            {
                throw std::runtime_error(
                    "range file contains " +
                    std::to_string(file_query_count) +
                    " queries, expected " +
                    std::to_string(expected_query_count));
            }
            if (query_nb > 0 &&
                file_query_count != static_cast<std::size_t>(query_nb))
            {
                throw std::runtime_error(
                    "range query count does not match loaded query count");
            }
            if (attr_nb > 0 &&
                attribute_count != static_cast<std::size_t>(attr_nb))
            {
                throw std::runtime_error(
                    "range attribute count does not match loaded attributes");
            }
            if (file_query_count >
                static_cast<std::size_t>(std::numeric_limits<int>::max()))
            {
                throw std::runtime_error("range query count exceeds int range");
            }

            query_nb = static_cast<int>(file_query_count);
            std::vector<Attr_Constraint> &ranges =
                query_range_multi[query_key];
            ranges.clear();
            ranges.reserve(file_query_count);
            for (std::size_t query = 0; query < file_query_count; ++query)
            {
                Attr_Constraint constraint;
                constraint.attr_constraints.reserve(attribute_count);
                const std::size_t query_offset = query * values_per_query;
                for (std::size_t attribute = 0;
                     attribute < attribute_count; ++attribute)
                {
                    const std::size_t offset =
                        query_offset + attribute * 2;
                    const int lower = values[offset];
                    const int upper = values[offset + 1];
                    if (lower > upper)
                    {
                        throw std::runtime_error(
                            "invalid range for query " +
                            std::to_string(query) + ", attribute " +
                            std::to_string(attribute) +
                            ": lower bound exceeds upper bound");
                    }
                    constraint.attr_constraints.emplace_back(lower, upper);
                }
                ranges.push_back(std::move(constraint));
            }
            std::cout << "Loaded " << file_query_count << " queries x "
                      << attribute_count << " attribute ranges from "
                      << filename << std::endl;
        }

        /** Calculate one arbitrary-width Morton code for every attribute row. */
        void CalMultiZorder()
        {
            multi_zorder.clear();
            multi_zorder.reserve(attributes.size());
            for (std::size_t point = 0; point < attributes.size(); ++point)
            {
                if (attributes[point].size() !=
                    static_cast<std::size_t>(attr_nb))
                {
                    throw std::runtime_error(
                        "inconsistent attribute count at point " +
                        std::to_string(point));
                }
                multi_zorder.push_back(
                    CalculateMultiAttributeZOrder(attributes[point]));
            }
        }



        float dis_compute(std::vector<float> &v1, std::vector<float> &v2)
        {
            hnswlib::DISTFUNC<float> fstdistfunc_ = space->get_dist_func();
            float dis = fstdistfunc_((char *)v1.data(), (char *)v2.data(), space->get_dist_func_param());
            return dis;
        }

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
            space = new hnswlib::L2Space(Dim);
            infile.close();
        }

        void LoadData(std::string filename)
        {
            std::ifstream infile(filename, std::ios::in | std::ios::binary);
            if (!infile.is_open())
                throw Exception("cannot open " + filename);
            infile.read((char *)&data_nb, sizeof(int));
            infile.read((char *)&Dim, sizeof(int));
            // std::cout << data_nb << std::endl;
            data_points.resize(data_nb);
            for (int i = 0; i < data_nb; i++)
            {
                data_points[i].resize(Dim);
                infile.read((char *)data_points[i].data(), Dim * sizeof(float));
            }
            attributes.resize(data_nb);
            infile.close();
        }





        // void LoadRanges_simple(std::string saveprefix)
        // {
        //     std::string savepath = saveprefix + "mixed.bin";
        //     std::ifstream infile(savepath, std::ios::in | std::ios::binary);
        //     if (!infile.is_open())
        //     {
        //         throw Exception("cannot open " + savepath);
        //     }
        //     for (int k = 0; k < query_nb; k++)
        //     {
        //         int l1, r1, l2, r2;
        //         infile.read((char *)&l1, sizeof(int));
        //         infile.read((char *)&r1, sizeof(int));
        //         infile.read((char *)&l2, sizeof(int));
        //         infile.read((char *)&r2, sizeof(int));
        //         Attr_Constraint t;
        //         t.attr_constraints.emplace_back(l1, r1);
        //         t.attr_constraints.emplace_back(l2, r2);
        //         query_range["mixed"].emplace_back(t);
        //     }
        //     infile.close();
        // }

        /**
         * 加载 OR 查询范围
         * @param file_attr0 属性 0 的范围文件路径 (例如 ranges_x.csv)
         * @param file_attr1 属性 1 的范围文件路径 (例如 ranges_y.csv)
         * @param query_key 存入 map 的键名 (例如 "mixed")
         */





        void LoadGroundtruth(std::string saveprefix)
        {
            for (auto t : query_range_2d)
            {
                std::string domain = t.first;
                std::string savepath = saveprefix + domain + ".bin";
                std::ifstream infile(savepath, std::ios::in | std::ios::binary);
                if (!infile.is_open())
                {
                    throw Exception("cannot open " + savepath);
                }
                ground_truth[domain].resize(query_nb);
                for (int qid = 0; qid < query_nb; qid++)
                {
                    ground_truth[domain][qid].resize(query_K);
                    infile.read((char *)ground_truth[domain][qid].data(), query_K * sizeof(int));
                }
                infile.close();
            }
        }
        /**
         * Load ground truth produced by the generator.
         * Layout (flat integer list): [q0_id0, q0_id1, ..., q0_id(K-1),
         *                              q1_id0, ..., q(N-1)_id(K-1)]
         * One file per domain; the domain name is taken from query_key.
         */
        void LoadGroundtruthJson(const std::string &filename,
                                 const std::string &query_key = "mixed")
        {
            const std::vector<int> values = ReadJsonIntegers(filename);
            if (values.empty())
            {
                throw std::runtime_error("ground truth file is empty: " + filename);
            }
            if (query_K <= 0)
            {
                throw std::runtime_error("query_K must be set before loading ground truth");
            }
            if (query_nb <= 0)
            {
                throw std::runtime_error("query_nb must be set before loading ground truth");
            }

            const std::size_t values_per_query = static_cast<std::size_t>(query_K);
            if (values.size() % values_per_query != 0)
            {
                throw std::runtime_error(
                    "ground truth value count (" + std::to_string(values.size()) +
                    ") is not divisible by query_K (" + std::to_string(values_per_query) + ")");
            }

            const std::size_t file_query_count = values.size() / values_per_query;
            if (file_query_count != static_cast<std::size_t>(query_nb))
            {
                throw std::runtime_error(
                    "ground truth file contains " + std::to_string(file_query_count) +
                    " queries, but query_nb = " + std::to_string(query_nb));
            }
            if (file_query_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            {
                throw std::runtime_error("ground truth query count exceeds int range");
            }

            std::vector<std::vector<int>> &gt = ground_truth[query_key];
            gt.clear();
            gt.resize(file_query_count);

            for (std::size_t qid = 0; qid < file_query_count; ++qid)
            {
                const std::size_t offset = qid * values_per_query;
                gt[qid].assign(values.begin() + static_cast<std::ptrdiff_t>(offset),
                               values.begin() + static_cast<std::ptrdiff_t>(offset + values_per_query));
            }
            std::cout << "Loaded ground truth for \"" << query_key << "\": "
                      << file_query_count << " queries x " << query_K << " ids from "
                      << filename << std::endl;
        }
    };


}
