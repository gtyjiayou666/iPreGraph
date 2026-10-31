#pragma once

#include <vector>
#include <iterator>
#include "multi_attribute_utils.h"
#include "search_containers.h"
#include "memory_utils.h"
#include <bitset>
#include <stack>
#include <cmath>
#include <climits>
#include <algorithm>

// ============================================================================
// search_mixed.h —— 支持"标签 + 数值"混合属性的向量近似近邻检索
//
// 数据组织（与已有 label / zorder 两套方案对齐）：
//   数据点先按标签（label）升序排列；标签相同的点再按数值属性的 z-order 码升序排列，
//   即全局顺序为 (label, z_code) 字典序。因此：
//     - 每个标签在数组中是连续区间（可用 equal_range 定位）；
//     - 每个标签内部 z-code 单调不减。
// 树节点（TreeNodeMixed）维护剪枝信息：
//    1. label_bmp : 节点范围内出现过的标签位图（子节点并集，精确）；
//    2. label_mbrs : 每个出现过的标签各自的数值包围盒（int32 紧凑，按 label 升序），
//       外加联合盒；剪枝时先位图、再联合盒、再逐"命中标签"盒，比单一 MBR 更强。
// 查询语义：标签集合（标签间 OR）+ 数值属性范围（属性间 AND）。
// 查询流程：
//   1) 查询前定候选区间 [l, r]：数据按 (label, z) 全局字典序排序、查询标签集升序，
//      只需处理第一个和最后一个查询标签：
//        l = 最小查询标签块内第一个 z >= q_z_min 的点（块内满足数值约束的最小值）
//        r = 最大查询标签块内最后一个 z <= q_z_max 的点（块内满足数值约束的最大值）
//      块边界由 BuildLabelBlocks 预建（O(1)），块内 z 单调用 STL 定位；
//      角点 z 区间 [q_z_min, q_z_max] 是数值约束的安全近似（数值命中的点 z 必落在其内），
//      故 [l,r] 覆盖全部真值点，非命中点由点级 flags 过滤兜底。
//   2) 检索过程中节点剪枝（均安全，不会剪掉含真值点的节点）：
//        label_bmp ∩ 查询标签位图 ≠ ∅  且  联合盒 ∩ 查询MBR ≠ ∅
//        且  ∃命中标签：该标签的 MBR ∩ 查询MBR ≠ ∅
// ============================================================================

namespace iPreGraph_mixed
{
    // segment_tree_utils.h 中 tableint / linklistsizeint 定义于 iPreGraph 命名空间，这里引入以便裸用
    using iPreGraph::linklistsizeint;
    using iPreGraph::tableint;

    // ------------------------------------------------------------------------
    // 混合属性树节点
    // ------------------------------------------------------------------------
    class TreeNodeMixed
    {
    public:
        int node_id;
        int lbound, rbound;
        int depth;
        int range_id;
        std::vector<TreeNodeMixed *> childs;

        iPreGraph::Bitmap label_bmp; // 节点范围内出现过的标签（并集）

        // 每标签数值包围盒（int32 紧凑，替代原 double MBR；仅存出现过的标签）：
        //   label_mbr_data 扁平存储，每个标签 (1 + 2*z_attr_dim) 个 int：
        //   [label, mn[0..dim), mx[0..dim)]，按 label 升序。
        //   20 标签场景：底层节点 1 个标签（1 条 20B），高层最多 20 条（400B）。
        std::vector<int> label_mbr_data;
        std::vector<int> joint_mn; // 联合盒（全部标签盒的并），第一道快速拒绝
        std::vector<int> joint_mx;

        TreeNodeMixed(int l, int r, int d, int z_attr_dim)
            : lbound(l), rbound(r), depth(d),
              joint_mn(z_attr_dim, std::numeric_limits<int>::max()),
              joint_mx(z_attr_dim, std::numeric_limits<int>::lowest())
        {
        }
    };

    // ------------------------------------------------------------------------
    // 混合属性线段树：树形与 SegmentTree / SegmentTreeZorder 完全一致（ways_=2），
    // 保证层级与图索引的 layer 导航对齐。
    // ------------------------------------------------------------------------
    class SegmentTreeMixed
    {
    public:
        int ways_ = 2;
        int z_attr_dim_ = 2; // 数值属性列数
        int node_count{0};   // 已创建的节点数（同时作为 node_id 分配器）
        TreeNodeMixed *root{nullptr};
        int max_depth{-1};
        std::vector<TreeNodeMixed *> treenodes;

        SegmentTreeMixed(int data_nb, int z_attr_dim)
            : z_attr_dim_(z_attr_dim)
        {
            if (z_attr_dim <= 0)
                throw Exception("SegmentTreeMixed: z_attr_dim must be positive");
            root = new TreeNodeMixed(0, data_nb - 1, 0, z_attr_dim);
            root->node_id = node_count++;
        }

        // 输入数据必须已按 (label, z_code) 排序：
        //   labels : N 个标签，非递减；
        //   z_attrs: N × z_attr_dim 非负整数数值属性（混合排序后的顺序）。
        void BuildTree(TreeNodeMixed *u,
                       const std::vector<int> &labels,
                       const std::vector<std::vector<int>> &z_attrs)
        {
            if (u == nullptr)
                throw Exception("Tree node is a nullptr");
            if (z_attr_dim_ <= 0)
                throw Exception("SegmentTreeMixed: z_attr_dim_ not set");

            treenodes.emplace_back(u);
            max_depth = std::max(max_depth, u->depth);
            int L = u->lbound, R = u->rbound;

            if (L == R)
            {
                // 叶子：范围内只有该点
                u->label_bmp.set(labels[L]);
                u->label_mbr_data.push_back(labels[L]);
                for (int d = 0; d < z_attr_dim_; d++)
                    u->label_mbr_data.push_back(z_attrs[L][d]); // mn[d]
                for (int d = 0; d < z_attr_dim_; d++)
                    u->label_mbr_data.push_back(z_attrs[L][d]); // mx[d]
                for (int d = 0; d < z_attr_dim_; d++)
                {
                    u->joint_mn[d] = z_attrs[L][d];
                    u->joint_mx[d] = z_attrs[L][d];
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
                TreeNodeMixed *childnode = new TreeNodeMixed(l, r, u->depth + 1, z_attr_dim_);
                childnode->node_id = node_count++;
                u->childs.emplace_back(childnode);
                BuildTree(childnode, labels, z_attrs);
                l = r + 1;
            }

            // 内部节点：标签位图取子节点并集；每标签 MBR 按 label 归并合并
            u->label_bmp.reset();
            for (const auto *ch : u->childs)
                u->label_bmp.union_with(ch->label_bmp);
            // 联合盒 = 子节点联合盒合并
            for (int d = 0; d < z_attr_dim_; d++)
            {
                u->joint_mn[d] = std::numeric_limits<int>::max();
                u->joint_mx[d] = std::numeric_limits<int>::lowest();
            }
            for (const auto *ch : u->childs)
            {
                for (int d = 0; d < z_attr_dim_; d++)
                {
                    u->joint_mn[d] = std::min(u->joint_mn[d], ch->joint_mn[d]);
                    u->joint_mx[d] = std::max(u->joint_mx[d], ch->joint_mx[d]);
                }
            }
            // 每标签 MBR：子节点各按 label 升序（ways_=2 恒两个子节点），双指针归并
            const size_t stride = 1 + 2 * (size_t)z_attr_dim_;
            const auto &A = u->childs[0]->label_mbr_data;
            const auto &B = u->childs[1]->label_mbr_data;
            size_t i = 0, j = 0;
            while (i < A.size() || j < B.size())
            {
                int la = (i < A.size()) ? A[i] : std::numeric_limits<int>::max();
                int lb = (j < B.size()) ? B[j] : std::numeric_limits<int>::max();
                if (la < lb)
                {
                    for (size_t k = 0; k < stride; k++)
                        u->label_mbr_data.push_back(A[i + k]);
                    i += stride;
                }
                else if (lb < la)
                {
                    for (size_t k = 0; k < stride; k++)
                        u->label_mbr_data.push_back(B[j + k]);
                    j += stride;
                }
                else
                {
                    u->label_mbr_data.push_back(la);
                    for (int d = 0; d < z_attr_dim_; d++)
                        u->label_mbr_data.push_back(std::min(A[i + 1 + d], B[j + 1 + d]));
                    for (int d = 0; d < z_attr_dim_; d++)
                        u->label_mbr_data.push_back(
                            std::max(A[i + 1 + z_attr_dim_ + d], B[j + 1 + z_attr_dim_ + d]));
                    i += stride;
                    j += stride;
                }
            }
        }

        // 与 SegmentTree / SegmentTreeZorder 同语义：按 pid 区间 [ql,qr] 返回覆盖它的最少节点集合
        std::vector<TreeNodeMixed *> range_filter(TreeNodeMixed *u, int ql, int qr)
        {
            std::vector<TreeNodeMixed *> res;
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
                auto t = range_filter(static_cast<TreeNodeMixed *>(child), ql, qr);
                res.insert(res.end(), t.begin(), t.end());
            }
            return res;
        }

        ~SegmentTreeMixed()
        {
            for (auto *p : treenodes)
                delete p;
            treenodes.clear();
            root = nullptr;
        }
    };

    // ------------------------------------------------------------------------
    // 混合属性数据装载：向量 + 标签 + 数值属性（z-code），以及查询的标签集合与数值范围
    // 约定：data_points / labels / z_attrs / z_codes 均已按 (label, z_code) 排序。
    // ------------------------------------------------------------------------
    // Parse mixed predicates and enforce (label, Morton code) row ordering.
    class MixedDataLoader
    {
    public:
        int Dim{0}, query_nb{0}, query_K{0}, data_nb{0};
        std::vector<std::vector<float>> query_points;
        std::vector<std::vector<float>> data_points;                 // 混合排序后的底库向量
        std::vector<int> labels;                                     // 混合排序后的标签（非递减）
        std::vector<std::vector<int>> z_attrs;                       // 混合排序后的数值属性 N×D
        std::vector<iPreGraph_multi::MultiAttributeZCode> z_codes; // 混合排序后的 z-code
        std::vector<int> label_block_start;                          // label_block_start[label] = 该标签块起点（-1 表示不存在）
        std::vector<int> label_block_end;                            // label_block_end[label]   = 该标签块终点（-1 表示不存在）

        // 每条查询：标签集合（OR）+ 数值范围约束（AND）
        std::vector<std::vector<int>> query_labels;
        std::vector<Attr_Constraint> query_constraints;

        MixedDataLoader() {}
        ~MixedDataLoader() {}

        // ------------------------------------------------------------------
        // 简单 JSON 工具：仅解析本仓库数据文件用到的格式（数组 + 整数），
        // 不做通用 JSON 支持（不含字符串/对象/小数）。
        // ------------------------------------------------------------------
        // 扁平整数数组："[v0, v1, ..., vn-1]"
        static std::vector<int> ParseJsonFlatInts(const std::string &s)
        {
            std::vector<int> out;
            out.reserve(s.size() / 4);
            size_t i = 0;
            const size_t n = s.size();
            while (i < n)
            {
                char c = s[i];
                if (c == '-' || (c >= '0' && c <= '9'))
                {
                    bool neg = (c == '-');
                    if (neg)
                        i++;
                    long long v = 0;
                    while (i < n && s[i] >= '0' && s[i] <= '9')
                    {
                        v = v * 10 + (s[i] - '0');
                        i++;
                    }
                    if (v > 2147483647LL)
                        throw Exception("ParseJsonFlatInts: int overflow");
                    out.push_back(neg ? (int)-v : (int)v);
                }
                else
                {
                    i++;
                }
            }
            return out;
        }

        // 嵌套整数数组解析（仅支持数组 + 整数）：
        //   "[v0, v1, ...]" / "[[..],[..]]" 等任意深度
        struct JsonArr
        {
            std::vector<int> ints;       // 本层直接包含的整数（叶子层）
            std::vector<JsonArr> childs; // 本层嵌套的子数组
        };

        static long long ParseJsonInt(const std::string &s, size_t &i, size_t n)
        {
            bool neg = false;
            if (s[i] == '-')
            {
                neg = true;
                i++;
            }
            long long v = 0;
            while (i < n && s[i] >= '0' && s[i] <= '9')
            {
                v = v * 10 + (s[i] - '0');
                i++;
            }
            return neg ? -v : v;
        }

        // 前提：s[i] == '['
        static JsonArr ParseJsonArr(const std::string &s, size_t &i, size_t n)
        {
            JsonArr out;
            i++; // 跳过 '['
            while (i < n && s[i] != ']')
            {
                if (s[i] == '[')
                {
                    out.childs.push_back(ParseJsonArr(s, i, n));
                }
                else if (s[i] == '-' || (s[i] >= '0' && s[i] <= '9'))
                {
                    long long v = ParseJsonInt(s, i, n);
                    if (v > 2147483647LL || v < -2147483648LL)
                        throw Exception("ParseJsonArr: int out of range");
                    out.ints.push_back((int)v);
                }
                else
                {
                    i++; // 空白 / 逗号等
                }
            }
            if (i < n)
                i++; // 跳过 ']'
            return out;
        }

        static JsonArr ParseJson(const std::string &s)
        {
            size_t i = 0;
            if (i >= s.size() || s[i] != '[')
                throw Exception("ParseJson: expected '[' at start");
            return ParseJsonArr(s, i, s.size());
        }

        // ------------------------------------------------------------------
        // 属性文件加载：扁平 JSON 数组 [l0,a0,b0, l1,a1,b1, ...]（每点 attr_nb 列，
        // 已按 (label, z_code) 排序）。label_col 为标签所在列，其余列为数值属性。
        // 加载后自动计算 z-code 并做排序不变量校验。
        // n_points <= 0 时自动从文件推导（int 总数 / attr_nb，需整除）。
        // ------------------------------------------------------------------
        void LoadAttrJson(const std::string &filename, int n_points, int attr_nb, int label_col)
        {
            std::ifstream in(filename, std::ios::in | std::ios::binary);
            if (!in.is_open())
                throw Exception("cannot open " + filename);
            std::string content((std::istreambuf_iterator<char>(in)),
                                std::istreambuf_iterator<char>());
            in.close();

            std::vector<int> vals = ParseJsonFlatInts(content);
            if (attr_nb <= 0)
                throw Exception("LoadAttrJson: attr_nb must be > 0");
            if (vals.size() % attr_nb != 0)
                throw Exception("LoadAttrJson: int count " + std::to_string(vals.size()) +
                                " not divisible by attr_nb " + std::to_string(attr_nb));
            if (n_points <= 0)
                n_points = (int)(vals.size() / attr_nb);
            else if ((int)vals.size() != n_points * attr_nb)
                throw Exception("LoadAttrJson: expected " + std::to_string(n_points * attr_nb) +
                                " ints, got " + std::to_string(vals.size()));
            if (label_col < 0 || label_col >= attr_nb)
                throw Exception("LoadAttrJson: label_col out of range");

            data_nb = n_points;
            int n_num = attr_nb - 1;
            labels.assign(n_points, 0);
            z_attrs.assign(n_points, std::vector<int>(n_num));
            z_codes.assign(n_points, iPreGraph_multi::MultiAttributeZCode());
            for (int p = 0; p < n_points; p++)
            {
                labels[p] = vals[p * attr_nb + label_col];
                int k = 0;
                for (int c = 0; c < attr_nb; c++)
                {
                    if (c == label_col)
                        continue;
                    int v = vals[p * attr_nb + c];
                    if (v < 0)
                        throw Exception("LoadAttrJson: numeric attrs must be non-negative for z-order");
                    z_attrs[p][k++] = v;
                }
                z_codes[p] = iPreGraph_multi::CalculateMultiAttributeZOrder(z_attrs[p]);
            }
            BuildLabelBlocks();
        }

        // ------------------------------------------------------------------
        // 查询文件加载：[[标签列表, [lo1,hi1], [lo2,hi2], ...], ...]
        // 每条查询 = 标签集合（OR）+ 若干数值范围（AND）。
        // ------------------------------------------------------------------
        void LoadQueryRangeJson(const std::string &filename)
        {
            std::ifstream in(filename, std::ios::in | std::ios::binary);
            if (!in.is_open())
                throw Exception("cannot open " + filename);
            std::string content((std::istreambuf_iterator<char>(in)),
                                std::istreambuf_iterator<char>());
            in.close();

            JsonArr root = ParseJson(content);
            query_nb = (int)root.childs.size();
            query_labels.resize(query_nb);
            query_constraints.resize(query_nb);
            for (int q = 0; q < query_nb; q++)
            {
                const JsonArr &query = root.childs[q]; // 一条查询 = [标签列表, [lo,hi], ...]
                if (query.childs.empty())
                    continue;
                // 第 1 个元素：标签列表
                query_labels[q] = query.childs[0].ints;
                // 其余元素：数值范围 [lo, hi]
                for (size_t k = 1; k < query.childs.size(); k++)
                {
                    const auto &r = query.childs[k].ints;
                    if (r.size() < 2)
                        continue;
                    query_constraints[q].attr_constraints.emplace_back(r[0], r[1]);
                }
            }
            PrintQueryLabelStats();
        }

        // ------------------------------------------------------------------
        // 查询标签统计：打印所有查询中的 label 数量信息
        //   - 查询数、总标签引用数、每条查询标签数 min/max/均值
        //   - 每个标签被引用的查询数（频次）
        // ------------------------------------------------------------------
        void PrintQueryLabelStats() const
        {
            if (query_nb == 0)
                return;
            long long total_refs = 0;
            int perq_min = INT_MAX, perq_max = 0;
            int max_label = -1;
            for (const auto &ql : query_labels)
            {
                total_refs += (long long)ql.size();
                perq_min = std::min(perq_min, (int)ql.size());
                perq_max = std::max(perq_max, (int)ql.size());
                for (int l : ql)
                    max_label = std::max(max_label, l);
            }
            // 直方图：标签数量为 k 的查询数（k = 1..perq_max）
            std::vector<int> hist(perq_max + 1, 0);
            for (const auto &ql : query_labels)
                hist[ql.size()]++;
            std::vector<int> label_freq(max_label + 1, 0);
            for (const auto &ql : query_labels)
                for (int l : ql)
                    if (l >= 0 && l <= max_label)
                        label_freq[l]++;
            printf("query label stats: queries=%d total_label_refs=%lld "
                   "labels_per_query[min=%d max=%d avg=%.2f]\n",
                   query_nb, total_refs, perq_min, perq_max,
                   (double)total_refs / query_nb);
            printf("  label-count histogram (#queries with k labels):");
            for (size_t k = 1; k < hist.size(); k++)
                if (hist[k] > 0)
                    printf(" k=%zu:%d", k, hist[k]);
            printf("\n");
            printf("  label usage freq:");
            for (size_t l = 0; l < label_freq.size(); l++)
                printf(" %d:%d", (int)l, label_freq[l]);
            printf("\n");
        }

        // ---- helpers for label_or: read integer payloads from JSON/text files ----
        // 读取整个文件为字符串
        static std::string fan_read_whole_file(const std::string &filename)
        {
            std::ifstream in(filename);
            if (!in.is_open())
                throw Exception("cannot open " + filename);
            std::stringstream buf;
            buf << in.rdbuf();
            return buf.str();
        }

        // 解析 [[...],[...],[...]] 形式，保留外层分组，内层是整数列表
        static std::vector<std::vector<int>> fan_parse_json_array_of_array(const std::string &filename)
        {
            const std::string s = fan_read_whole_file(filename);
            const std::string who = filename;
            std::vector<std::vector<int>> out;
            size_t i = 0;
            auto skip_ws = [&]()
            {
                while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])))
                    ++i;
            };
            auto read_int = [&]() -> int
            {
                skip_ws();
                size_t start = i;
                if (i < s.size() && (s[i] == '-' || s[i] == '+'))
                    ++i;
                if (i >= s.size() || !std::isdigit(static_cast<unsigned char>(s[i])))
                    throw Exception("JSON parse error in " + who + ": expected integer at offset " + std::to_string(start));
                while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])))
                    ++i;
                std::string token = s.substr(start, i - start);
                errno = 0;
                char *end = nullptr;
                long long v = std::strtoll(token.c_str(), &end, 10);
                if (errno == ERANGE || v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max())
                    throw Exception("JSON parse error in " + who + ": integer out of int32 range: " + token);
                return static_cast<int>(v);
            };

            skip_ws();
            if (i >= s.size() || s[i] != '[')
                throw Exception("JSON parse error in " + who + ": outer structure must be an array");
            ++i;
            skip_ws();
            if (i < s.size() && s[i] == ']') // 顶层为空 []
            {
                ++i;
                return out;
            }
            while (true)
            {
                skip_ws();
                if (i >= s.size() || s[i] != '[')
                    throw Exception("JSON parse error in " + who + ": each query must itself be an array at offset " + std::to_string(i));
                ++i; // inner [
                std::vector<int> cur;
                skip_ws();
                if (i < s.size() && s[i] == ']')
                {
                    ++i; // 允许空标签列表 []
                }
                else
                {
                    while (true)
                    {
                        cur.push_back(read_int());
                        skip_ws();
                        if (i >= s.size())
                            throw Exception("JSON parse error in " + who + ": unexpected end of file");
                        if (s[i] == ',')
                        {
                            ++i;
                            continue;
                        }
                        if (s[i] == ']')
                        {
                            ++i;
                            break;
                        }
                        throw Exception("JSON parse error in " + who + ": expected ',' or ']' at offset " + std::to_string(i));
                    }
                }
                out.push_back(std::move(cur));
                skip_ws();
                if (i >= s.size())
                    throw Exception("JSON parse error in " + who + ": unexpected end, expect ',' or ']'");
                if (s[i] == ',')
                {
                    ++i;
                    continue;
                }
                if (s[i] == ']')
                {
                    ++i;
                    break;
                }
                throw Exception("JSON parse error in " + who + ": expected ',' or ']' at offset " + std::to_string(i));
            }
            return out;
        }

        // 从文本里顺序提取所有整数（用于 label range 文件：label l r 三元组）
        static std::vector<int> fan_parse_all_ints(const std::string &text, const std::string &who)
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

        // ------------------------------------------------------------------
        // 真值文件加载：[[pid,...],[pid,...],...]（每条查询一个 pid 列表，
        // pid 为混合排序后的底库下标，与 search_multi-or-json 的 gt 口径一致）。
        // ------------------------------------------------------------------
        static std::vector<std::vector<int>> LoadGtJson(const std::string &filename, int query_nb, int query_K)
        {
            const std::string s = fan_read_whole_file(filename);
            size_t i = 0;
            while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])))
                ++i;
            bool nested = false;
            if (i < s.size() && s[i] == '[')
            {
                ++i;
                while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])))
                    ++i;
                if (i < s.size() && s[i] == '[')
                    nested = true;
            }

            if (nested)
            {
                std::vector<std::vector<int>> gt = fan_parse_json_array_of_array(filename);
                if ((int)gt.size() != query_nb)
                    throw Exception("ground truth file " + filename + ": " + std::to_string(gt.size()) +
                                    " groups, but query_nb = " + std::to_string(query_nb));
                std::cout << "loaded nested ground truth: " << gt.size() << " groups from " << filename << std::endl;
                return gt;
            }

            if (query_nb <= 0 || query_K <= 0)
                throw Exception("ground truth file " + filename + ": flat format needs valid query_nb and query_K");
            std::vector<int> v = fan_parse_all_ints(s, filename);
            if ((long long)v.size() != (long long)query_nb * query_K)
                throw Exception("ground truth file " + filename + ": " + std::to_string(v.size()) +
                                " integers, expected query_nb * query_K = " +
                                std::to_string((long long)query_nb * query_K) +
                                " (if the file is nested [[...],[...]], the outer groups must start with '[')");
            std::vector<std::vector<int>> gt(query_nb);
            for (int q = 0; q < query_nb; q++)
                gt[q].assign(v.begin() + (size_t)q * query_K, v.begin() + (size_t)(q + 1) * query_K);
            std::cout << "loaded flat ground truth: " << query_nb << " x " << query_K << " from " << filename << std::endl;
            return gt;
        }
        // ------------------------------------------------------------------
        // 标签块边界：数据按 (label, z) 字典序排列，每个标签在数组中是一个
        // 连续块。预先构建 label_block_start/end，检索时 O(1) 取块边界，
        // 无需任何二分。幂等：已构建则直接返回。
        // ------------------------------------------------------------------
        void BuildLabelBlocks()
        {
            if (!label_block_start.empty() || labels.empty())
                return;
            int max_label = labels.back();
            if (max_label < 0)
                return;
            label_block_start.assign(max_label + 1, -1);
            label_block_end.assign(max_label + 1, -1);
            int cur = labels[0];
            int start = 0;
            for (int i = 1; i <= (int)labels.size(); i++)
            {
                if (i == (int)labels.size() || labels[i] != cur)
                {
                    if (cur >= 0 && cur <= max_label)
                    {
                        label_block_start[cur] = start;
                        label_block_end[cur] = i - 1;
                    }
                    if (i == (int)labels.size())
                        break;
                    cur = labels[i];
                    start = i;
                }
            }
        }

        // ------------------------------------------------------------------
        // 排序不变量校验（(label, z_code) 字典序）：
        //   1) 标签非递减；2) 标签内 z-code 非递减。
        // 违反契约将导致候选区间 [l,r] 计算错误，故违反时抛异常。
        // ------------------------------------------------------------------
        void ValidateMixedOrder() const
        {
            const int n = (int)labels.size();
            if (n == 0)
                return;
            if ((int)z_codes.size() != n || (int)z_attrs.size() != n)
                throw Exception("ValidateMixedOrder: labels/z_attrs/z_codes size mismatch");
            for (int i = 1; i < n; i++)
            {
                if (labels[i] < labels[i - 1])
                    throw Exception("ValidateMixedOrder: labels not non-decreasing "
                                    "(data must be sorted by label first)");
                if (labels[i] == labels[i - 1] &&
                    iPreGraph_multi::MultiAttributeZOrderLess(z_codes[i], z_codes[i - 1]))
                    throw Exception("ValidateMixedOrder: z-codes not non-decreasing within a label "
                                    "(data must be z-order sorted within each label)");
            }
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
            infile.close();
        }

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
    };


    // ============================================================================
    // 混合属性 ANN 检索
    // ============================================================================
    template <typename dist_t>
    class iPreGraph_Search_Mixed
    {
    public:
        MixedDataLoader *storage;
        SegmentTreeMixed *tree = nullptr;
        size_t max_elements_{0};
        size_t dim_{0};
        size_t z_attr_dim_{0};
        size_t M_out{0};

        size_t size_data_per_element_{0};
        size_t size_links_per_element_{0};
        size_t data_size_{0};
        size_t size_links_per_layer_{0};
        size_t offsetData_{0};

        char *data_memory_{nullptr};

        hnswlib::L2Space *space;
        hnswlib::DISTFUNC<dist_t> fstdistfunc_;
        void *dist_func_param_{nullptr};

        size_t metric_distance_computations{0};
        size_t metric_hops{0};

        int prefetch_lines{0};

        std::vector<bool> flags; // 点级混合过滤：flags[pid] = 满足全部约束（标签 ∧ 数值）

        iPreGraph_Search_Mixed(std::string vectorfilename, std::string edgefilename,
                                 MixedDataLoader *store, int M)
            : storage(store)
        {
            std::ifstream vectorfile(vectorfilename, std::ios::in | std::ios::binary);
            if (!vectorfile.is_open())
                throw Exception("cannot open " + vectorfilename);
            std::ifstream edgefile(edgefilename, std::ios::in | std::ios::binary);
            if (!edgefile.is_open())
                throw Exception("cannot open " + edgefilename);

            vectorfile.read((char *)&max_elements_, sizeof(int));
            vectorfile.read((char *)&dim_, sizeof(int));
            if ((int)max_elements_ != storage->data_nb)
                throw Exception("vector file count != storage->data_nb");

            z_attr_dim_ = storage->z_attrs.empty() ? 0 : storage->z_attrs[0].size();
            tree = new SegmentTreeMixed((int)max_elements_, (int)z_attr_dim_);
            tree->BuildTree(tree->root, storage->labels, storage->z_attrs);
            // 检索正确性依赖 (label, z_code) 字典序契约，加载后立即校验
            storage->ValidateMixedOrder();

            space = new hnswlib::L2Space(dim_);
            fstdistfunc_ = space->get_dist_func();
            dist_func_param_ = space->get_dist_func_param();
            M_out = M;

            data_size_ = dim_ * sizeof(float);
            size_links_per_layer_ = M_out * sizeof(tableint) + sizeof(linklistsizeint);
            size_links_per_element_ = size_links_per_layer_ * (tree->max_depth + 1);
            size_data_per_element_ = size_links_per_element_ + data_size_;
            offsetData_ = size_links_per_element_;
            prefetch_lines = data_size_ >> 4;

            data_memory_ = (char *)malloc(max_elements_ * size_data_per_element_);
            if (data_memory_ == nullptr)
                throw std::runtime_error("Not enough memory");

            for (int pid = 0; pid < (int)max_elements_; pid++)
            {
                for (int layer = 0; layer <= tree->max_depth; layer++)
                {
                    linklistsizeint *data = get_linklist(pid, layer);
                    edgefile.read((char *)data, sizeof(tableint));
                    int size = getListCount(data);
                    if (size > (int)M_out)
                        throw Exception("real linklist size is bigger than defined M_out");
                    for (int i = 0; i < size; i++)
                    {
                        char *current_neighbor_ = (char *)(data + 1 + i);
                        edgefile.read(current_neighbor_, sizeof(tableint));
                    }
                }
                char *data = getDataByInternalId(pid);
                vectorfile.read(data, dim_ * sizeof(float));
            }

            edgefile.close();
            vectorfile.close();
            flags.assign(max_elements_, false);
            std::cout << "load mixed index finished (max_depth=" << tree->max_depth
                      << ", z_attr_dim=" << z_attr_dim_ << ")" << std::endl;
        }

        ~iPreGraph_Search_Mixed()
        {
            free(data_memory_);
            data_memory_ = nullptr;
            if (tree)
            {
                delete tree;
                tree = nullptr;
            }
        }

        inline char *getDataByInternalId(tableint internal_id) const
        {
            return (data_memory_ + internal_id * size_data_per_element_ + offsetData_);
        }

        linklistsizeint *get_linklist(tableint internal_id, int layer) const
        {
            return (linklistsizeint *)(data_memory_ + internal_id * size_data_per_element_ + layer * size_links_per_layer_);
        }

        int getListCount(linklistsizeint *ptr) const
        {
            return *((int *)ptr);
        }

        int GetOverLap(int l, int r, int ql, int qr)
        {
            int L = std::max(l, ql);
            int R = std::min(r, qr);
            return R - L + 1;
        }


        // 数值剪枝（单区间场景专用）：区间内只对应一个查询标签，标签约束恒真，
        // 跳过 label∩ 判断，仅 joint MBR∩ + 任一标签盒 ∩ wq ≠ ∅。
        // （比 NodePrune 宽松：不按标签过滤盒，仅作为数值维度的剪枝手段）
        inline bool NodePruneNumeric(const TreeNodeMixed *u,
                                     const iPreGraph_multi::MBR &wq) const
        {
            // int nid = u->node_id;
            // uint8_t c = node_cache[nid];
            // if (c != 0)
            //     return c == 1;

            // bool res = false;
            const size_t dim = u->joint_mn.size();
            // bool joint_ok = true;
            for (size_t d = 0; d < dim; d++)
            {
                if (u->joint_mn[d] > (int)wq.max_vals[d] ||
                    (int)wq.min_vals[d] > u->joint_mx[d])
                {
                    return false;
                }
            }
            return true;
            // if (joint_ok)
            // {
            //     const size_t stride = 1 + 2 * dim;
            //     for (size_t i = 0; i + stride <= u->label_mbr_data.size(); i += stride)
            //     {
            //         bool box_ok = true;
            //         for (size_t d = 0; d < dim; d++)
            //         {
            //             int mn = u->label_mbr_data[i + 1 + d];
            //             int mx = u->label_mbr_data[i + 1 + dim + d];
            //             if (mn > (int)wq.max_vals[d] ||
            //                 (int)wq.min_vals[d] > mx)
            //             {
            //                 box_ok = false;
            //                 break;
            //             }
            //         }
            //         if (box_ok)
            //         {
            //             res = true;
            //             break;
            //         }
            //     }
            // }

            // node_cache[nid] = res ? 1 : 2;
            // return res;
        }

        // 点级混合过滤：filter_map[pid] = 满足全部约束（标签命中 ∧ 数值命中）
        inline bool PointPass(int pid,
                              const std::vector<bool> &filter_map) const
        {
            return filter_map[pid];
        }

        // ------------------------------------------------------------------------
        // 边选择：在 (label,z) 混合树上导航，读取 pid 所在深度的邻接表，
        // 仅保留"标签命中 + 数值命中"且未被访问的邻居。
        // ------------------------------------------------------------------------
        std::vector<tableint> SelectEdgeMixed(int pid, int ql, int qr, int edge_limit,
                                              searcher::Bitset<uint64_t> &visited_set,
                                              const iPreGraph_multi::MBR &wq,
                                              const std::vector<bool> &filter_map)
        {
            TreeNodeMixed *cur_node = nullptr, *nxt_node = tree->root;
            std::vector<tableint> selected_edges;
            selected_edges.reserve(edge_limit);
            do
            {
                cur_node = nxt_node;
                bool contain = false;
                // 沿包含 pid 的子树下降，直到 [ql,qr] 交叠不再扩大
                do
                {
                    contain = false;
                    if (cur_node->childs.size() == 0)
                        nxt_node = nullptr;
                    else
                    {
                        for (int i = 0; i < (int)cur_node->childs.size(); ++i)
                        {
                            if (cur_node->childs[i]->lbound <= pid && cur_node->childs[i]->rbound >= pid)
                            {
                                nxt_node = cur_node->childs[i];
                                break;
                            }
                        }
                        if (GetOverLap(cur_node->lbound, cur_node->rbound, ql, qr) ==
                            GetOverLap(nxt_node->lbound, nxt_node->rbound, ql, qr))
                        {
                            cur_node = nxt_node;
                            contain = true;
                        }
                    }
                } while (contain);

                // 当前节点被混合剪枝剪掉 -> 其子树不可能含真值点，直接返回
                if (!NodePruneNumeric(cur_node, wq))
                    return selected_edges;

                // 若兄弟子树与查询无交叠，继续沿 pid 所在子树下降
                do
                {
                    contain = false;
                    if (cur_node->childs.size() == 0)
                        nxt_node = nullptr;
                    else
                    {
                        for (int i = 0; i < (int)cur_node->childs.size(); ++i)
                        {
                            if (cur_node->childs[i]->lbound <= pid && cur_node->childs[i]->rbound >= pid)
                            {
                                nxt_node = cur_node->childs[i];
                            }
                            else
                            {
                                if (!NodePruneNumeric(cur_node->childs[i], wq))
                                    contain = true;
                            }
                        }
                        if (contain)
                            cur_node = nxt_node;
                    }
                } while (contain);

                int *data = (int *)get_linklist(pid, cur_node->depth);
                size_t size = getListCount((linklistsizeint *)data);
                for (size_t j = 1; j <= size; ++j)
                {
                    int neighborId = *(data + j);
                    if (visited_set.get(neighborId))
                        continue;
                    if (!PointPass(neighborId, filter_map))
                        continue;
                    selected_edges.emplace_back(neighborId);
                    if (selected_edges.size() == (size_t)edge_limit)
                        return selected_edges;
                }
            } while (nxt_node);
            return selected_edges;
        }

        // ------------------------------------------------------------------------
        // DFS：沿邻接表向下再探索两层，同样只收集"标签命中 + 数值命中"的点
        // start_node 非空时从该节点继续（跳过从 root 的第一段下降，用于
        // 多区间 DFS 收缩到单区间后就地转交，避免重复扫描）；
        // skip_label_check=true 时仅做数值剪枝（单区间内标签约束恒真）。
        // ------------------------------------------------------------------------
        void DFSMixed(int pid, int ql, int qr, int edge_limit, int deep,
                      std::vector<tableint> &edges,
                      const iPreGraph_multi::MBR &wq,
                      const std::vector<bool> &filter_map)
        {
            TreeNodeMixed *cur_node = tree->root;
            TreeNodeMixed *nxt_node = nullptr;
            bool contain = false;
            do
            {
                contain = false;
                if (cur_node->childs.size() == 0)
                    nxt_node = nullptr;
                else
                {
                    for (int i = 0; i < (int)cur_node->childs.size(); ++i)
                    {
                        if (cur_node->childs[i]->lbound <= pid && cur_node->childs[i]->rbound >= pid)
                        {
                            nxt_node = cur_node->childs[i];
                            break;
                        }
                    }
                    if (GetOverLap(cur_node->lbound, cur_node->rbound, ql, qr) ==
                        GetOverLap(nxt_node->lbound, nxt_node->rbound, ql, qr))
                    {
                        cur_node = nxt_node;
                        contain = true;
                    }
                }
            } while (contain);
            while (cur_node)
            {
                if (!NodePruneNumeric(cur_node, wq))
                    break;

                bool inters = false;
                if (cur_node->childs.size() == 0)
                    nxt_node = nullptr;
                else
                {
                    for (int i = 0; i < (int)cur_node->childs.size(); ++i)
                    {
                        if (cur_node->childs[i]->lbound <= pid && cur_node->childs[i]->rbound >= pid)
                        {
                            nxt_node = cur_node->childs[i];
                        }
                        else
                        {
                            if (NodePruneNumeric(cur_node->childs[i], wq))
                                inters = true;
                        }
                    }
                }

                if (inters)
                {
                    int *data = (int *)get_linklist(pid, cur_node->depth);
                    size_t size = getListCount((linklistsizeint *)data);
                    for (size_t j = 1; j <= size; ++j)
                    {
                        int neighborId = *(data + j);
                        if (PointPass(neighborId, filter_map))
                        {
                            edges.emplace_back(neighborId);
                        }
                        else if (deep > 1)
                        {
                            DFSMixed(neighborId, ql, qr, edge_limit / 2, deep - 1,
                                     edges, wq, filter_map);
                        }
                        if ((int)edges.size() >= edge_limit)
                            return;
                    }
                }
                cur_node = nxt_node;
            }
        }

        // ------------------------------------------------------------------------
        // 顶层搜索：从过滤后的树节点随机取进入点，贪心扩展（HNSW 风格）
        // ------------------------------------------------------------------------
        std::priority_queue<PFI> TopDown_nodeentries_search_mixed(
            std::vector<TreeNodeMixed *> &filterednodes,
            const void *query_data,
            int l, int r, int ef, int query_k, int edge_limit,
            const iPreGraph_multi::MBR &wq,
            const std::vector<bool> &filter_map)
        {
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine e(seed);
            std::priority_queue<PFI, std::vector<PFI>, std::greater<PFI>> candidate_set;
            std::priority_queue<PFI> top_candidates;
            searcher::Bitset<uint64_t> visited_set(max_elements_);

            float lowerBound = std::numeric_limits<float>::max();

            // 进入点：每个过滤节点内随机取一点（仅真值点进入 top_candidates）
            for (auto u : filterednodes)
            {
                std::uniform_int_distribution<int> u_start(u->lbound, u->rbound);
                int pid = u_start(e);
                if (visited_set.get(pid))
                    continue;
                visited_set.set(pid);
                char *ep_data = getDataByInternalId(pid);
                float dis = fstdistfunc_(query_data, ep_data, dist_func_param_);
                ++metric_distance_computations;
                if (top_candidates.size() < ef)
                {
                    candidate_set.emplace(dis, pid);
                    if (PointPass(pid, filter_map))
                    {
                        top_candidates.emplace(dis, pid);
                        lowerBound = top_candidates.top().first;
                    }
                }
                else if (dis < lowerBound)
                {
                    candidate_set.emplace(dis, pid);
                    if (PointPass(pid, filter_map))
                    {
                        top_candidates.emplace(dis, pid);
                        top_candidates.pop();
                        lowerBound = top_candidates.top().first;
                    }
                }
            }

            while (!candidate_set.empty())
            {
                auto current_point_pair = candidate_set.top();
                ++metric_hops;
                if (current_point_pair.first > lowerBound)
                    break;
                candidate_set.pop();
                int current_pid = current_point_pair.second;

                auto selected_edges = SelectEdgeMixed(current_pid, l, r, edge_limit, visited_set,
                                                      wq, filter_map);
                int num_edges = (int)selected_edges.size();
                for (int i = 0; i < std::min(num_edges, 3); ++i)
                {
                    memory::mem_prefetch_L1(getDataByInternalId(selected_edges[i]), this->prefetch_lines);
                }
                for (int i = 0; i < num_edges; ++i)
                {
                    int neighbor_id = selected_edges[i];
                    if (visited_set.get(neighbor_id))
                        continue;
                    visited_set.set(neighbor_id);
                    char *neighbor_data = getDataByInternalId(neighbor_id);
                    float dis = fstdistfunc_(query_data, neighbor_data, dist_func_param_);
                    ++metric_distance_computations;

                    if (top_candidates.size() < ef)
                    {
                        candidate_set.emplace(dis, neighbor_id);
                        top_candidates.emplace(dis, neighbor_id);
                        lowerBound = top_candidates.top().first;
                    }
                    else if (dis < lowerBound)
                    {
                        candidate_set.emplace(dis, neighbor_id);
                        top_candidates.emplace(dis, neighbor_id);
                        top_candidates.pop();
                        lowerBound = top_candidates.top().first;
                    }
                }

                // DFS 补充探索
                std::vector<tableint> edges;
                edges.reserve(edge_limit / 2);
                DFSMixed(current_pid, l, r, edge_limit / 2, 2, edges, wq, filter_map);
                int num_edges1 = (int)edges.size();
                for (int i = 0; i < std::min(num_edges1, 3); ++i)
                {
                    memory::mem_prefetch_L1(getDataByInternalId(edges[i]), this->prefetch_lines);
                }
                for (int i = 0; i < num_edges1; ++i)
                {
                    int neighbor_id = edges[i];
                    if (visited_set.get(neighbor_id))
                        continue;
                    visited_set.set(neighbor_id);
                    char *neighbor_data = getDataByInternalId(neighbor_id);
                    float dis = fstdistfunc_(query_data, neighbor_data, dist_func_param_);
                    ++metric_distance_computations;
                    if (top_candidates.size() < ef)
                    {
                        candidate_set.emplace(dis, neighbor_id);
                        top_candidates.emplace(dis, neighbor_id);
                        lowerBound = top_candidates.top().first;
                    }
                    else if (dis < lowerBound)
                    {
                        candidate_set.emplace(dis, neighbor_id);
                        top_candidates.emplace(dis, neighbor_id);
                        top_candidates.pop();
                        lowerBound = top_candidates.top().first;
                    }
                }
            }

            while ((int)top_candidates.size() > query_k)
                top_candidates.pop();
            return top_candidates;
        }

        // ------------------------------------------------------------------------
        // 以下为"多区间 OR"检索组件，移植自 iRG_search_or.h 的
        //   SelectEdge / DFS_new1 / TopDown_nodeentries_search_or_Pro，
        // 适配 TreeNodeMixed（SelectEdgeMixedOr 保留 NodePrune 树下降剪枝）：
        // 查询预处理得到多个候选区间 Q_L/Q_R（每个查询标签一个区间，即大区间，
        // 无需 MergeSmallGaps 合并），检索时按区间导航，
        // flags（满足全部约束）只过滤最终 top_candidates。
        // ------------------------------------------------------------------------

        // 多区间候选点比较器：针对 iPreGraph_multi::PFIORPRO 的小顶堆
        // （dis 小的优先；segment_tree_utils.h 的 iPreGraph::Compare 接受的是
        //  iPreGraph::PFIORPRO，类型不同，故在本地定义）
        struct ComparePFIORPRO
        {
            bool operator()(const iPreGraph_multi::PFIORPRO &a,
                            const iPreGraph_multi::PFIORPRO &b) const
            {
                return a.dis > b.dis;
            }
        };

        // 边选择（方法同 SelectEdgeMixed，剪枝用数值 MBR）：
        // 沿树下降到 [ql,qr] 内，兄弟子树被数值剪枝剪掉则继续沿 pid 子树下潜，
        // 收集时 PointPass 过滤——只收满足全部约束且未被访问的邻居。
        // 调用场景为单标签区间（主循环传 Q_L[range_id]/Q_R[range_id]，区间内
        // 全是同一标签的点）：标签约束恒真，故树下降剪枝用 NodePruneNumeric
        // （数值 MBR 剪枝，剪掉必不满足数值约束的子树）。
        std::vector<tableint> SelectEdgeMixedOr(int pid, int ql, int qr, int edge_limit,
                                                searcher::Bitset<uint64_t> &visited_set,
                                                const iPreGraph::Bitmap &label_bmp,
                                                const iPreGraph_multi::MBR &wq,
                                                std::vector<uint8_t> &node_cache,
                                                const std::vector<bool> &filter_map)
        {
            TreeNodeMixed *cur_node = nullptr, *nxt_node = tree->root;
            std::vector<tableint> selected_edges;
            selected_edges.reserve(edge_limit);
            do
            {
                cur_node = nxt_node;
                bool contain = false;
                // 沿包含 pid 的子树下降，直到 [ql,qr] 交叠不再扩大
                do
                {
                    contain = false;
                    if (cur_node->childs.size() == 0)
                        nxt_node = nullptr;
                    else
                    {
                        for (int i = 0; i < (int)cur_node->childs.size(); ++i)
                        {
                            if (cur_node->childs[i]->lbound <= pid && cur_node->childs[i]->rbound >= pid)
                            {
                                nxt_node = cur_node->childs[i];
                                break;
                            }
                        }
                        if (GetOverLap(cur_node->lbound, cur_node->rbound, ql, qr) ==
                            GetOverLap(nxt_node->lbound, nxt_node->rbound, ql, qr))
                        {
                            cur_node = nxt_node;
                            contain = true;
                        }
                    }
                } while (contain);

                // 当前节点被数值 MBR 剪枝剪掉 -> 其子树不可能含真值点，直接返回
                if (!NodePruneNumeric(cur_node, wq))
                    return selected_edges;

                // 若兄弟子树与查询数值范围无交叠，继续沿 pid 所在子树下降
                do
                {
                    contain = false;
                    if (cur_node->childs.size() == 0)
                        nxt_node = nullptr;
                    else
                    {
                        for (int i = 0; i < (int)cur_node->childs.size(); ++i)
                        {
                            if (cur_node->childs[i]->lbound <= pid && cur_node->childs[i]->rbound >= pid)
                            {
                                nxt_node = cur_node->childs[i];
                            }
                            else
                            {
                                if (!NodePruneNumeric(cur_node->childs[i], wq))
                                    contain = true;
                            }
                        }
                        if (contain)
                            cur_node = nxt_node;
                    }
                } while (contain);

                int *data = (int *)get_linklist(pid, cur_node->depth);
                size_t size = getListCount((linklistsizeint *)data);
                for (size_t j = 1; j <= size; ++j)
                {
                    int neighborId = *(data + j);
                    if (visited_set.get(neighborId))
                        continue;
                    if (!PointPass(neighborId, filter_map))
                        continue;
                    selected_edges.emplace_back(neighborId);
                    if (selected_edges.size() == (size_t)edge_limit)
                        return selected_edges;
                }
            } while (nxt_node);
            return selected_edges;
        }

        std::vector<tableint> SelectEdge(int pid, int ql, int qr, int edge_limit, searcher::Bitset<uint64_t> &visited_set)
        {
            TreeNodeMixed *cur_node = nullptr, *nxt_node = tree->root;
            std::vector<tableint> selected_edges;
            selected_edges.reserve(edge_limit);
            do
            {
                cur_node = nxt_node;
                bool contain = false;
                do
                {
                    contain = false;
                    if (cur_node->childs.size() == 0)
                        nxt_node = nullptr;
                    else
                    {
                        for (int i = 0; i < cur_node->childs.size(); ++i)
                        {
                            if (cur_node->childs[i]->lbound <= pid && cur_node->childs[i]->rbound >= pid)
                            {
                                nxt_node = cur_node->childs[i];
                                break;
                            }
                        }
                        if (GetOverLap(cur_node->lbound, cur_node->rbound, ql, qr) == GetOverLap(nxt_node->lbound, nxt_node->rbound, ql, qr))
                        {
                            cur_node = nxt_node;
                            contain = true;
                        }
                    }
                } while (contain);

                int *data = (int *)get_linklist(pid, cur_node->depth);
                size_t size = getListCount((linklistsizeint *)data);

                for (size_t j = 1; j <= size; ++j)
                {
                    int neighborId = *(data + j);
                    if (neighborId < ql || neighborId > qr)
                        continue;
                    // if (visitedpool[neighborId] == visited_tag)
                    //     continue;
                    if (visited_set.get(neighborId))
                        continue;
                    selected_edges.emplace_back(neighborId);
                    if (selected_edges.size() == edge_limit)
                        return selected_edges;
                }

            } while (cur_node->lbound < ql || cur_node->rbound > qr);

            return selected_edges;
        }

        std::priority_queue<PFI> TopDown_nodeentries_search_filter(std::vector<TreeNodeMixed *> &filterednodes, const void *query_data, int l, int r, int ef, int query_k, int edge_limit)
        {
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine e(seed);
            std::priority_queue<PFI, std::vector<PFI>, std::greater<PFI>> candidate_set;
            std::priority_queue<PFI> top_candidates;
            searcher::Bitset<uint64_t> visited_set(max_elements_);

            float lowerBound = std::numeric_limits<float>::max();
            for (auto u : filterednodes)
            {
                std::uniform_int_distribution<int> u_start(u->lbound, u->rbound);
                int pid = u_start(e);
                if (visited_set.get(pid))
                {
                    continue;
                }
                visited_set.set(pid);
                char *ep_data = getDataByInternalId(pid);
                float dis = fstdistfunc_(query_data, ep_data, dist_func_param_);
                if (top_candidates.size() < ef)
                {
                    candidate_set.emplace(dis, pid);
                    if (flags[pid])
                    {
                        top_candidates.emplace(dis, pid);
                        lowerBound = top_candidates.top().first;
                    }
                }
                else if (dis < lowerBound)
                {
                    candidate_set.emplace(dis, pid);
                    if (flags[pid])
                    {
                        top_candidates.emplace(dis, pid);
                        top_candidates.pop();
                        lowerBound = top_candidates.top().first;
                    }
                }
            }
            while (!candidate_set.empty())
            {
                auto current_point_pair = candidate_set.top();
                ++metric_hops;
                if (current_point_pair.first > lowerBound)
                    break;
                candidate_set.pop();
                int current_pid = current_point_pair.second;
                auto selected_edges = SelectEdge(current_pid, l, r, edge_limit, visited_set);
                int num_edges = selected_edges.size();
                for (int i = 0; i < std::min(num_edges, 3); ++i)
                {
                    memory::mem_prefetch_L1(getDataByInternalId(selected_edges[i]), this->prefetch_lines);
                }
                for (int i = 0; i < num_edges; ++i)
                {
                    int neighbor_id = selected_edges[i];
                    if (visited_set.get(neighbor_id))
                        continue;
                    visited_set.set(neighbor_id);
                    char *neighbor_data = getDataByInternalId(neighbor_id);
                    float dis = fstdistfunc_(query_data, neighbor_data, dist_func_param_);
                    ++metric_distance_computations;

                    if (top_candidates.size() < ef)
                    {
                        candidate_set.emplace(dis, neighbor_id);
                        if (flags[neighbor_id])
                        {
                            top_candidates.emplace(dis, neighbor_id);
                            lowerBound = top_candidates.top().first;
                        }
                    }
                    else if (dis < lowerBound)
                    {
                        candidate_set.emplace(dis, neighbor_id);
                        if (flags[neighbor_id])
                        {
                            top_candidates.emplace(dis, neighbor_id);
                            top_candidates.pop();
                            lowerBound = top_candidates.top().first;
                        }
                    }
                }
            }
            while (top_candidates.size() > query_k)
                top_candidates.pop();
            return top_candidates;
        }
        // 多区间 DFS（移植 or_Pro DFS_new1）：以全部区间联合范围 [QL[0], QR[last]]
        // 导航，flags=true 的邻居收进 edges，否则递归（deep-1）；随下潜用二分
        // 收缩当前节点覆盖的区间子集 [left, right]。
        // single_mode=true：区间已收缩到只剩一个（单标签块），该区间内只对应
        // 一个查询标签、标签约束恒真——节点剪枝仅用数值（NodePruneNumeric），
        // 收集改为 DFSMixed 风格（存在被剪兄弟子树才收集）。递归始终调用本函数
        // （deep-1）并传播 single_mode，保证递归逻辑统一、不切换到其他函数。
        void DFSMixedOr(std::vector<int> &QL, std::vector<int> &QR, int pid,
                        int edge_limit, int deep,
                        std::vector<tableint> &edges,
                        const iPreGraph::Bitmap &label_bmp,
                        const iPreGraph_multi::MBR &wq,
                        std::vector<uint8_t> &node_cache,
                        const std::vector<bool> &filter_map,
                        bool single_mode = false)
        {
            int Qsize = (int)QL.size();
            if (deep == 0 || Qsize <= 1)
                return;
            int start_idx = 0;
            int end_idx = Qsize - 1;
            int left = QL[0];
            int right = QR[end_idx];
            TreeNodeMixed *cur_node = nullptr;
            TreeNodeMixed *nxt_node = nullptr;
            cur_node = tree->root;
            bool contain = false;
            while (cur_node)
            {
                // 多区间模式：快速下降（进入子节点不改变联合 overlap 则继续下潜）
                do
                {
                    contain = false;
                    if (cur_node->childs.size() == 0)
                        nxt_node = nullptr;
                    else
                    {
                        for (int i = 0; i < (int)cur_node->childs.size(); ++i)
                        {
                            if (cur_node->childs[i]->lbound <= pid && cur_node->childs[i]->rbound >= pid)
                            {
                                nxt_node = cur_node->childs[i];
                                break;
                            }
                        }
                        if (GetOverLap(cur_node->lbound, cur_node->rbound, left, right) ==
                            GetOverLap(nxt_node->lbound, nxt_node->rbound, left, right))
                        {
                            cur_node = nxt_node;
                            contain = true;
                        }
                    }
                } while (contain);

                if (single_mode)
                {
                    // 单区间模式（while 逐层，仿 DFSMixed）：数值剪枝当前节点，
                    // 被剪兄弟子树（inters）才收集；区间为单标签块、标签恒真，
                    // 不判断标签；从当前节点继续，不重复从 root 扫描。
                    if (!NodePruneNumeric(cur_node, wq))
                        break;
                    bool inters = false;
                    if (cur_node->childs.size() == 0)
                        nxt_node = nullptr;
                    else
                    {
                        for (int i = 0; i < (int)cur_node->childs.size(); ++i)
                        {
                            if (cur_node->childs[i]->lbound <= pid && cur_node->childs[i]->rbound >= pid)
                            {
                                nxt_node = cur_node->childs[i];
                            }
                            else if (NodePruneNumeric(cur_node->childs[i], wq))
                            {
                                inters = true;
                            }
                        }
                    }
                    if (inters)
                    {
                        int *data = (int *)get_linklist(pid, cur_node->depth);
                        size_t size = getListCount((linklistsizeint *)data);
                        for (size_t j = 1; j <= size; ++j)
                        {
                            int neighborId = *(data + j);
                            if (neighborId < left || neighborId > right)
                                continue;
                            if (PointPass(neighborId, filter_map))
                            {
                                edges.emplace_back(neighborId);
                            }
                            else if (deep > 1)
                            {
                                // 递归只面对该单区间（避免重新从联合范围扫描）
                                DFSMixedOr(QL, QR, neighborId, edge_limit, deep - 1,
                                           edges, label_bmp, wq, node_cache, filter_map);
                            }
                            if ((int)edges.size() >= edge_limit)
                                return;
                        }
                    }
                    if (nxt_node == nullptr)
                        return;
                    cur_node = nxt_node;
                    continue;
                }
                else
                {
                    int *data = (int *)get_linklist(pid, cur_node->depth);
                    size_t size = getListCount((linklistsizeint *)data);
                    for (size_t j = 1; j <= size; ++j)
                    {
                        int neighborId = *(data + j);
                        if (neighborId < left || neighborId > right)
                            continue;
                        if (PointPass(neighborId, filter_map))
                        {
                            edges.emplace_back(neighborId);
                        }
                        else if (deep > 1)
                        {
                            DFSMixedOr(QL, QR, neighborId, edge_limit, deep - 1,
                                       edges, label_bmp, wq, node_cache, filter_map);
                        }
                        if ((int)edges.size() >= edge_limit)
                            return;
                    }
                    if (nxt_node == nullptr)
                        return;
                    cur_node = nxt_node;
                    int low = start_idx;
                    int high = end_idx;
                    start_idx = end_idx + 1;
                    while (low <= high)
                    {
                        int mid = low + ((high - low) >> 1);
                        if (QR[mid] >= cur_node->lbound)
                        {
                            start_idx = mid;
                            high = mid - 1;
                        }
                        else
                        {
                            low = mid + 1;
                        }
                    }
                    if (start_idx > end_idx)
                        return;
                    low = start_idx;
                    high = end_idx;
                    end_idx = start_idx;
                    while (low <= high)
                    {
                        int mid = low + ((high - low) >> 1);
                        if (QL[mid] > cur_node->rbound)
                        {
                            high = mid - 1;
                        }
                        else
                        {
                            end_idx = mid;
                            low = mid + 1;
                        }
                    }
                    left = QL[start_idx];
                    right = QR[end_idx];
                    if (start_idx == end_idx)
                    {
                        // 收缩到只剩一个区间（单标签块）：进入单区间模式——
                        // 区间内只对应一个查询标签，标签约束恒真，跳过标签判断，
                        // 仅数值剪枝。不调用其他函数：循环继续、递归仍为本函数
                        // （模式传播），当前下降位置不重复从 root 扫描。
                        single_mode = true;
                    }
                }
            }
        }

        // 多区间顶层搜索（移植 or_Pro TopDown_nodeentries_search_or_Pro，无 merge_small_gaps）：
        //   - Q_L/Q_R 即大区间（每查询标签一个，查询预处理直接得到，无需再合并）
        //   - 从 filterednodes（最大区间 range_filter 结果）随机取进入点
        //   - 候选点按其所属区间 [Q_L[range_id], Q_R[range_id]] 导航取边
        //   - 仅 flags=true 的点进入 top_candidates（结果过滤）
        //   - 多区间时用 DFSMixedOr 跨区间补充探索
        // id: 最大区间在 Q_L/Q_R 中的索引（range_filter 的来源）。
        std::priority_queue<PFI> TopDown_nodeentries_search_mixed_or(
            std::vector<TreeNodeMixed *> &filterednodes,
            const void *query_data,
            std::vector<int> &Q_L, std::vector<int> &Q_R,
            int ef, int query_k, int edge_limit, int id,
            const iPreGraph::Bitmap &label_bmp,
            const iPreGraph_multi::MBR &wq,
            std::vector<uint8_t> &node_cache,
            const std::vector<bool> &filter_map)
        {
            int range_num = (int)Q_L.size();

            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine e(seed);

            std::priority_queue<iPreGraph_multi::PFIORPRO,
                                std::vector<iPreGraph_multi::PFIORPRO>,
                                ComparePFIORPRO>
                candidate_set;
            std::priority_queue<PFI> top_candidates;
            searcher::Bitset<uint64_t> visited_set(max_elements_);

            float lowerBound = std::numeric_limits<float>::max();
            for (auto u : filterednodes)
            {
                std::uniform_int_distribution<int> u_start(u->lbound, u->rbound);
                int pid = u_start(e);
                if (visited_set.get(pid))
                    continue;
                visited_set.set(pid);
                char *ep_data = getDataByInternalId(pid);
                float dis = fstdistfunc_(query_data, ep_data, dist_func_param_);
                ++metric_distance_computations;
                if (top_candidates.size() < ef)
                {
                    candidate_set.emplace(dis, pid, id);
                    if (PointPass(pid, filter_map))
                    {
                        top_candidates.emplace(dis, pid);
                        lowerBound = top_candidates.top().first;
                    }
                }
                else if (dis < lowerBound)
                {
                    candidate_set.emplace(dis, pid, id);
                    if (PointPass(pid, filter_map))
                    {
                        top_candidates.emplace(dis, pid);
                        top_candidates.pop();
                        lowerBound = top_candidates.top().first;
                    }
                }
            }
            bool single_mode = false;
            if (range_num == 1)
            {
                single_mode = true;
            }

            while (!candidate_set.empty())
            {
                auto current_point_pair = candidate_set.top();
                ++metric_hops;
                if (current_point_pair.dis > lowerBound)
                    break;
                candidate_set.pop();
                int current_pid = current_point_pair.id;
                int range_id = current_point_pair.range_id;
                if (range_id == -1)
                {
                    // 用 lower_bound 定位 current_pid 属于哪个区间
                    auto it = std::lower_bound(Q_L.begin(), Q_L.end(), current_pid);
                    if (it != Q_L.end())
                    {
                        int i = (int)std::distance(Q_L.begin(), it);
                        range_id = (*it == current_pid) ? i : i - 1;
                    }
                    else
                    {
                        range_id = range_num - 1;
                    }
                }
                auto selected_edges = SelectEdgeMixedOr(current_pid, Q_L[range_id],
                                                        Q_R[range_id], edge_limit,
                                                        visited_set, label_bmp, wq,
                                                        node_cache, filter_map);
                int num_edges = (int)selected_edges.size();
                for (int i = 0; i < std::min(num_edges, 3); ++i)
                {
                    memory::mem_prefetch_L1(getDataByInternalId(selected_edges[i]), this->prefetch_lines);
                }
                for (int i = 0; i < num_edges; ++i)
                {
                    int neighbor_id = selected_edges[i];
                    if (visited_set.get(neighbor_id))
                        continue;
                    visited_set.set(neighbor_id);
                    char *neighbor_data = getDataByInternalId(neighbor_id);
                    float dis = fstdistfunc_(query_data, neighbor_data, dist_func_param_);
                    ++metric_distance_computations;
                    if (top_candidates.size() < ef)
                    {
                        candidate_set.emplace(dis, neighbor_id, range_id);
                        // if (PointPass(neighbor_id, filter_map))
                        // {
                        top_candidates.emplace(dis, neighbor_id);
                        lowerBound = top_candidates.top().first;
                        // }
                    }
                    else if (dis < lowerBound)
                    {
                        candidate_set.emplace(dis, neighbor_id, range_id);
                        // if (PointPass(neighbor_id, filter_map))
                        // {
                        top_candidates.emplace(dis, neighbor_id);
                        top_candidates.pop();
                        lowerBound = top_candidates.top().first;
                        // }
                    }
                }

                // if (range_num == 1)
                //     continue;
                std::vector<tableint> edges;
                edges.reserve(edge_limit / 2);
                // DFS 以 Q_L/Q_R（大区间，每查询标签一个）为输入做跨区间补充探索
                DFSMixedOr(Q_L, Q_R, current_pid, edge_limit / 2, 2, edges,
                           label_bmp, wq, node_cache, filter_map, single_mode);
                int num_edges1 = (int)edges.size();
                for (int i = 0; i < std::min(num_edges1, 3); ++i)
                {
                    memory::mem_prefetch_L1(getDataByInternalId(edges[i]), this->prefetch_lines);
                }
                for (int i = 0; i < num_edges1; ++i)
                {
                    int neighbor_id = edges[i];
                    if (visited_set.get(neighbor_id))
                        continue;
                    visited_set.set(neighbor_id);
                    char *neighbor_data = getDataByInternalId(neighbor_id);
                    float dis = fstdistfunc_(query_data, neighbor_data, dist_func_param_);
                    ++metric_distance_computations;
                    if (top_candidates.size() < ef)
                    {
                        candidate_set.emplace(dis, neighbor_id, -1);
                        top_candidates.emplace(dis, neighbor_id);
                        lowerBound = top_candidates.top().first;
                    }
                    else if (dis < lowerBound)
                    {
                        candidate_set.emplace(dis, neighbor_id, -1);
                        top_candidates.emplace(dis, neighbor_id);
                        top_candidates.pop();
                        lowerBound = top_candidates.top().first;
                    }
                }
            }

            while ((int)top_candidates.size() > query_k)
                top_candidates.pop();
            return top_candidates;
        }

        // ------------------------------------------------------------------------
        // 检索入口：逐查询执行"标签集合 OR + 数值范围 AND"的近似最近邻搜索
        // gt: [query][k] 真值（pid，混合排序后的下标）；为空则 recall 输出 -1
        // 输出 CSV: ef,recall,qps,dco,hop
        // ------------------------------------------------------------------------
        void search_mixed(std::vector<int> &SearchEF, std::string saveprefix, int edge_limit,
                          const std::vector<std::vector<int>> &gt = {})
        {
            const int N = storage->data_nb;

            std::string savepath = saveprefix + "_mixed.csv";
            std::cout << "Savepath: " << savepath << std::endl;
            CheckPath(savepath);
            std::ofstream outfile(savepath);
            if (!outfile.is_open())
                throw Exception("cannot open " + savepath);

            if (flags.size() < (size_t)N)
                flags.assign(N, false);
            storage->BuildLabelBlocks(); // 标签块边界（幂等），检索时 O(1) 取 [l,r]

            for (auto ef : SearchEF)
            {
                int tp = 0;
                int valid_cnt = 0;
                int total_cnt = 0;
                float searchtime = 0;
                metric_hops = 0;
                metric_distance_computations = 0;

                for (int i = 0; i < storage->query_nb; i++)
                {
                    // 1. 标签集合：排序去重
                    std::vector<int> labels = storage->query_labels[i];
                    std::sort(labels.begin(), labels.end());
                    labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
                    if (labels.empty())
                        continue;

                    // 2. 查询标签位图
                    iPreGraph::Bitmap label_bitmap;
                    for (int label : labels)
                    {
                        if (label < 0)
                            continue;
                        label_bitmap.set(label);
                    }
                    if (label_bitmap.blocks.empty())
                        continue;

                    // 3. 数值范围：查询 MBR（节点剪枝 联合盒/每标签盒 ∩ 查询MBR 用）+ 角点 z 码
                    //    （箱角性质：盒内任意点的 z 码落在角点 z 区间 [q_z_min, q_z_max] 内，
                    //      用于第 6 步在最小/最大标签块内收窄 [l, r] 到满足数值约束的范围）
                    const auto &cons = storage->query_constraints[i];
                    int num_attrs = (int)cons.attr_constraints.size();
                    iPreGraph_multi::MBR wq(num_attrs);
                    std::vector<int> coords_min(num_attrs), coords_max(num_attrs);
                    for (int d = 0; d < num_attrs; d++)
                    {
                        wq.min_vals[d] = cons.attr_constraints[d].first;
                        wq.max_vals[d] = cons.attr_constraints[d].second;
                        coords_min[d] = cons.attr_constraints[d].first;
                        coords_max[d] = cons.attr_constraints[d].second;
                    }
                    iPreGraph_multi::MultiAttributeZCode q_z_min =
                        iPreGraph_multi::CalculateMultiAttributeZOrder(coords_min);
                    iPreGraph_multi::MultiAttributeZCode q_z_max =
                        iPreGraph_multi::CalculateMultiAttributeZOrder(coords_max);

                    // 4. 点级混合过滤（单一数组 flags：满足全部约束 = true）：
                    //    先按标签块置 true（标签命中），再在同一数组上做数值 AND。
                    std::fill(flags.begin(), flags.end(), false);
                    for (int label : labels)
                    {
                        auto pp = std::equal_range(storage->labels.begin(), storage->labels.end(), label);
                        if (pp.first == pp.second)
                            continue;
                        int cur_l = (int)(pp.first - storage->labels.begin());
                        int cur_r = (int)(pp.second - storage->labels.begin()) - 1;
                        for (int xb = cur_l; xb <= cur_r; xb++)
                        {
                            bool in_range = true;
                            for (int d = 0; d < num_attrs; d++)
                            {
                                int v = storage->z_attrs[xb][d];
                                if (v < cons.attr_constraints[d].first || v > cons.attr_constraints[d].second)
                                {
                                    in_range = false;
                                    break;
                                }
                            }
                            flags[xb] = in_range;
                        }
                    }

                    // 6. 查询前定候选区间列表 [Q_L, Q_R]（每个查询标签一个区间）：
                    //    数据按 (label, z) 全局字典序排序、查询标签集升序，
                    //    对每个查询标签 lab：
                    //      l_lab = lab 块内第一个 z >= q_z_min 的点（块边界 O(1) + 块内二分）
                    //      r_lab = lab 块内最后一个 z <= q_z_max 的点
                    //    （角点 z 区间 [q_z_min, q_z_max] 是数值约束的安全近似：数值命中的
                    //      点 z 必落在其内，故每区间覆盖对应标签块内全部真值点；
                    //      多区间精确到标签，中间非查询标签块不进入候选范围。）
                    timeval t1, t2;
                    gettimeofday(&t1, NULL);
                    std::vector<int> Q_L, Q_R;
                    for (int lab : labels)
                    {
                        if (lab < 0 || lab >= (int)storage->label_block_start.size())
                            continue;
                        int bl = storage->label_block_start[lab];
                        int br = storage->label_block_end[lab];
                        if (bl < 0 || br < bl)
                            continue;
                        int ll = (int)(std::lower_bound(storage->z_codes.begin() + bl,
                                                        storage->z_codes.begin() + br + 1, q_z_min,
                                                        iPreGraph_multi::MultiAttributeZOrderLess) -
                                       storage->z_codes.begin());
                        int rr = (int)(std::upper_bound(storage->z_codes.begin() + bl,
                                                        storage->z_codes.begin() + br + 1, q_z_max,
                                                        iPreGraph_multi::MultiAttributeZOrderLess) -
                                       storage->z_codes.begin()) -
                                 1;
                        if (ll <= rr)
                        {
                            Q_L.push_back(ll);
                            Q_R.push_back(rr);
                        }
                    }
                    // A query with no matching label block or Morton interval
                    // has no candidates; do not index the empty interval arrays.
                    if (Q_L.empty())
                    {
                        gettimeofday(&t2, NULL);
                        searchtime += GetTime(t1, t2);
                        continue;
                    }
                    int id = 0, max_len = 0;
                    for (size_t k = 0; k < Q_L.size(); k++)
                    {
                        if (Q_R[k] - Q_L[k] + 1 > max_len)
                        {
                            max_len = Q_R[k] - Q_L[k] + 1;
                            id = (int)k;
                        }
                    }
                    std::vector<TreeNodeMixed *> node = tree->range_filter(tree->root, Q_L[id], Q_R[id]);
                    std::vector<uint8_t> node_cache((size_t)tree->node_count, 0);

                    std::priority_queue<PFI> res = TopDown_nodeentries_search_mixed_or(
                        node, storage->query_points[i].data(), Q_L, Q_R, ef,
                        storage->query_K, edge_limit, id,
                        label_bitmap, wq,
                        node_cache, flags);
                    gettimeofday(&t2, NULL);
                    searchtime += GetTime(t1, t2);

                    std::map<int, int> record;
                    while (!res.empty())
                    {
                        auto x = res.top().second;
                        res.pop();
                        if (record.count(x))
                            continue;
                        record[x] = 1;
                        total_cnt++;
                        if (PointPass(x, flags))
                            valid_cnt++;
                        if (!gt.empty())
                        {
                            if (std::find(gt[i].begin(), gt[i].end(), x) != gt[i].end())
                                tp++;
                        }
                    }
                }

                float recall = gt.empty() ? -1.0f
                                          : 1.0f * tp / storage->query_nb / storage->query_K;
                float qps = storage->query_nb / searchtime;
                float dco = metric_distance_computations * 1.0f / storage->query_nb;
                float hop = metric_hops * 1.0f / storage->query_nb;
                std::cout << "ef=" << ef << " recall=" << recall << " qps=" << qps
                          << " filter_valid=" << valid_cnt << "/" << total_cnt << std::endl;
                outfile << ef << "," << recall << "," << qps << "," << dco << "," << hop << std::endl;
                if (recall >= 0 && recall < 0.9f)
                    break;
            }
            outfile.close();
        }
        void search_filter(std::vector<int> &SearchEF, std::string saveprefix, int edge_limit,
                           const std::vector<std::vector<int>> &gt = {})
        {
            const int N = storage->data_nb;

            std::string savepath = saveprefix + "_filter.csv";
            std::cout << "Savepath: " << savepath << std::endl;
            CheckPath(savepath);
            std::ofstream outfile(savepath);
            if (!outfile.is_open())
                throw Exception("cannot open " + savepath);

            if (flags.size() < (size_t)N)
                flags.assign(N, false);
            storage->BuildLabelBlocks(); // 标签块边界（幂等），检索时 O(1) 取 [l,r]

            for (auto ef : SearchEF)
            {
                int tp = 0;
                int valid_cnt = 0;
                int total_cnt = 0;
                float searchtime = 0;
                metric_hops = 0;
                metric_distance_computations = 0;

                for (int i = 0; i < storage->query_nb; i++)
                {
                    // 1. 标签集合：排序去重
                    std::vector<int> labels = storage->query_labels[i];
                    std::sort(labels.begin(), labels.end());
                    labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
                    if (labels.empty())
                        continue;

                    // 2. 查询标签位图
                    iPreGraph::Bitmap label_bitmap;
                    for (int label : labels)
                    {
                        if (label < 0)
                            continue;
                        label_bitmap.set(label);
                    }
                    if (label_bitmap.blocks.empty())
                        continue;

                    // 3. 数值范围：查询 MBR（节点剪枝 联合盒/每标签盒 ∩ 查询MBR 用）+ 角点 z 码
                    //    （箱角性质：盒内任意点的 z 码落在角点 z 区间 [q_z_min, q_z_max] 内，
                    //      用于第 6 步在最小/最大标签块内收窄 [l, r] 到满足数值约束的范围）
                    const auto &cons = storage->query_constraints[i];
                    int num_attrs = (int)cons.attr_constraints.size();
                    iPreGraph_multi::MBR wq(num_attrs);
                    std::vector<int> coords_min(num_attrs), coords_max(num_attrs);
                    for (int d = 0; d < num_attrs; d++)
                    {
                        wq.min_vals[d] = cons.attr_constraints[d].first;
                        wq.max_vals[d] = cons.attr_constraints[d].second;
                        coords_min[d] = cons.attr_constraints[d].first;
                        coords_max[d] = cons.attr_constraints[d].second;
                    }
                    iPreGraph_multi::MultiAttributeZCode q_z_min =
                        iPreGraph_multi::CalculateMultiAttributeZOrder(coords_min);
                    iPreGraph_multi::MultiAttributeZCode q_z_max =
                        iPreGraph_multi::CalculateMultiAttributeZOrder(coords_max);

                    // 4. 点级混合过滤（单一数组 flags：满足全部约束 = true）：
                    //    先按标签块置 true（标签命中），再在同一数组上做数值 AND。
                    std::fill(flags.begin(), flags.end(), false);
                    for (int label : labels)
                    {
                        auto pp = std::equal_range(storage->labels.begin(), storage->labels.end(), label);
                        if (pp.first == pp.second)
                            continue;
                        int cur_l = (int)(pp.first - storage->labels.begin());
                        int cur_r = (int)(pp.second - storage->labels.begin()) - 1;
                        for (int xb = cur_l; xb <= cur_r; xb++)
                        {
                            bool in_range = true;
                            for (int d = 0; d < num_attrs; d++)
                            {
                                int v = storage->z_attrs[xb][d];
                                if (v < cons.attr_constraints[d].first || v > cons.attr_constraints[d].second)
                                {
                                    in_range = false;
                                    break;
                                }
                            }
                            flags[xb] = in_range;
                        }
                    }

                    std::vector<int> Q_L, Q_R;
                    int bl = storage->label_block_start[labels[0]];
                    int br = storage->label_block_end[labels[0]];
                    int l = (int)(std::lower_bound(storage->z_codes.begin() + bl,
                                                   storage->z_codes.begin() + br + 1, q_z_min,
                                                   iPreGraph_multi::MultiAttributeZOrderLess) -
                                  storage->z_codes.begin());
                    bl = storage->label_block_start[labels.back()];
                    br = storage->label_block_end[labels.back()];

                    int r = (int)(std::upper_bound(storage->z_codes.begin() + bl,
                                                   storage->z_codes.begin() + br + 1, q_z_max,
                                                   iPreGraph_multi::MultiAttributeZOrderLess) -
                                  storage->z_codes.begin()) -
                            1;

                    std::vector<TreeNodeMixed *> node = tree->range_filter(tree->root, l, r);

                    timeval t1, t2;
                    gettimeofday(&t1, NULL);
                    std::priority_queue<PFI> res = TopDown_nodeentries_search_filter(
                        node, storage->query_points[i].data(), l, r, ef,
                        storage->query_K, edge_limit);
                    gettimeofday(&t2, NULL);
                    searchtime += GetTime(t1, t2);

                    std::map<int, int> record;
                    while (!res.empty())
                    {
                        auto x = res.top().second;
                        res.pop();
                        if (record.count(x))
                            continue;
                        record[x] = 1;
                        total_cnt++;
                        if (PointPass(x, flags))
                            valid_cnt++;
                        if (!gt.empty())
                        {
                            if (std::find(gt[i].begin(), gt[i].end(), x) != gt[i].end())
                                tp++;
                        }
                    }
                }

                float recall = gt.empty() ? -1.0f
                                          : 1.0f * tp / storage->query_nb / storage->query_K;
                float qps = storage->query_nb / searchtime;
                float dco = metric_distance_computations * 1.0f / storage->query_nb;
                float hop = metric_hops * 1.0f / storage->query_nb;
                std::cout << "ef=" << ef << " recall=" << recall << " qps=" << qps
                          << " filter_valid=" << valid_cnt << "/" << total_cnt << std::endl;
                outfile << ef << "," << recall << "," << qps << "," << dco << "," << hop << std::endl;
                if (recall >= 0 && recall < 0.9f)
                    break;
            }
            outfile.close();
        }

        // ------------------------------------------------------------------------
        // 检索入口（逐范围版）：标签集合 OR + 数值范围 AND
        // 前处理与 search_mixed 完全一致（标签位图、查询 MBR、点级 flags、
        // 候选区间列表 Q_L/Q_R）；区别在检索阶段：不再用多区间联合搜索
        // TopDown_nodeentries_search_mixed_or，而是对每个候选区间 [Q_L[k], Q_R[k]]
        // 单独调用单区间搜索 TopDown_nodeentries_search_mixed（区间内只含同一
        // 标签块的点），各区间返回的（近似）前 K 候选按 pid 去重、保留最小距离，
        // 最后全局按距离升序取前 K 作为合并结果。
        // 注意：逐范围搜索各自维护独立的 visited_set 与 ef 预算，总开销随范围数
        // 线性增长，用于与多区间联合版对比"每区间独立查询再合并"的开销。
        // 输出 CSV: ef,recall,qps,dco,hop
        // ------------------------------------------------------------------------
        void search_mixed_per_range(std::vector<int> &SearchEF, std::string saveprefix,
                                    int edge_limit,
                                    const std::vector<std::vector<int>> &gt = {})
        {
            const int N = storage->data_nb;

            std::string savepath = saveprefix + "_mixed_per_range.csv";
            std::cout << "Savepath: " << savepath << std::endl;
            CheckPath(savepath);
            std::ofstream outfile(savepath);
            if (!outfile.is_open())
                throw Exception("cannot open " + savepath);

            if (flags.size() < (size_t)N)
                flags.assign(N, false);
            storage->BuildLabelBlocks(); // 标签块边界（幂等），检索时 O(1) 取 [l,r]

            for (auto ef : SearchEF)
            {
                int tp = 0;
                int valid_cnt = 0;
                int total_cnt = 0;
                float searchtime = 0;
                metric_hops = 0;
                metric_distance_computations = 0;

                for (int i = 0; i < storage->query_nb; i++)
                {
                    // 1. 标签集合：排序去重
                    std::vector<int> labels = storage->query_labels[i];
                    std::sort(labels.begin(), labels.end());
                    labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
                    if (labels.empty())
                        continue;

                    // 2. 查询标签位图
                    iPreGraph::Bitmap label_bitmap;
                    for (int label : labels)
                    {
                        if (label < 0)
                            continue;
                        label_bitmap.set(label);
                    }
                    if (label_bitmap.blocks.empty())
                        continue;

                    // 3. 数值范围：查询 MBR + 角点 z 码（同 search_mixed）
                    const auto &cons = storage->query_constraints[i];
                    int num_attrs = (int)cons.attr_constraints.size();
                    iPreGraph_multi::MBR wq(num_attrs);
                    std::vector<int> coords_min(num_attrs), coords_max(num_attrs);
                    for (int d = 0; d < num_attrs; d++)
                    {
                        wq.min_vals[d] = cons.attr_constraints[d].first;
                        wq.max_vals[d] = cons.attr_constraints[d].second;
                        coords_min[d] = cons.attr_constraints[d].first;
                        coords_max[d] = cons.attr_constraints[d].second;
                    }
                    iPreGraph_multi::MultiAttributeZCode q_z_min =
                        iPreGraph_multi::CalculateMultiAttributeZOrder(coords_min);
                    iPreGraph_multi::MultiAttributeZCode q_z_max =
                        iPreGraph_multi::CalculateMultiAttributeZOrder(coords_max);

                    // 4. 点级混合过滤（同一数组 flags：满足全部约束 = true）
                    std::fill(flags.begin(), flags.end(), false);
                    for (int label : labels)
                    {
                        auto pp = std::equal_range(storage->labels.begin(), storage->labels.end(), label);
                        if (pp.first == pp.second)
                            continue;
                        int cur_l = (int)(pp.first - storage->labels.begin());
                        int cur_r = (int)(pp.second - storage->labels.begin()) - 1;
                        for (int xb = cur_l; xb <= cur_r; xb++)
                        {
                            bool in_range = true;
                            for (int d = 0; d < num_attrs; d++)
                            {
                                int v = storage->z_attrs[xb][d];
                                if (v < cons.attr_constraints[d].first || v > cons.attr_constraints[d].second)
                                {
                                    in_range = false;
                                    break;
                                }
                            }
                            flags[xb] = in_range;
                        }
                    }

                    timeval t1, t2;
                    gettimeofday(&t1, NULL);
                    // 5. 候选区间列表 [Q_L, Q_R]（每个查询标签一个区间，同 search_mixed）
                    std::vector<int> Q_L, Q_R;
                    for (int lab : labels)
                    {
                        if (lab < 0 || lab >= (int)storage->label_block_start.size())
                            continue;
                        int bl = storage->label_block_start[lab];
                        int br = storage->label_block_end[lab];
                        if (bl < 0 || br < bl)
                            continue;
                        int ll = (int)(std::lower_bound(storage->z_codes.begin() + bl,
                                                        storage->z_codes.begin() + br + 1, q_z_min,
                                                        iPreGraph_multi::MultiAttributeZOrderLess) -
                                       storage->z_codes.begin());
                        int rr = (int)(std::upper_bound(storage->z_codes.begin() + bl,
                                                        storage->z_codes.begin() + br + 1, q_z_max,
                                                        iPreGraph_multi::MultiAttributeZOrderLess) -
                                       storage->z_codes.begin()) -
                                 1;
                        if (ll <= rr)
                        {
                            Q_L.push_back(ll);
                            Q_R.push_back(rr);
                        }
                    }
                    // === 优化后：K路归并 + 最大堆动态Top-K ===
                    // 最大堆：堆顶是当前Top-K中距离最大的元素，用于快速淘汰
                    std::priority_queue<PFI> topk_heap;
                    // 记录已在堆中的pid及其当前最优距离，避免重复入堆
                    std::unordered_map<int, float> in_heap;

                    for (size_t k = 0; k < Q_L.size(); k++)
                    {
                        std::vector<TreeNodeMixed *> node = tree->range_filter(tree->root, Q_L[k], Q_R[k]);
                        if (node.empty())
                            continue;

                        std::priority_queue<PFI> res_range = TopDown_nodeentries_search_mixed(
                            node, storage->query_points[i].data(), Q_L[k], Q_R[k], ef,
                            storage->query_K, edge_limit, wq, flags);

                        while (!res_range.empty())
                        {
                            auto [dist, pid] = res_range.top();
                            res_range.pop();

                            // 剪枝1：如果堆已满且当前距离 >= 堆顶（Top-K中最大距离），直接跳过
                            if ((int)topk_heap.size() == storage->query_K && dist >= topk_heap.top().first)
                                continue;

                            // 去重：同一pid只保留最小距离
                            auto it = in_heap.find(pid);
                            if (it != in_heap.end())
                            {
                                if (dist >= it->second)
                                    continue;      // 不是更优解，跳过
                                it->second = dist; // 更新为更小距离
                            }
                            else
                            {
                                in_heap[pid] = dist;
                            }

                            topk_heap.emplace(dist, pid);

                            // 堆超过K时弹出最大值，保持堆大小恒为K
                            if ((int)topk_heap.size() > storage->query_K)
                            {
                                auto removed = topk_heap.top();
                                topk_heap.pop();
                                in_heap.erase(removed.second);
                            }
                        }
                    }
                    gettimeofday(&t2, NULL);
                    searchtime += GetTime(t1, t2);
                    std::map<int, int> record;
                    while (!topk_heap.empty())
                    {
                        auto x = topk_heap.top().second;
                        topk_heap.pop();
                        if (record.count(x))
                            continue;
                        record[x] = 1;
                        total_cnt++;
                        if (PointPass(x, flags))
                            valid_cnt++;
                        if (!gt.empty())
                        {
                            if (std::find(gt[i].begin(), gt[i].end(), x) != gt[i].end())
                                tp++;
                        }
                    }
                }

                float recall = gt.empty() ? -1.0f
                                          : 1.0f * tp / storage->query_nb / storage->query_K;
                float qps = storage->query_nb / searchtime;
                float dco = metric_distance_computations * 1.0f / storage->query_nb;
                float hop = metric_hops * 1.0f / storage->query_nb;
                std::cout << "ef=" << ef << " recall=" << recall << " qps=" << qps
                          << " filter_valid=" << valid_cnt << "/" << total_cnt << std::endl;
                outfile << ef << "," << recall << "," << qps << "," << dco << "," << hop << std::endl;
                if (recall >= 0 && recall < 0.9f)
                    break;
            }
            outfile.close();
        }
    };
}
