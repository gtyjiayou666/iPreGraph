#pragma once

#include <vector>
#include "utils_multi.h"
#include "searcher.hpp"
#include "memory.hpp"
#include <bitset>
#include <stack>
#include <cmath>
#include <climits>

// ============================================================================
// iRG_search_zorder_label.h —— (z-order, label) 字典序变体
//
// 数据组织：数据点先按数值属性的 z-order 码升序排列；相同 z 码的点再按标签升序
// 排列，即全局顺序为 (z_code, label) 字典序（与 iRG_search_mixed.h 的
// (label, z_code) 顺序相反）。因此：
//   - 数值范围谓词对应数组上的一个连续 z 区间 [z_min, z_max]（用 lower_bound /
//     upper_bound 定位，与 iRG_search_zorder.h 完全一致）；
//   - 同一标签的点散布在各 z 组内（不连续），标签约束只能在检索过程中用
//     点级过滤 + 节点级标签位图剪枝解决。
// 树节点（TreeNodeZLabel）在 MBR 之外新增 label_bmp（节点范围内出现过的标签
// 位图，子节点并集，精确）；剪枝时要求：
//   label_bmp ∩ 查询标签集 ≠ ∅  且  节点 MBR ∩ 查询数值范围 ≠ ∅
// 查询语义：标签集合（标签间 OR）+ 数值属性范围（属性间 AND）。
// 数据装载：属性 JSON 每行 [label, a0, a1, ...]（attr_nb 列，label_col 指明标签列），
//   数据必须已按 (z_code, label) 排好序；装载后 ValidateZLabelOrder 校验不变量。
// ============================================================================

namespace iPreGraph_multi
{
    // utils.h 中 tableint / linklistsizeint 定义于 iPreGraph 命名空间，这里引入以便裸用
    using iPreGraph::linklistsizeint;
    using iPreGraph::tableint;

    // ------------------------------------------------------------------------
    // (z-order, label) 变体的树节点：MBR + 标签位图
    // ------------------------------------------------------------------------
    class TreeNodeZLabel
    {
    public:
        MBR mbr;
        int node_id;
        int lbound, rbound;
        int depth;
        std::vector<TreeNodeZLabel *> childs;

        TreeNodeZLabel(int l, int r, int d, int dim)
            : lbound(l), rbound(r), depth(d), mbr(dim)
        {
        }
    };

    // ------------------------------------------------------------------------
    // 线段树（结构与 SegmentTreeZorder 完全一致：ways_=2），构建时额外维护
    // 每节点的标签位图（叶子 = 该点标签，内部 = 子节点并集）。
    // ------------------------------------------------------------------------
    class SegmentTreeZorderLabel
    {
    public:
        int ways_ = 2;
        int dim_ = 2; // 数值属性维度
        TreeNodeZLabel *root{nullptr};
        int max_depth{-1};
        int node_count{0}; // 节点数（兼 node_id 分配器）
        std::vector<TreeNodeZLabel *> treenodes;

        SegmentTreeZorderLabel(int data_nb, int dim) : dim_(dim)
        {
            root = new TreeNodeZLabel(0, data_nb - 1, 0, dim);
            root->node_id = node_count++;
        }

        // 输入数据必须已按 (z_code, label) 排序：
        //   attrs : N × dim 数值属性（z-order 重排后的顺序）；
        //   labels: N 个标签（与 attrs 同序）。
        void BuildTree(TreeNodeZLabel *u,
                       const std::vector<std::vector<int>> &attrs,
                       const std::vector<int> &labels)
        {
            if (u == nullptr)
                throw Exception("Tree node is a nullptr");
            if (dim_ <= 0)
                throw Exception("SegmentTreeZorderLabel: dim_ not set");

            treenodes.emplace_back(u);
            max_depth = std::max(max_depth, u->depth);
            int L = u->lbound, R = u->rbound;

            if (L == R)
            {
                // 叶子：MBR 退化为该点，标签位图 = 该点标签
                for (int d = 0; d < dim_; d++)
                {
                    double v = static_cast<double>(attrs[L][d]);
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
                TreeNodeZLabel *childnode = new TreeNodeZLabel(l, r, u->depth + 1, dim_);
                childnode->node_id = node_count++;
                u->childs.emplace_back(childnode);
                BuildTree(childnode, attrs, labels);
                l = r + 1;
            }
            if (L != R)
            {
                for (const auto *child : u->childs)
                    u->mbr.Merge(child->mbr);
            }
        }

        // 按区间 [ql, qr] 筛出覆盖该区间的最少树节点集合
        std::vector<TreeNodeZLabel *> range_filter(TreeNodeZLabel *u, int ql, int qr)
        {
            std::vector<TreeNodeZLabel *> res;
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

        ~SegmentTreeZorderLabel()
        {
            for (auto *p : treenodes)
                delete p;
            treenodes.clear();
            root = nullptr;
        }
    };

    // ------------------------------------------------------------------------
    // (z-order, label) 变体的数据装载：向量 + 标签 + 数值属性（z-code）+
    // 查询（标签集合 OR + 数值范围 AND）。
    // 约定：data_points / labels / z_attrs / z_codes 均已按 (z_code, label) 排序。
    // 继承 iPreGraph_multi::DataLoader 以获得向量装载与 ground truth 装载。
    // ------------------------------------------------------------------------
    // Parse mixed predicates and enforce (Morton code, label) row ordering.
    class ZLabelDataLoader : public iPreGraph_multi::DataLoader
    {
    public:
        std::vector<int> labels;                                     // (z,label) 排序后的标签
        std::vector<std::vector<int>> z_attrs;                       // 数值属性 N×D（不含标签列）
        std::vector<iPreGraph_multi::MultiAttributeZCode> z_codes; // 排序后的 z-code（非递减）

        // 每条查询：标签集合（OR）+ 数值范围约束（AND）
        std::vector<std::vector<int>> query_labels;
        std::vector<Attr_Constraint> query_constraints;

        ZLabelDataLoader() {}
        ~ZLabelDataLoader() {}

        // ------------------------------------------------------------------
        // 简单 JSON 工具：仅解析本仓库数据文件用到的格式（数组 + 整数）
        // ------------------------------------------------------------------
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
        // 已按 (z_code, label) 排序）。label_col 为标签所在列，其余列为数值属性。
        // 加载后自动计算 z-code 并校验 (z, label) 排序不变量。
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
            if (data_nb > 0 && data_nb != n_points)
                throw Exception("LoadAttrJson: data_nb (" + std::to_string(data_nb) +
                                ") != attribute points (" + std::to_string(n_points) + ")");
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
            ValidateZLabelOrder();
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
            int q_nb = (int)root.childs.size();
            if (query_nb > 0 && query_nb != q_nb)
                throw Exception("LoadQueryRangeJson: query_nb (" + std::to_string(query_nb) +
                                ") != queries in file (" + std::to_string(q_nb) + ")");
            query_nb = q_nb;
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
        // 排序不变量校验：(z_code, label) 字典序非递减。
        // ------------------------------------------------------------------
        void ValidateZLabelOrder() const
        {
            if (z_codes.size() != labels.size() || z_codes.size() != z_attrs.size())
                throw Exception("ValidateZLabelOrder: size mismatch among labels/z_attrs/z_codes");
            for (size_t i = 1; i < z_codes.size(); i++)
            {
                if (iPreGraph_multi::MultiAttributeZOrderLess(z_codes[i], z_codes[i - 1]))
                    throw Exception("ValidateZLabelOrder: z_codes not non-decreasing "
                                    "(data must be sorted by (z_code, label)) at " +
                                    std::to_string(i));
                bool same_z = !iPreGraph_multi::MultiAttributeZOrderLess(z_codes[i - 1], z_codes[i]) &&
                              !iPreGraph_multi::MultiAttributeZOrderLess(z_codes[i], z_codes[i - 1]);
                if (same_z && labels[i] < labels[i - 1])
                    throw Exception("ValidateZLabelOrder: labels not non-decreasing "
                                    "within equal z_code at " +
                                    std::to_string(i));
            }
        }

        // ------------------------------------------------------------------
        // 查询标签统计：查询数、每条查询标签数 min/max/均值、标签频次直方图
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
                printf(" %zu:%d", k, hist[k]);
            printf("\n  label frequency: nonzero labels=%d, max_freq=%d\n",
                   (int)label_freq.size(), max_label >= 0 ? label_freq[max_label] : 0);
        }
    };

    // ------------------------------------------------------------------------
    // 检索类：结构与 iPreGraph_Search_Zorder 完全一致（边装载、层级数据布局、
    // 检索循环），仅树节点带标签位图，剪枝增加标签维度。
    // ------------------------------------------------------------------------
    template <typename dist_t>
    class iPreGraph_Search_Zorder_Label
    {
    public:
        ZLabelDataLoader *storage;
        SegmentTreeZorderLabel *tree;
        size_t max_elements_{0};
        size_t dim_{0};
        size_t attr_dim_{0}; // 数值属性维度（用于 Z-order / MBR / 属性过滤）
        size_t M_out{0};
        size_t ef_construction{0};

        size_t size_data_per_element_{0};
        size_t size_links_per_element_{0};
        size_t data_size_{0};

        size_t size_links_per_layer_{0};
        size_t offsetData_{0};
        int range_num_avg{0};

        char *data_memory_{nullptr};

        hnswlib::L2Space *space;
        hnswlib::DISTFUNC<dist_t> fstdistfunc_;
        void *dist_func_param_{nullptr};

        size_t metric_distance_computations{0};
        size_t metric_hops{0};

        int prefetch_lines{0};

        iPreGraph_Search_Zorder_Label(std::string vectorfilename, std::string edgefilename, ZLabelDataLoader *store, int M)
            : storage(store)
        {
            std::ifstream edgefile(edgefilename, std::ios::in | std::ios::binary);
            if (!edgefile.is_open())
                throw Exception("cannot open " + edgefilename);
            std::ifstream vectorfile(vectorfilename, std::ios::in | std::ios::binary);
            if (!vectorfile.is_open())
                throw Exception("cannot open " + vectorfilename);
            max_elements_ = storage->data_nb;
            dim_ = storage->Dim;
            vectorfile.read((char *)&max_elements_, sizeof(int));
            vectorfile.read((char *)&dim_, sizeof(int));

            attr_dim_ = storage->z_attrs.empty() ? 0 : (int)storage->z_attrs[0].size();
            std::cout << "attr_dim:" << attr_dim_ << std::endl;
            tree = new SegmentTreeZorderLabel(max_elements_, attr_dim_);
            tree->BuildTree(tree->root, storage->z_attrs, storage->labels);

            space = new hnswlib::L2Space(dim_);
            fstdistfunc_ = space->get_dist_func();
            dist_func_param_ = space->get_dist_func_param();
            M_out = M;

            data_size_ = dim_ * sizeof(float);
            size_links_per_layer_ = M_out * sizeof(tableint) + sizeof(linklistsizeint);
            size_links_per_element_ = size_links_per_layer_ * (tree->max_depth + 1);
            size_data_per_element_ = size_links_per_element_ + data_size_;
            offsetData_ = size_links_per_element_;

            data_memory_ = (char *)malloc(max_elements_ * size_data_per_element_);
            if (data_memory_ == nullptr)
                throw std::runtime_error("Not enough memory");

            for (int pid = 0; pid < max_elements_; pid++)
            {
                for (int layer = 0; layer <= tree->max_depth; layer++)
                {
                    linklistsizeint *data = get_linklist(pid, layer);
                    edgefile.read((char *)data, sizeof(tableint));
                    int size = getListCount(data);
                    if (size > M_out)
                        throw Exception("real linklist size is bigger than defined M_out");
                    for (int i = 0; i < size; i++)
                    {
                        char *current_neighbor_ = (char *)(data + 1 + i);
                        edgefile.read(current_neighbor_, sizeof(tableint));
                    }
                }

                size_t size_in_bytes = dim_ * sizeof(float);
                char *data = getDataByInternalId(pid);
                vectorfile.read(data, dim_ * sizeof(float));
                // std::memcpy(data, reinterpret_cast<char *>(storage->data_points[pid].data()), size_in_bytes);
            }

            edgefile.close();
            vectorfile.close();
        }

        ~iPreGraph_Search_Zorder_Label()
        {
            free(data_memory_);
            data_memory_ = nullptr;
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

        // ------------------------------------------------------------------------
        // 节点剪枝（带缓存）：保留 iff 标签位图与查询标签集有交集 且 MBR 与查询
        // 数值范围相交。缓存按 node_id 存 1（保留）/2（剪枝）；判定只依赖查询
        // 标签集与查询 MBR，与候选区间无关，同一查询内跨区间复用安全。
        // ------------------------------------------------------------------------
        inline bool NodePruneZLabel(const TreeNodeZLabel *u,
                                    const MBR &wp,
                                    std::vector<uint8_t> &node_cache) const
        {
            int nid = u->node_id;
            uint8_t c = node_cache[nid];
            if (c != 0)
                return c == 1; // 1=可能含真值点, 2=可剪枝

            bool res = u->mbr.Intersects(wp);
            node_cache[nid] = res ? 1 : 2;
            return res;
        }

        // ------------------------------------------------------------------------
        // 边选择（镜像 iRG_search_zorder.h 的 SelectEdgeHybrid，剪枝改为
        // NodePruneZLabel = 标签位图 + 数值 MBR 联合剪枝）：
        // 沿树下降到 [ql,qr] 内，兄弟子树被剪掉则继续沿 pid 子树下潜；
        // 收集邻居时用 filter_ids_map（标签命中 AND 数值命中）过滤。
        // ------------------------------------------------------------------------
        std::vector<tableint> SelectEdgeZLabel(int pid, int ql, int qr, int edge_limit,
                                               searcher::Bitset<uint64_t> &visited_set,
                                               MBR &wp,
                                               std::vector<uint8_t> &node_cache,
                                               const std::vector<char> &filter_ids_map)
        {
            TreeNodeZLabel *cur_node = nullptr, *nxt_node = tree->root;
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
                if (!NodePruneZLabel(cur_node, wp, node_cache))
                {
                    return selected_edges;
                }
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
                                if (!NodePruneZLabel(cur_node->childs[i], wp, node_cache))
                                {
                                    contain = true;
                                }
                            }
                        }
                        if (contain)
                        {
                            cur_node = nxt_node;
                        }
                    }
                } while (contain);
                int *data = (int *)get_linklist(pid, cur_node->depth);
                size_t size = getListCount((linklistsizeint *)data);
                for (size_t j = 1; j <= size; ++j)
                {
                    int neighborId = *(data + j);
                    if (visited_set.get(neighborId))
                        continue;
                    if (filter_ids_map[neighborId] == 0)
                        continue;
                    selected_edges.emplace_back(neighborId);
                    if (selected_edges.size() == (size_t)edge_limit)
                        return selected_edges;
                }
            } while (nxt_node);
            return selected_edges;
        }

        // ------------------------------------------------------------------------
        // DFS（镜像 iRG_search_zorder.h 的 DFS，剪枝改为 NodePruneZLabel）：
        // 沿邻接表向下再探索两层，只收集"标签命中 + 数值命中"的点。
        // ------------------------------------------------------------------------
        void DFSZLabel(int pid, int ql, int qr, int edge_limit, int deep,
                       std::vector<tableint> &edges,
                       MBR &wp,
                       std::vector<uint8_t> &node_cache,
                       const std::vector<char> &filter_ids_map)
        {
            TreeNodeZLabel *cur_node = tree->root;
            TreeNodeZLabel *nxt_node = nullptr;
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
                if (!NodePruneZLabel(cur_node, wp, node_cache))
                {
                    break;
                }
                bool inters = false;
                if (cur_node->childs.size() == 0)
                {
                    nxt_node = nullptr;
                }
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
                            if (NodePruneZLabel(cur_node->childs[i], wp, node_cache))
                            {
                                inters = true;
                            }
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
                        if (filter_ids_map[neighborId] != 0)
                        {
                            edges.emplace_back(neighborId);
                        }
                        else if (deep > 1)
                        {
                            DFSZLabel(neighborId, ql, qr, edge_limit, deep - 1, edges, wp, node_cache, filter_ids_map);
                        }
                        if (edges.size() >= (size_t)edge_limit)
                        {
                            return;
                        }
                    }
                }
                cur_node = nxt_node;
            }
        }

        // ------------------------------------------------------------------------
        // 单区间 top-down 检索（镜像 iRG_search_zorder.h 的
        // TopDown_nodeentries_search_zorder_and，进入点/邻居走 SelectEdgeZLabel +
        // DFSZLabel 的标签+数值联合剪枝）。
        // ------------------------------------------------------------------------
        std::priority_queue<PFI> TopDown_nodeentries_search_zorder_label(
            std::vector<TreeNodeZLabel *> &filterednodes,
            const void *query_data, int l, int r, int ef, int query_k, int edge_limit,
            MBR &wp,
            std::vector<uint8_t> &node_cache,
            const std::vector<char> &filter_ids_map)
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
                    if (filter_ids_map[pid] != 0)
                    {
                        top_candidates.emplace(dis, pid);
                        lowerBound = top_candidates.top().first;
                    }
                }
                else if (dis < lowerBound)
                {
                    candidate_set.emplace(dis, pid);
                    if (filter_ids_map[pid] != 0)
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
                auto selected_edges = SelectEdgeZLabel(current_pid, l, r, edge_limit, visited_set,
                                                       wp, node_cache, filter_ids_map);
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
                std::vector<tableint> edges;
                edges.reserve(edge_limit / 2);
                DFSZLabel(current_pid, l, r, edge_limit / 2, 2, edges, wp, node_cache, filter_ids_map);
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
        // 检索入口：逐查询执行"标签集合 OR + 数值范围 AND"的近似最近邻搜索。
        // gt: [query][k] 真值（pid，(z,label) 排序后的下标）；为空则 recall 输出 -1
        // 输出 CSV: ef,recall,qps,dco,hop
        // ------------------------------------------------------------------------
        void search_zorder_label(std::vector<int> &SearchEF, std::string saveprefix, int edge_limit)
        {
            const int N = storage->data_nb;

            std::string savepath = saveprefix + "_zorder_label_memory.csv";
            std::cout << "Savepath: " << savepath << std::endl;
            CheckPath(savepath);
            std::ofstream outfile(savepath);
            if (!outfile.is_open())
                throw Exception("cannot open " + savepath);

            std::vector<char> filter_ids_map(N, 0);

            std::vector<std::vector<int>> &gt = storage->ground_truth["mixed"];
            for (auto ef : SearchEF)
            {
                int tp = 0;
                float searchtime = 0;
                metric_hops = 0;
                metric_distance_computations = 0;

                for (int i = 0; i < storage->query_nb; i++)
                {
                    // 1. 查询标签集合：排序去重
                    std::vector<int> labels = storage->query_labels[i];
                    std::sort(labels.begin(), labels.end());
                    labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
                    if (labels.empty())
                        continue;

                    // 2. 查询标签位图
                    iPreGraph::Bitmap q_labels;
                    for (int lb : labels)
                    {
                        if (lb < 0)
                            continue;
                        q_labels.set(lb);
                    }
                    if (q_labels.blocks.empty())
                        continue;

                    // 3. 数值范围 -> 查询 MBR（节点剪枝用）
                    const auto &cons = storage->query_constraints[i];
                    int num_attrs = (int)cons.attr_constraints.size();
                    MBR wq;
                    wq.dim = num_attrs;
                    wq.min_vals.resize(num_attrs);
                    wq.max_vals.resize(num_attrs);
                    for (int d = 0; d < num_attrs; d++)
                    {
                        wq.min_vals[d] = cons.attr_constraints[d].first;
                        wq.max_vals[d] = cons.attr_constraints[d].second;
                    }

                    // 4. 点级过滤：标签命中 AND 数值命中
                    for (int xb = 0; xb < N; xb++)
                    {
                        if (!q_labels.test(storage->labels[xb]))
                        {
                            filter_ids_map[xb] = 0;
                            continue;
                        }
                        bool in_range = true;
                        for (int d = 0; d < num_attrs; d++)
                        {
                            int v = storage->z_attrs[xb][d];
                            if (v < static_cast<int>(wq.min_vals[d]) ||
                                v > static_cast<int>(wq.max_vals[d]))
                            {
                                in_range = false;
                                break;
                            }
                        }
                        filter_ids_map[xb] = static_cast<char>(in_range);
                    }

                    timeval t1, t2;
                    gettimeofday(&t1, NULL);
                    // 5. 数值范围 -> z 区间 [l, r]（数据按 z 非递减，二分定位）；
                    //    无数值属性时退化为全数组。
                    int l = 0, r = N - 1;
                    std::vector<int> coords_min(num_attrs), coords_max(num_attrs);
                    for (int d = 0; d < num_attrs; d++)
                    {
                        coords_min[d] = cons.attr_constraints[d].first;
                        coords_max[d] = cons.attr_constraints[d].second;
                    }
                    MultiAttributeZCode z_min = CalculateMultiAttributeZOrder(coords_min);
                    MultiAttributeZCode z_max = CalculateMultiAttributeZOrder(coords_max);

                    auto it1 = std::lower_bound(
                        storage->z_codes.begin(), storage->z_codes.end(),
                        z_min, MultiAttributeZOrderLess);
                    l = (int)std::distance(storage->z_codes.begin(), it1);
                    auto it2 = std::upper_bound(
                        storage->z_codes.begin(), storage->z_codes.end(),
                        z_max, MultiAttributeZOrderLess);
                    if (it2 != storage->z_codes.begin())
                        --it2;
                    r = (int)std::distance(storage->z_codes.begin(), it2);
                    if (l > r)
                        continue;

                    std::vector<TreeNodeZLabel *> node = tree->range_filter(tree->root, l, r);
                    std::vector<uint8_t> node_cache((size_t)tree->node_count, 0);
                    std::priority_queue<PFI> res = TopDown_nodeentries_search_zorder_label(
                        node, storage->query_points[i].data(), l, r, ef,
                        storage->query_K, edge_limit, wq, node_cache, filter_ids_map);
                    gettimeofday(&t2, NULL);
                    searchtime += GetTime(t1, t2);

                    std::map<int, int> record;
                    while (!res.empty())
                    {
                        auto x = res.top().second;
                        res.pop();
                        if (record.count(x))
                            throw Exception("repetitive search results");
                        record[x] = 1;
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
                          << " dco=" << dco << " hop=" << hop << std::endl;
                outfile << ef << "," << recall << "," << qps << "," << dco << "," << hop << std::endl;
                if (recall >= 0 && recall < 0.9f)
                    break;
            }
            outfile.close();
        }
    };
}
