#pragma once

#include <vector>
#include "segment_tree_utils.h"
#include "search_containers.h"
#include "memory_utils.h"
#include <bitset>
#include <stack>
#include <cmath>
#include <climits>

namespace iPreGraph
{

    // 解析 [[...],[...],[...]] 形式，保留外层分组，内层是整数列表
    inline std::vector<std::vector<int>> fan_parse_json_array_of_array(const std::string &filename)
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

    // 解析真值 JSON，自动识别两种格式：
    //   嵌套 [[id,id,...],[id,...],...] ：外层每个数组 = 一个查询的真值列表
    //   扁平 [a,b,c, d,e,f, ...]       ：query_nb * query_K 个整数按查询顺序拼接
    inline std::vector<std::vector<int>> fan_parse_gt_json(const std::string &filename, int query_nb, int query_K)
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

    // Locate label blocks in sorted base rows and search their graph candidates.
    // The label bitmap prunes nodes that cannot satisfy any requested label.
    template <typename dist_t>
    class iPreGraph_Search_Label
    {
    public:
        DataLoaderOr *storage;
        SegmentTreeLabel *tree = nullptr;
        size_t max_elements_{0};
        size_t dim_{0};
        size_t M_out{0};
        size_t ef_construction{0};

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

        size_t metric_distance_computations1{0};
        size_t metric_hops1{0};
        int prefetch_lines{0};

        iPreGraph_Search_Label(std::string vectorfilename, std::string edgefilename, DataLoaderOr *store, int M) : storage(store)
        {
            std::ifstream vectorfile(vectorfilename, std::ios::in | std::ios::binary);
            if (!vectorfile.is_open())
                throw Exception("cannot open " + vectorfilename);
            std::ifstream edgefile(edgefilename, std::ios::in | std::ios::binary);
            if (!edgefile.is_open())
                throw Exception("cannot open " + edgefilename);

            vectorfile.read((char *)&max_elements_, sizeof(int));
            vectorfile.read((char *)&dim_, sizeof(int));

            tree = new SegmentTreeLabel(max_elements_, storage->label_arr, 1);
            // tree->BuildTree(tree->root);

            space = new hnswlib::L2Space(dim_);
            fstdistfunc_ = space->get_dist_func();
            dist_func_param_ = space->get_dist_func_param();
            M_out = M;

            data_size_ = (dim_ + 7) / 8 * 8 * sizeof(float);
            size_links_per_layer_ = M_out * sizeof(tableint) + sizeof(linklistsizeint);
            size_links_per_element_ = (size_links_per_layer_ * (tree->max_depth + 1) + 31) / 32 * 32;
            size_data_per_element_ = size_links_per_element_ + data_size_;
            offsetData_ = size_links_per_element_;
            prefetch_lines = data_size_ >> 4;

            data_memory_ = (char *)memory::align_mm<1 << 21>(max_elements_ * size_data_per_element_);
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
                    // std::sort(data + 1, data + 1 + size);
                }

                char *data = getDataByInternalId(pid);
                // vectorfile.read(data, data_size_);
                vectorfile.read(data, dim_ * sizeof(float));
            }

            edgefile.close();
            vectorfile.close();
            std::cout << "load index finished ..." << std::endl;
        }

        ~iPreGraph_Search_Label()
        {
            memory::aligned_free(data_memory_);
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

        std::vector<tableint> SelectEdge(int pid, int ql, int qr, int edge_limit, searcher::Bitset<uint64_t> &visited_set)
        {
            TreeNodeLabel *cur_node = nullptr, *nxt_node = tree->root;
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

        std::vector<bool> flags = std::vector<bool>(10000000, false);

        struct GapInfo
        {
            int index; // 间隙在原始区间之间的索引 (位于 Q_R[i] 和 Q_L[i+1] 之间)
            int len;
        };
        struct CompareGap
        {
            bool operator()(const GapInfo &a, const GapInfo &b)
            {
                return a.len < b.len; // 最大堆：大的元素优先级高
            }
        };
        void merge_small_gaps(const std::vector<int> &Q_L, const std::vector<int> &Q_R,
                              std::vector<int> &Q_L1, std::vector<int> &Q_R1,
                              std::vector<int> &mapping, double p)
        {
            int n = Q_L.size();
            mapping.assign(n, 0);

            // std::vector<GapInfo> gaps;
            double total_data = Q_R[n - 1] - Q_L[0] + 1;
            double total_gaps = 0;

            std::priority_queue<GapInfo, std::vector<GapInfo>, CompareGap> max_heap_gaps;
            // max_heap_gaps.reserve(n > 0 ? n - 1 : 0);

            for (int i = 0; i < n - 1; ++i)
            {
                int g_l = Q_R[i] + 1;
                int g_r = Q_L[i + 1] - 1;
                max_heap_gaps.push({i, g_r - g_l + 1});
                total_gaps = total_gaps + g_r - g_l + 1;
            }
            int total_nogaps = total_data - total_gaps;

            std::vector<bool> is_keeper(n - 1, false);

            int s = total_data - (p * total_nogaps);
            if (s <= 0)
            {
                Q_L1.push_back(Q_L[0]);
                Q_R1.push_back(Q_R[n - 1]);
                // std::cout << current_gaps << "," << max_heap_gaps.size() << std::endl;
                return;
            }
            int current_gaps = 0;
            while (!max_heap_gaps.empty())
            {
                GapInfo gap = max_heap_gaps.top();
                current_gaps = current_gaps + gap.len;
                // std::cout << current_gaps << ",";
                if (current_gaps >= s)
                {
                    is_keeper[gap.index] = true;
                    // std::cout << current_gaps << "," << max_heap_gaps.size() << std::endl;
                    break;
                }
                max_heap_gaps.pop();
                is_keeper[gap.index] = true;
            }
            int current_l = Q_L[0];
            int current_r = Q_R[0];

            int new_idx = 0;
            mapping[0] = new_idx;
            for (int i = 0; i < n - 1; ++i)
            {
                if (is_keeper[i])
                {
                    Q_L1.push_back(current_l);
                    Q_R1.push_back(current_r);
                    new_idx++;
                    current_l = Q_L[i + 1];
                    current_r = Q_R[i + 1];
                    mapping[i + 1] = new_idx;
                }
                else
                {
                    current_r = Q_R[i + 1];
                    mapping[i + 1] = new_idx;
                }
            }
            // 推送最后一个区间
            Q_L1.push_back(current_l);
            Q_R1.push_back(current_r);
        }

        std::priority_queue<PFI> TopDown_nodeentries_search_or_Pro(std::vector<TreeNodeLabel *> &filterednodes, const void *query_data, int ef, int query_k, std::vector<int> &Q_L, std::vector<int> &Q_R, int edge_limit, int id)
        {
            std::vector<int> Q_L1, Q_R1, mapping;
            merge_small_gaps(Q_L, Q_R, Q_L1, Q_R1, mapping, 3);
            int range_num = Q_L1.size();
            // std::cout << "range_num = " << range_num << std::endl;

            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine e(seed);

            std::priority_queue<PFIORPRO, std::vector<PFIORPRO>, Compare> candidate_set;
            // std::priority_queue<PFI, std::vector<PFI>, std::greater<PFI>> candidate_set;
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
                    candidate_set.emplace(dis, pid, mapping[id]);
                    top_candidates.emplace(dis, pid);
                    lowerBound = top_candidates.top().first;
                }
                else if (dis < lowerBound)
                {
                    candidate_set.emplace(dis, pid, mapping[id]);
                    top_candidates.emplace(dis, pid);
                    top_candidates.pop();
                    lowerBound = top_candidates.top().first;
                }
            }
            // bool erfen = range_num < 16;
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
                    // if (erfen)
                    // {
                    //     for (int i = 0; i < range_num; i++)
                    //     {
                    //         if (current_pid >= Q_L1[i])
                    //         {
                    //             if (current_pid <= Q_R1[i])
                    //             {
                    //                 range_id = i;
                    //                 break;
                    //             }
                    //         }
                    //         else
                    //         {
                    //             break;
                    //         }
                    //     }
                    // }
                    // else
                    // {
                    auto it = std::lower_bound(Q_L1.begin(), Q_L1.end(), current_pid);
                    if (it != Q_L1.end())
                    {
                        int i = static_cast<int>(std::distance(Q_L1.begin(), it));
                        if (*it == current_pid)
                        {
                            range_id = i;
                        }
                        else
                        {
                            range_id = i - 1;
                        }
                    }
                    else
                    {
                        range_id = range_num - 1;
                    }
                    // }
                }
                auto selected_edges = SelectEdge(current_pid, Q_L1[range_id], Q_R1[range_id], edge_limit, visited_set);
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
                        candidate_set.emplace(dis, neighbor_id, range_id);
                        if (flags[neighbor_id])
                        {
                            top_candidates.emplace(dis, neighbor_id);
                            lowerBound = top_candidates.top().first;
                        }
                    }
                    else if (dis < lowerBound)
                    {
                        candidate_set.emplace(dis, neighbor_id, range_id);
                        if (flags[neighbor_id])
                        {
                            top_candidates.emplace(dis, neighbor_id);
                            top_candidates.pop();
                            lowerBound = top_candidates.top().first;
                        }
                    }
                }

                if (range_num == 1)
                {
                    continue;
                }
                std::vector<tableint> edges;
                edges.reserve(edge_limit / 2);
                DFS_new1(Q_L, Q_R, current_pid, edge_limit / 2, 2, edges);
                int num_edges1 = edges.size();
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
                        // candidate_set.emplace(dis, neighbor_id, mapping[edges[i].second]);
                        candidate_set.emplace(dis, neighbor_id, -1);
                        // if (flags[neighbor_id])
                        // {
                        top_candidates.emplace(dis, neighbor_id);
                        lowerBound = top_candidates.top().first;
                        // }
                    }
                    else if (dis < lowerBound)
                    {
                        // candidate_set.emplace(dis, neighbor_id, mapping[edges[i].second]);
                        candidate_set.emplace(dis, neighbor_id, -1);
                        // if (flags[neighbor_id])
                        // {
                        top_candidates.emplace(dis, neighbor_id);
                        top_candidates.pop();
                        lowerBound = top_candidates.top().first;
                        // }
                    }
                }
            }
            while (top_candidates.size() > query_k)
                top_candidates.pop();
            return top_candidates;
        }

        void DFS_new1(std::vector<int> &QL, std::vector<int> &QR, int pid, int edge_limit, int deep, std::vector<tableint> &edges)
        {
            int Qsize = QL.size();
            if (deep == 0 || Qsize <= 1)
            {
                return;
            }
            int start_idx = 0;
            int end_idx = Qsize - 1;
            int left = QL[0];
            int right = QR[end_idx];
            TreeNodeLabel *cur_node = nullptr;
            TreeNodeLabel *nxt_node = tree->root;
            cur_node = nxt_node;
            bool contain = false;
            while (cur_node)
            {
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
                        if (GetOverLap(cur_node->lbound, cur_node->rbound, left, right) == GetOverLap(nxt_node->lbound, nxt_node->rbound, left, right))
                        {
                            cur_node = nxt_node;
                            contain = true;
                        }
                    }
                } while (contain);
                int *data = (int *)get_linklist(pid, cur_node->depth);
                size_t size = getListCount((linklistsizeint *)data);
                for (int j = 1; j <= size; ++j)
                {
                    int neighborId = *(data + j);
                    if (neighborId < left || neighborId > right)
                        continue;
                    // count--;
                    if (flags[neighborId])
                    {
                        edges.emplace_back(neighborId);
                    }
                    else
                    {
                        DFS_new1(QL, QR, neighborId, edge_limit, deep - 1, edges);
                    }
                    if (edges.size() >= edge_limit)
                    {
                        return;
                    }
                }
                if (nxt_node == nullptr)
                {
                    return;
                }
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
                {
                    return;
                }

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
            }
        }


        // 带缓存的 NodePossible：每个查询分配一个 bitmap 保存是否已计算相交
        inline bool NodePossibleCached(TreeNodeLabel *node, const Bitmap &label_bitmap, std::vector<uint8_t> &node_intersect_cache)
        {
            int nid = node->node_id;
            uint8_t c = node_intersect_cache[nid];
            if (c != 0)
                return c == 1; // 1=相交, 2=不相交

            // 首次计算
            bool res = false;
            for (size_t d = 0; d < node->bmps.size(); d++)
            {
                if (node->bmps[d].has_intersection(label_bitmap))
                {
                    res = true;
                    break;
                }
            }
            node_intersect_cache[nid] = res ? 1 : 2;
            return res;
        }

        // SelectEdgeLabel: 参数改为 Bitmap
        std::vector<tableint> SelectEdgeLabel(int pid, int ql, int qr, int edge_limit,
                                              std::vector<bool> &visited_set,
                                              const Bitmap &label_bitmap, std::vector<uint8_t> &node_intersect_cache)
        {
            TreeNodeLabel *cur_node = nullptr, *nxt_node = tree->root;
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
                if (!NodePossibleCached(cur_node, label_bitmap, node_intersect_cache))
                    return selected_edges;
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
                            }
                            else
                            {
                                if (!NodePossibleCached(cur_node->childs[i], label_bitmap, node_intersect_cache))
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
                    if (visited_set[neighborId])
                        continue;
                    if (!flags[neighborId])
                        continue;
                    selected_edges.emplace_back(neighborId);
                    if (selected_edges.size() == edge_limit)
                        return selected_edges;
                }
            } while (nxt_node);
            return selected_edges;
        }

        // DFSLabel: 参数改为 Bitmap
        void DFSLabel(int pid, int ql, int qr, int edge_limit, int deep,
                      std::vector<tableint> &edges, const Bitmap &label_bitmap, std::vector<uint8_t> &node_intersect_cache)
        {
            TreeNodeLabel *cur_node = tree->root;
            TreeNodeLabel *nxt_node = nullptr;
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

            while (cur_node)
            {
                if (!NodePossibleCached(cur_node, label_bitmap, node_intersect_cache))
                    break;
                bool inters = false;
                if (cur_node->childs.size() == 0)
                    nxt_node = nullptr;
                else
                {
                    for (int i = 0; i < cur_node->childs.size(); ++i)
                    {
                        if (cur_node->childs[i]->lbound <= pid && cur_node->childs[i]->rbound >= pid)
                        {
                            nxt_node = cur_node->childs[i];
                        }
                        else
                        {
                            if (NodePossibleCached(cur_node->childs[i], label_bitmap, node_intersect_cache))
                                inters = true;
                        }
                    }
                }

                if (inters)
                {
                    int *data = (int *)get_linklist(pid, cur_node->depth);
                    size_t size = getListCount((linklistsizeint *)data);
                    for (int j = 1; j <= size; ++j)
                    {
                        int neighborId = *(data + j);
                        if (flags[neighborId])
                        {
                            edges.emplace_back(neighborId);
                        }
                        else if (deep > 1)
                        {
                            DFSLabel(neighborId, ql, qr, edge_limit, deep - 1, edges, label_bitmap, node_intersect_cache);
                        }
                        if (edges.size() >= edge_limit)
                        {
                            return;
                        }
                    }
                }
                cur_node = nxt_node;
            }
        }

        // TopDown_nodeentries_search_Label: 参数改为 Bitmap
        std::priority_queue<PFI> TopDown_nodeentries_search_Label(std::vector<TreeNodeLabel *> &filterednodes,
                                                                  const void *query_data,
                                                                  int l, int r, int ef, int query_k, int edge_limit,
                                                                  const Bitmap &label_bitmap, std::vector<uint8_t> &node_intersect_cache)
        {
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine e(seed);
            std::priority_queue<PFI, std::vector<PFI>, std::greater<PFI>> candidate_set;
            std::priority_queue<PFI> top_candidates;
            std::vector<bool> visited_set(max_elements_, false);
            float lowerBound = std::numeric_limits<float>::max();
            for (auto u : filterednodes)
            {
                std::uniform_int_distribution<int> u_start(u->lbound, u->rbound);
                int pid = u_start(e);
                if (visited_set[pid])
                {
                    continue;
                }
                visited_set[pid] = true;
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
                auto selected_edges = SelectEdgeLabel(current_pid, l, r, edge_limit, visited_set, label_bitmap, node_intersect_cache);
                int num_edges = selected_edges.size();
                for (int i = 0; i < std::min(num_edges, 3); ++i)
                {
                    memory::mem_prefetch_L1(getDataByInternalId(selected_edges[i]), this->prefetch_lines);
                }
                for (int i = 0; i < num_edges; ++i)
                {
                    int neighbor_id = selected_edges[i];
                    if (visited_set[neighbor_id])
                        continue;
                    visited_set[neighbor_id] = true;
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
                DFSLabel(current_pid, l, r, edge_limit / 2, 2, edges, label_bitmap, node_intersect_cache);
                int num_edges1 = edges.size();
                for (int i = 0; i < std::min(num_edges1, 3); ++i)
                {
                    memory::mem_prefetch_L1(getDataByInternalId(edges[i]), this->prefetch_lines);
                }
                for (int i = 0; i < num_edges1; ++i)
                {
                    int neighbor_id = edges[i];
                    if (visited_set[neighbor_id])
                        continue;
                    visited_set[neighbor_id] = true;
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
            while (top_candidates.size() > query_k)
                top_candidates.pop();
            return top_candidates;
        }

        std::priority_queue<PFI> TopDown_nodeentries_search_filter(std::vector<TreeNodeLabel *> &filterednodes, const void *query_data, int l, int r, int ef, int query_k, int edge_limit)
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

        // 全局/类成员：标签值 -> [左边界下标, 右边界下标]
        // 标签范围 0~999，数组大小 1000；{-1,-1} 表示该标签不存在
        std::vector<std::pair<int, int>> label_range_map;

        void precompute_label_ranges()
        {
            if (storage->label_arr.empty())
                return;

            // 初始化数组：1000个元素，初值 {-1,-1} 表示标签不存在
            label_range_map.assign(1000, std::make_pair(-1, -1));

            // 提取数据集中所有唯一标签
            std::vector<int> unique_labels(storage->label_arr.begin(), storage->label_arr.end());
            std::sort(unique_labels.begin(), unique_labels.end());
            auto last = std::unique(unique_labels.begin(), unique_labels.end());
            unique_labels.erase(last, unique_labels.end());

            // 每个标签仅做一次 equal_range，预存区间
            for (int label : unique_labels)
            {
                if (label < 0 || label >= 1000)
                    continue;
                auto pp = std::equal_range(storage->label_arr.begin(), storage->label_arr.end(), label);
                int l = static_cast<int>(std::distance(storage->label_arr.begin(), pp.first));
                int r = static_cast<int>(std::distance(storage->label_arr.begin(), pp.second)) - 1;
                label_range_map[label] = {l, r};
            }
        }


        void label_or(std::vector<int> &SearchEF, std::string saveprefix, int edge_limit,
                      const std::string &label_json, const std::string &gt_json,
                      const std::vector<int> *vid_map = nullptr)
        {
            std::vector<std::vector<int>> label_sets = fan_parse_json_array_of_array(label_json);
            std::vector<std::vector<int>> gt = fan_parse_gt_json(gt_json, storage->query_nb, storage->query_K);

            if (flags.size() < max_elements_)
                flags.resize(max_elements_, false);

            std::string savepath = saveprefix + "_label_or.csv";
            CheckPath(savepath);
            std::ofstream outfile(savepath);
            if (!outfile.is_open())
                throw Exception("cannot open " + savepath);

            for (auto ef : SearchEF)
            {
                int tp = 0;
                float searchtime = 0;
                int missing_label_cnt = 0;
                int empty_query_cnt = 0;

                metric_hops = 0;
                metric_distance_computations = 0;

                for (int i = 0; i < storage->query_nb; i++)
                {
                    std::fill(flags.begin(), flags.end(), false); // 可选：清理脏数据
                    // 1. 排序去重标签
                    std::vector<int> labels = label_sets[i];
                    std::sort(labels.begin(), labels.end());
                    labels.erase(std::unique(labels.begin(), labels.end()), labels.end());

                    if (labels.empty())
                    {
                        empty_query_cnt++;
                        continue;
                    }
                    for (int label : labels)
                    {
                        auto pp = std::equal_range(storage->label_arr.begin(),
                                                   storage->label_arr.end(), label);
                        if (pp.first == pp.second)
                            continue; // 该标签不存在，跳过
                        int cur_l = (int)(pp.first - storage->label_arr.begin());
                        int cur_r = (int)(pp.second - storage->label_arr.begin()) - 1;
                        std::fill(flags.begin() + cur_l, flags.begin() + cur_r + 1, true);
                    }
                    // 2. 把标签集合转成 Bitmap
                    Bitmap label_bitmap;
                    for (int label : labels)
                    {
                        if (label < 0)
                        {
                            missing_label_cnt++;
                            continue;
                        }
                        label_bitmap.set(label);
                    }

                    if (label_bitmap.blocks.empty())
                    {
                        empty_query_cnt++;
                        continue;
                    }

                    std::vector<uint8_t> node_intersect_cache;
                    node_intersect_cache.resize(TreeNodeLabel::next_id, 0);

                    // 3. 调用 TopDown_nodeentries_search_Label
                    int k = storage->query_K;
                    timeval t1, t2;
                    gettimeofday(&t1, NULL);
                    // 只需查最小值和最大值两个端点
                    auto pp_min = std::equal_range(storage->label_arr.begin(),
                                                   storage->label_arr.end(), labels.front());
                    auto pp_max = std::equal_range(storage->label_arr.begin(),
                                                   storage->label_arr.end(), labels.back());

                    int l = static_cast<int>(std::distance(storage->label_arr.begin(), pp_min.first));
                    int r = static_cast<int>(std::distance(storage->label_arr.begin(), pp_max.second)) - 1;

                    // 一次性填充整个区间
                    std::vector<TreeNodeLabel *> filterednodes = tree->range_filter(tree->root, l, r);

                    std::priority_queue<PFI> res = TopDown_nodeentries_search_Label(
                        filterednodes,
                        storage->query_points[i].data(),
                        l, r,
                        ef, k, edge_limit,
                        label_bitmap, node_intersect_cache);

                    gettimeofday(&t2, NULL);
                    auto duration = GetTime(t1, t2);
                    searchtime += duration;

                    std::map<int, int> record;
                    while (!res.empty())
                    {
                        auto x = res.top().second;
                        res.pop();
                        if (vid_map != nullptr)
                        {
                            if (x < 0 || (size_t)x >= vid_map->size())
                                throw Exception("label_or: pid " + std::to_string(x) +
                                                " out of vid_map size " +
                                                std::to_string(vid_map->size()));
                            x = (*vid_map)[x];
                        }
                        if (record.count(x))
                            continue;
                        record[x] = 1;
                        if (std::find(gt[i].begin(), gt[i].end(), x) != gt[i].end())
                            tp++;
                    }
                }

                float recall = 1.0f * tp / storage->query_nb / storage->query_K;
                float qps = storage->query_nb / searchtime;
                float dco = metric_distance_computations * 1.0f / storage->query_nb;
                float hop = metric_hops * 1.0f / storage->query_nb;
                outfile << ef << "," << recall << "," << qps << "," << dco << "," << hop << std::endl;
                std::cout << "ef=" << ef << " recall=" << recall << " qps=" << qps
                          << " missing_label_refs=" << missing_label_cnt
                          << " empty_queries=" << empty_query_cnt << std::endl;
                if (recall < 0.9f)
                    break;
            }
            outfile.close();
        }
        void searchfilter(std::vector<int> &SearchEF, std::string saveprefix, int edge_limit,
                          const std::string &label_json, const std::string &gt_json)
        {
            std::vector<std::vector<int>> label_sets = fan_parse_json_array_of_array(label_json);
            std::vector<std::vector<int>> gt = fan_parse_gt_json(gt_json, storage->query_nb, storage->query_K);

            std::string savepath = saveprefix + "filter.csv";
            std::cout << "Savepath: " << savepath << std::endl;
            CheckPath(savepath);

            std::ofstream outfile(savepath);
            if (!outfile.is_open())
            {
                throw Exception("cannot open " + savepath);
            }

            for (auto ef : SearchEF)
            {
                int tp = 0;
                float searchtime = 0;
                int missing_label_cnt = 0;
                int empty_query_cnt = 0;

                metric_hops = 0;
                metric_distance_computations = 0;

                for (int i = 0; i < storage->query_nb; i++)
                {
                    std::fill(flags.begin(), flags.end(), false);

                    // 1. 排序去重标签
                    std::vector<int> labels = label_sets[i];
                    std::sort(labels.begin(), labels.end());
                    labels.erase(std::unique(labels.begin(), labels.end()), labels.end());

                    if (labels.empty())
                    {
                        empty_query_cnt++;
                        continue;
                    }

                    for (int label : labels)
                    {
                        auto pp = std::equal_range(storage->label_arr.begin(),
                                                   storage->label_arr.end(), label);
                        if (pp.first == pp.second)
                            continue; // 该标签不存在，跳过
                        int cur_l = (int)(pp.first - storage->label_arr.begin());
                        int cur_r = (int)(pp.second - storage->label_arr.begin()) - 1;
                        std::fill(flags.begin() + cur_l, flags.begin() + cur_r + 1, true);
                    }
                    timeval t1, t2;
                    gettimeofday(&t1, NULL);

                    // 只需查最小值和最大值两个端点
                    auto pp_min = std::equal_range(storage->label_arr.begin(),
                                                   storage->label_arr.end(), labels.front());
                    auto pp_max = std::equal_range(storage->label_arr.begin(),
                                                   storage->label_arr.end(), labels.back());

                    int l = static_cast<int>(std::distance(storage->label_arr.begin(), pp_min.first));
                    int r = static_cast<int>(std::distance(storage->label_arr.begin(), pp_max.second)) - 1;

                    std::vector<TreeNodeLabel *> node = tree->range_filter(tree->root, l, r);
                    std::priority_queue<PFI> res = TopDown_nodeentries_search_filter(
                        node, storage->query_points[i].data(), l, r, ef, storage->query_K, edge_limit);

                    gettimeofday(&t2, NULL);
                    searchtime += GetTime(t1, t2);

                    std::map<int, int> record;
                    while (res.size())
                    {
                        auto x = res.top().second;
                        res.pop();
                        if (record.count(x))
                            throw Exception("repetitive search results");
                        record[x] = 1;
                        if (std::find(gt[i].begin(), gt[i].end(), x) != gt[i].end())
                        {
                            tp++;
                        }
                    }
                }

                float recall = 1.0 * tp / storage->query_nb / storage->query_K;
                float qps = storage->query_nb / searchtime;
                float dco = metric_distance_computations * 1.0 / storage->query_nb;
                float hop = metric_hops * 1.0 / storage->query_nb;
                outfile << ef << "," << recall << "," << qps << "," << dco << "," << hop << std::endl;
                std::cout << "ef=" << ef << " recall=" << recall << " qps=" << qps << std::endl;
                if (recall < 0.9)
                {
                    break;
                }
            }

            outfile.close();
        }
    };
}