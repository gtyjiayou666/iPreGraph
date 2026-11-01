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

    // Parse nested integer arrays while preserving the outer groups.
    inline std::vector<std::vector<int>> read_nested_integer_arrays_json(const std::string &filename)
    {
        const std::string s = read_text_file(filename);
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
        if (i < s.size() && s[i] == ']') // Empty outer array.
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
                ++i; // Allow an empty label list.
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

    // Parse ground-truth JSON in either supported layout:
    // nested arrays contain one ID list per query;
    // a flat array contains query_nb * query_K IDs in query order.
    inline std::vector<std::vector<int>> load_ground_truth_json(const std::string &filename, int query_nb, int query_K)
    {
        const std::string s = read_text_file(filename);
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
            std::vector<std::vector<int>> gt = read_nested_integer_arrays_json(filename);
            if ((int)gt.size() != query_nb)
                throw Exception("ground truth file " + filename + ": " + std::to_string(gt.size()) +
                                " groups, but query_nb = " + std::to_string(query_nb));
            std::cout << "loaded nested ground truth: " << gt.size() << " groups from " << filename << std::endl;
            return gt;
        }

        if (query_nb <= 0 || query_K <= 0)
            throw Exception("ground truth file " + filename + ": flat format needs valid query_nb and query_K");
        std::vector<int> v = parse_integer_values(s, filename);
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
                    linklistsizeint *data = get_adjacency_list(pid, layer);
                    edgefile.read((char *)data, sizeof(tableint));
                    int size = get_neighbor_count(data);
                    if (size > M_out)
                        throw Exception("real linklist size is bigger than defined M_out");
                    for (int i = 0; i < size; i++)
                    {
                        char *current_neighbor_ = (char *)(data + 1 + i);
                        edgefile.read(current_neighbor_, sizeof(tableint));
                    }
                }

                char *data = get_vector_data(pid);
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

        inline char *get_vector_data(tableint internal_id) const
        {
            return (data_memory_ + internal_id * size_data_per_element_ + offsetData_);
        }

        linklistsizeint *get_adjacency_list(tableint internal_id, int layer) const
        {
            return (linklistsizeint *)(data_memory_ + internal_id * size_data_per_element_ + layer * size_links_per_layer_);
        }

        int get_neighbor_count(linklistsizeint *ptr) const
        {
            return *((int *)ptr);
        }

        int interval_overlap_span(int l, int r, int ql, int qr)
        {
            int L = std::max(l, ql);
            int R = std::min(r, qr);
            return R - L + 1;
        }

        std::vector<tableint> select_intra_range_edges(int pid, int ql, int qr, int edge_limit, searcher::Bitset<uint64_t> &visited_set)
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
                        if (interval_overlap_span(cur_node->lbound, cur_node->rbound, ql, qr) == interval_overlap_span(nxt_node->lbound, nxt_node->rbound, ql, qr))
                        {
                            cur_node = nxt_node;
                            contain = true;
                        }
                    }
                } while (contain);

                int *data = (int *)get_adjacency_list(pid, cur_node->depth);
                size_t size = get_neighbor_count((linklistsizeint *)data);

                for (size_t j = 1; j <= size; ++j)
                {
                    int neighborId = *(data + j);
                    if (neighborId < ql || neighborId > qr)
                        continue;
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
            int index; // Gap index i, between Q_R[i] and Q_L[i + 1].
            int len;
        };
        struct CompareGap
        {
            bool operator()(const GapInfo &a, const GapInfo &b)
            {
                return a.len < b.len; // Max-heap: the longest gap comes first.
            }
        };
        void merge_small_gaps(const std::vector<int> &Q_L, const std::vector<int> &Q_R,
                              std::vector<int> &Q_L1, std::vector<int> &Q_R1,
                              std::vector<int> &mapping, double p)
        {
            int n = Q_L.size();
            mapping.assign(n, 0);

            double total_data = Q_R[n - 1] - Q_L[0] + 1;
            double total_gaps = 0;

            std::priority_queue<GapInfo, std::vector<GapInfo>, CompareGap> max_heap_gaps;

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
                return;
            }
            int current_gaps = 0;
            while (!max_heap_gaps.empty())
            {
                GapInfo gap = max_heap_gaps.top();
                current_gaps = current_gaps + gap.len;
                if (current_gaps >= s)
                {
                    is_keeper[gap.index] = true;
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
            // Emit the final range.
            Q_L1.push_back(current_l);
            Q_R1.push_back(current_r);
        }

        std::priority_queue<PFI> search_merged_ranges(std::vector<TreeNodeLabel *> &filterednodes, const void *query_data, int ef, int query_k, std::vector<int> &Q_L, std::vector<int> &Q_R, int edge_limit, int id)
        {
            std::vector<int> Q_L1, Q_R1, mapping;
            merge_small_gaps(Q_L, Q_R, Q_L1, Q_R1, mapping, 3);
            int range_num = Q_L1.size();

            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine e(seed);

            std::priority_queue<PFIORPRO, std::vector<PFIORPRO>, Compare> candidate_set;
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
                char *ep_data = get_vector_data(pid);
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
                }
                auto selected_edges = select_intra_range_edges(current_pid, Q_L1[range_id], Q_R1[range_id], edge_limit, visited_set);
                int num_edges = selected_edges.size();
                for (int i = 0; i < std::min(num_edges, 3); ++i)
                {
                    memory::mem_prefetch_L1(get_vector_data(selected_edges[i]), this->prefetch_lines);
                }
                for (int i = 0; i < num_edges; ++i)
                {
                    int neighbor_id = selected_edges[i];
                    if (visited_set.get(neighbor_id))
                        continue;
                    visited_set.set(neighbor_id);
                    char *neighbor_data = get_vector_data(neighbor_id);
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
                select_cross_range_edges_from_ranges(Q_L, Q_R, current_pid, edge_limit / 2, 2, edges);
                int num_edges1 = edges.size();
                for (int i = 0; i < std::min(num_edges1, 3); ++i)
                {
                    memory::mem_prefetch_L1(get_vector_data(edges[i]), this->prefetch_lines);
                }
                for (int i = 0; i < num_edges1; ++i)
                {
                    int neighbor_id = edges[i];
                    if (visited_set.get(neighbor_id))
                        continue;
                    visited_set.set(neighbor_id);
                    char *neighbor_data = get_vector_data(neighbor_id);
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
            while (top_candidates.size() > query_k)
                top_candidates.pop();
            return top_candidates;
        }

        void select_cross_range_edges_from_ranges(std::vector<int> &QL, std::vector<int> &QR, int pid, int edge_limit, int deep, std::vector<tableint> &edges)
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
                        if (interval_overlap_span(cur_node->lbound, cur_node->rbound, left, right) == interval_overlap_span(nxt_node->lbound, nxt_node->rbound, left, right))
                        {
                            cur_node = nxt_node;
                            contain = true;
                        }
                    }
                } while (contain);
                int *data = (int *)get_adjacency_list(pid, cur_node->depth);
                size_t size = get_neighbor_count((linklistsizeint *)data);
                for (int j = 1; j <= size; ++j)
                {
                    int neighborId = *(data + j);
                    if (neighborId < left || neighborId > right)
                        continue;
                    if (flags[neighborId])
                    {
                        edges.emplace_back(neighborId);
                    }
                    else
                    {
                        select_cross_range_edges_from_ranges(QL, QR, neighborId, edge_limit, deep - 1, edges);
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


        // Cache whether a node's label summary intersects this query's labels.
        inline bool may_match_labels_cached(TreeNodeLabel *node, const Bitmap &label_bitmap, std::vector<uint8_t> &node_intersect_cache)
        {
            int nid = node->node_id;
            uint8_t c = node_intersect_cache[nid];
            if (c != 0)
                return c == 1; // 1: intersecting; 2: disjoint.

            // Compute the intersection on the first visit.
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

        // Select direct qualifying neighbors using cached segment summaries.
        std::vector<tableint> select_direct_edges(int pid, int ql, int qr, int edge_limit,
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
                        if (interval_overlap_span(cur_node->lbound, cur_node->rbound, ql, qr) == interval_overlap_span(nxt_node->lbound, nxt_node->rbound, ql, qr))
                        {
                            cur_node = nxt_node;
                            contain = true;
                        }
                    }
                } while (contain);
                if (!may_match_labels_cached(cur_node, label_bitmap, node_intersect_cache))
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
                                if (!may_match_labels_cached(cur_node->childs[i], label_bitmap, node_intersect_cache))
                                    contain = true;
                            }
                        }
                        if (contain)
                            cur_node = nxt_node;
                    }
                } while (contain);
                int *data = (int *)get_adjacency_list(pid, cur_node->depth);
                size_t size = get_neighbor_count((linklistsizeint *)data);
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

        // Find qualifying endpoints through bounded-depth graph expansion.
        void select_cross_range_edges(int pid, int ql, int qr, int edge_limit, int deep,
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
                    if (interval_overlap_span(cur_node->lbound, cur_node->rbound, ql, qr) == interval_overlap_span(nxt_node->lbound, nxt_node->rbound, ql, qr))
                    {
                        cur_node = nxt_node;
                        contain = true;
                    }
                }
            } while (contain);

            while (cur_node)
            {
                if (!may_match_labels_cached(cur_node, label_bitmap, node_intersect_cache))
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
                            if (may_match_labels_cached(cur_node->childs[i], label_bitmap, node_intersect_cache))
                                inters = true;
                        }
                    }
                }

                if (inters)
                {
                    int *data = (int *)get_adjacency_list(pid, cur_node->depth);
                    size_t size = get_neighbor_count((linklistsizeint *)data);
                    for (int j = 1; j <= size; ++j)
                    {
                        int neighborId = *(data + j);
                        if (flags[neighborId])
                        {
                            edges.emplace_back(neighborId);
                        }
                        else if (deep > 1)
                        {
                            select_cross_range_edges(neighborId, ql, qr, edge_limit, deep - 1, edges, label_bitmap, node_intersect_cache);
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

        // Best-first label search using cached segment summaries.
        std::priority_queue<PFI> search_pdg(std::vector<TreeNodeLabel *> &filterednodes,
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
                char *ep_data = get_vector_data(pid);
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
                auto selected_edges = select_direct_edges(current_pid, l, r, edge_limit, visited_set, label_bitmap, node_intersect_cache);
                int num_edges = selected_edges.size();
                for (int i = 0; i < std::min(num_edges, 3); ++i)
                {
                    memory::mem_prefetch_L1(get_vector_data(selected_edges[i]), this->prefetch_lines);
                }
                for (int i = 0; i < num_edges; ++i)
                {
                    int neighbor_id = selected_edges[i];
                    if (visited_set[neighbor_id])
                        continue;
                    visited_set[neighbor_id] = true;
                    char *neighbor_data = get_vector_data(neighbor_id);
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
                select_cross_range_edges(current_pid, l, r, edge_limit / 2, 2, edges, label_bitmap, node_intersect_cache);
                int num_edges1 = edges.size();
                for (int i = 0; i < std::min(num_edges1, 3); ++i)
                {
                    memory::mem_prefetch_L1(get_vector_data(edges[i]), this->prefetch_lines);
                }
                for (int i = 0; i < num_edges1; ++i)
                {
                    int neighbor_id = edges[i];
                    if (visited_set[neighbor_id])
                        continue;
                    visited_set[neighbor_id] = true;
                    char *neighbor_data = get_vector_data(neighbor_id);
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

        std::priority_queue<PFI> search_post_filter(std::vector<TreeNodeLabel *> &filterednodes, const void *query_data, int l, int r, int ef, int query_k, int edge_limit)
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
                char *ep_data = get_vector_data(pid);
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
                auto selected_edges = select_intra_range_edges(current_pid, l, r, edge_limit, visited_set);
                int num_edges = selected_edges.size();
                for (int i = 0; i < std::min(num_edges, 3); ++i)
                {
                    memory::mem_prefetch_L1(get_vector_data(selected_edges[i]), this->prefetch_lines);
                }
                for (int i = 0; i < num_edges; ++i)
                {
                    int neighbor_id = selected_edges[i];
                    if (visited_set.get(neighbor_id))
                        continue;
                    visited_set.set(neighbor_id);
                    char *neighbor_data = get_vector_data(neighbor_id);
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

        void benchmark_pdg(std::vector<int> &SearchEF, std::string saveprefix, int edge_limit,
                      const std::string &label_json, const std::string &gt_json,
                      const std::vector<int> *vid_map = nullptr)
        {
            std::vector<std::vector<int>> label_sets = read_nested_integer_arrays_json(label_json);
            std::vector<std::vector<int>> gt = load_ground_truth_json(gt_json, storage->query_nb, storage->query_K);

            if (flags.size() < max_elements_)
                flags.resize(max_elements_, false);

            std::string savepath = saveprefix + "_label_or.csv";
            ensure_parent_directory(savepath);
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
                    std::fill(flags.begin(), flags.end(), false); // Reset eligibility flags for this query.
                    // Sort and deduplicate query labels.
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
                            continue; // Skip labels absent from the data.
                        int cur_l = (int)(pp.first - storage->label_arr.begin());
                        int cur_r = (int)(pp.second - storage->label_arr.begin()) - 1;
                        std::fill(flags.begin() + cur_l, flags.begin() + cur_r + 1, true);
                    }
                    // Convert the label set to a bitmap.
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

                    // Run the summary-guided label search.
                    int k = storage->query_K;
                    timeval t1, t2;
                    gettimeofday(&t1, NULL);
                    // Locate the cover using the smallest and largest query labels.
                    auto pp_min = std::equal_range(storage->label_arr.begin(),
                                                   storage->label_arr.end(), labels.front());
                    auto pp_max = std::equal_range(storage->label_arr.begin(),
                                                   storage->label_arr.end(), labels.back());

                    int l = static_cast<int>(std::distance(storage->label_arr.begin(), pp_min.first));
                    int r = static_cast<int>(std::distance(storage->label_arr.begin(), pp_max.second)) - 1;

                    // Obtain the tree nodes covering the entire interval.
                    std::vector<TreeNodeLabel *> filterednodes = tree->find_covering_nodes(tree->root, l, r);

                    std::priority_queue<PFI> res = search_pdg(
                        filterednodes,
                        storage->query_points[i].data(),
                        l, r,
                        ef, k, edge_limit,
                        label_bitmap, node_intersect_cache);

                    gettimeofday(&t2, NULL);
                    auto duration = elapsed_seconds(t1, t2);
                    searchtime += duration;

                    std::map<int, int> record;
                    while (!res.empty())
                    {
                        auto x = res.top().second;
                        res.pop();
                        if (vid_map != nullptr)
                        {
                            if (x < 0 || (size_t)x >= vid_map->size())
                                throw Exception("benchmark_pdg: pid " + std::to_string(x) +
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
        void benchmark_post_filter(std::vector<int> &SearchEF, std::string saveprefix, int edge_limit,
                          const std::string &label_json, const std::string &gt_json)
        {
            std::vector<std::vector<int>> label_sets = read_nested_integer_arrays_json(label_json);
            std::vector<std::vector<int>> gt = load_ground_truth_json(gt_json, storage->query_nb, storage->query_K);

            std::string savepath = saveprefix + "filter.csv";
            std::cout << "Savepath: " << savepath << std::endl;
            ensure_parent_directory(savepath);

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

                    // Sort and deduplicate query labels.
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
                            continue; // Skip labels absent from the data.
                        int cur_l = (int)(pp.first - storage->label_arr.begin());
                        int cur_r = (int)(pp.second - storage->label_arr.begin()) - 1;
                        std::fill(flags.begin() + cur_l, flags.begin() + cur_r + 1, true);
                    }
                    timeval t1, t2;
                    gettimeofday(&t1, NULL);

                    // Locate the cover using the smallest and largest query labels.
                    auto pp_min = std::equal_range(storage->label_arr.begin(),
                                                   storage->label_arr.end(), labels.front());
                    auto pp_max = std::equal_range(storage->label_arr.begin(),
                                                   storage->label_arr.end(), labels.back());

                    int l = static_cast<int>(std::distance(storage->label_arr.begin(), pp_min.first));
                    int r = static_cast<int>(std::distance(storage->label_arr.begin(), pp_max.second)) - 1;

                    std::vector<TreeNodeLabel *> node = tree->find_covering_nodes(tree->root, l, r);
                    std::priority_queue<PFI> res = search_post_filter(
                        node, storage->query_points[i].data(), l, r, ef, storage->query_K, edge_limit);

                    gettimeofday(&t2, NULL);
                    searchtime += elapsed_seconds(t1, t2);

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
