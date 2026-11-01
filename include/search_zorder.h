#pragma once

#include <vector>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <queue>
#include <random>
#include <utility>
// #include "segment_tree_utils.h"
#include "multi_attribute_utils.h"
#include "search_containers.h"
#include "memory_utils.h"
#include <bitset>
#include <stack>
#include <cmath>
#include <climits>

namespace iPreGraph_multi
{
    using iPreGraph::tableint;
    using iPreGraph::linklistsizeint;
    // Query a legacy graph whose rows are sorted by the shared Morton encoder.
    // MBR checks filter candidates without changing their base-row IDs.
    template <typename dist_t>
    class iPreGraph_Search_Zorder
    {
    public:
        DataLoader *storage;
        SegmentTreeZorder *tree;
        size_t max_elements_{0};
        size_t dim_{0};
        size_t attr_dim_{0}; // 属性维度（用于 Z-order / MBR / 属性过滤）
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

        iPreGraph_Search_Zorder(std::string edgefilename, DataLoader *store, int M) : storage(store)
        {
            std::ifstream edgefile(edgefilename, std::ios::in | std::ios::binary);
            if (!edgefile.is_open())
                throw Exception("cannot open " + edgefilename);

            max_elements_ = storage->data_nb;
            dim_ = storage->Dim;
            attr_dim_ = storage->attr_nb; // 新字段：属性个数（不再假定为 2）
            std::cout << "attr_dim:" << attr_dim_ << std::endl;
            tree = new SegmentTreeZorder(max_elements_, attr_dim_);
            tree->BuildTree(tree->root, storage->attributes);

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
                std::memcpy(data, reinterpret_cast<char *>(storage->data_points[pid].data()), size_in_bytes);
            }
            edgefile.close();
        }

        ~iPreGraph_Search_Zorder()
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


        std::vector<tableint> SelectEdge(int pid, int ql, int qr, int edge_limit, searcher::Bitset<uint64_t> &visited_set)
        {
            TreeNode *cur_node = nullptr, *nxt_node = tree->root;
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

        std::vector<tableint> SelectEdgeHybrid(int pid, int ql, int qr, int edge_limit, searcher::Bitset<uint64_t> &visited_set, MBR &wp, const std::vector<char> &filter_ids_map)
        {
            TreeNode *cur_node = nullptr, *nxt_node = tree->root;
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
                if (!cur_node->mbr.Intersects(wp))
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
                        for (int i = 0; i < cur_node->childs.size(); ++i)
                        {
                            if (cur_node->childs[i]->lbound <= pid && cur_node->childs[i]->rbound >= pid)
                            {
                                nxt_node = cur_node->childs[i];
                            }
                            else
                            {
                                if (!cur_node->childs[i]->mbr.Intersects(wp))
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
                    if (selected_edges.size() == edge_limit)
                        return selected_edges;
                }
            } while (nxt_node);
            return selected_edges;
        }

        void DFS(int pid, int ql, int qr, int edge_limit, int deep, std::vector<tableint> &edges, MBR &wp, const std::vector<char> &filter_ids_map)
        {
            TreeNode *cur_node = tree->root;
            TreeNode *nxt_node = nullptr;
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
                if (!cur_node->mbr.Intersects(wp))
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
                    for (int i = 0; i < cur_node->childs.size(); ++i)
                    {
                        if (cur_node->childs[i]->lbound <= pid && cur_node->childs[i]->rbound >= pid)
                        {
                            nxt_node = cur_node->childs[i];
                            // continue;
                        }
                        else
                        {
                            if (cur_node->childs[i]->mbr.Intersects(wp))
                            {
                                inters = true;
                                // break;
                            }
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
                        // if (neighborId < ql || neighborId > qr)
                        //     continue;
                        // count--;
                        if (filter_ids_map[neighborId] != 0)
                        {
                            edges.emplace_back(neighborId);
                        }
                        else if (deep > 1)
                        {
                            DFS(neighborId, ql, qr, edge_limit, deep - 1, edges, wp, filter_ids_map);
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
        int MaxStep{20};
        std::vector<double> probability;
        void setprob()
        {
            if (MaxStep <= 0)
                throw Exception("MaxStep must be positive");
            probability.resize(MaxStep);

            for (int x = 0; x < MaxStep; x++)
            {
                probability[x] = 1.0 / (1.0 + std::exp(x));
            }
        }
        std::vector<std::pair<int, bool>> SelectEdgeMutil(int pid, int l, int r, int edge_limit, int current_step, searcher::Bitset<uint64_t> &visited_set, const std::vector<char> &filter_ids_map)
        {
            if (MaxStep <= 0 || probability.size() != static_cast<size_t>(MaxStep))
                setprob();
            if (current_step < -1)
                throw Exception("current_step must be at least -1");
            TreeNode *cur_node = nullptr, *nxt_node = tree->root;
            std::vector<std::pair<int, bool>> selected_edges;
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
                        if (GetOverLap(cur_node->lbound, cur_node->rbound, l, r) == GetOverLap(nxt_node->lbound, nxt_node->rbound, l, r))
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
                    if (neighborId < l || neighborId > r)
                        continue;
                    if (visited_set.get(neighborId))
                        continue;
                    const bool qualifying = filter_ids_map[neighborId] != 0;
                    if (!qualifying)
                    {
                        const int next_step = current_step + 1;
                        if (next_step >= MaxStep)
                            continue;
                        const double sample = static_cast<double>(std::rand()) / RAND_MAX;
                        if (sample >= probability[next_step])
                            continue;
                    }
                    selected_edges.emplace_back(neighborId, qualifying);
                    if (selected_edges.size() == edge_limit)
                        return selected_edges;
                }

            } while (cur_node->lbound < l || cur_node->rbound > r);

            return selected_edges;
        }
        // iRangeGraph multi-attribute search on the supplied position cover [l,r].
        // flags encode the full predicate; returned IDs are current Z-order row IDs.
        // Nonqualifying transit vertices are sampled with logistic probability.
        // A qualifying point resets the transit-step counter to -1. The explicit
        // seed controls entry selection; transit sampling uses the C rand() state.
        std::priority_queue<PFI> TopDown_nodeentries_search_mutil(
            std::vector<TreeNode *> &filterednodes, const void *query_data,
            int l, int r, int ef, int query_k, int edge_limit,
            const std::vector<char> &filter_ids_map,
            unsigned seed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count()))
        {
            if (ef <= 0 || query_k <= 0 || edge_limit <= 0)
                throw Exception("ef, query_k and edge limit must be positive");
            if (filter_ids_map.size() != max_elements_)
                throw Exception("predicate flags must match the index size");
            if (MaxStep <= 0 || probability.size() != static_cast<size_t>(MaxStep))
                setprob();
            std::priority_queue<PFI> top_candidates;
            if (l > r)
                return top_candidates;
            if (l < 0 || static_cast<size_t>(r) >= max_elements_)
                throw Exception("query cover lies outside the index");
            using Candidate = std::pair<float, std::pair<int, int>>;
            std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> candidate_set;
            searcher::Bitset<uint64_t> visited_set(max_elements_);
            std::default_random_engine entry_random(seed);

            // Initialize entries and the result threshold as in the filter search.
            float lowerBound = std::numeric_limits<float>::max();
            for (auto *node : filterednodes)
            {
                std::uniform_int_distribution<int> sample(node->lbound, node->rbound);
                const int pid = sample(entry_random);
                if (visited_set.get(pid))
                    continue;
                visited_set.set(pid);
                const float dis = fstdistfunc_(query_data, getDataByInternalId(pid), dist_func_param_);
                if (top_candidates.size() < static_cast<size_t>(ef) || dis < lowerBound)
                {
                    candidate_set.emplace(dis, std::make_pair(pid, -1));
                    if (filter_ids_map[pid] != 0)
                    {
                        top_candidates.emplace(dis, pid);
                        if (top_candidates.size() > static_cast<size_t>(ef))
                            top_candidates.pop();
                        lowerBound = top_candidates.top().first;
                    }
                }
            }

            while (!candidate_set.empty())
            {
                const auto current = candidate_set.top();
                ++metric_hops;
                if (current.first > lowerBound)
                    break;
                candidate_set.pop();
                const int pid = current.second.first;
                const int current_step = current.second.second;
                auto selected = SelectEdgeMutil(pid, l, r, edge_limit, current_step, visited_set, filter_ids_map);
                int num_edges = selected.size();
                for (int i = 0; i < std::min(num_edges, 3); ++i)
                {
                    memory::mem_prefetch_L1(getDataByInternalId(selected[i].first), this->prefetch_lines);
                }
                // Preserve selection order, as in the filter search.
                for (int i = 0; i < num_edges; ++i)
                {
                    const auto &neighbor = selected[i];
                    const int neighbor_id = neighbor.first;
                    const bool qualifying = neighbor.second;
                    if (visited_set.get(neighbor_id))
                        continue;
                    visited_set.set(neighbor_id);
                    const float dis = fstdistfunc_(query_data, getDataByInternalId(neighbor_id), dist_func_param_);
                    ++metric_distance_computations;
                    if (top_candidates.size() < static_cast<size_t>(ef) || dis < lowerBound)
                    {
                        int next_step = current_step + 1;
                        if (qualifying)
                        {
                            top_candidates.emplace(dis, neighbor_id);
                            next_step = -1;
                        }
                        candidate_set.emplace(dis, std::make_pair(neighbor_id, next_step));
                        if (top_candidates.size() > static_cast<size_t>(ef))
                            top_candidates.pop();
                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
            while (top_candidates.size() > static_cast<size_t>(query_k))
                top_candidates.pop();
            return top_candidates;
        }

        std::priority_queue<PFI> TopDown_nodeentries_search_zorder_and(std::vector<TreeNode *> &filterednodes, const void *query_data, int l, int r, int ef, int query_k, int edge_limit, MBR &wp, const std::vector<char> &filter_ids_map)
        {

            // timeval t1, t2, t3, t4;
            // float searchtime1 = 0, searchtime2 = 0;
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
                auto selected_edges = SelectEdgeHybrid(current_pid, l, r, edge_limit, visited_set, wp, filter_ids_map);
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
                DFS(current_pid, l, r, edge_limit / 2, 2, edges, wp, filter_ids_map);
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
            // std::cout << "searchtime1 : " << searchtime1 << ", searchtime2 : " << searchtime2 << std::endl;
            // std::cout << "s1 : " << s1 / count << std::endl;
            // std::cout << "over" << std::endl;
            while (top_candidates.size() > query_k)
                top_candidates.pop();
            return top_candidates;
        }

        std::priority_queue<PFI> TopDown_nodeentries_search_zorder_and_filter(std::vector<TreeNode *> &filterednodes, const void *query_data, int l, int r, int ef, int query_k, int edge_limit, MBR &wp, const std::vector<char> &filter_ids_map)
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
                        if (filter_ids_map[neighbor_id] != 0)
                        {
                            top_candidates.emplace(dis, neighbor_id);
                            lowerBound = top_candidates.top().first;
                        }
                    }
                    else if (dis < lowerBound)
                    {
                        candidate_set.emplace(dis, neighbor_id);
                        if (filter_ids_map[neighbor_id] != 0)
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

        void search_zorder(std::vector<int> &SearchEF, std::string saveprefix, int edge_limit)
        {
            const int N = storage->data_nb; // 数据点总数
            std::vector<char> filter_ids_map(N);
            for (auto range : storage->query_range_multi)
            {
                std::string domain = range.first;
                std::vector<std::vector<int>> &gt = storage->ground_truth[domain];
                std::string savepath = saveprefix + domain + "-zorder_ways.csv";
                std::cout << "Directory: " << savepath << std::endl;
                CheckPath(savepath);

                std::ofstream outfile(savepath);
                if (!outfile.is_open())
                {
                    throw Exception("cannot open " + savepath);
                }

                std::vector<int> HOP;
                std::vector<int> DCO;
                std::vector<float> QPS;
                std::vector<float> RECALL;

                for (auto ef : SearchEF)
                {
                    int cc = 0;
                    int cccount = 0;
                    int tp = 0;
                    float searchtime = 0;

                    metric_hops = 0;
                    metric_distance_computations = 0;

                    for (int i = storage->query_nb - 1; i >= 0; i--)
                    {
                        auto &cons = range.second[i];
                        int num_attrs = cons.attr_constraints.size(); // 属性个数，不再硬编码为2

                        // 构建 N 维 MBR
                        MBR wq;
                        wq.dim = num_attrs;
                        wq.min_vals.resize(num_attrs);
                        wq.max_vals.resize(num_attrs);
                        for (int d = 0; d < num_attrs; d++)
                        {
                            wq.min_vals[d] = cons.attr_constraints[d].first;
                            wq.max_vals[d] = cons.attr_constraints[d].second;
                        }

                        timeval t1, t2;
                        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
                        std::default_random_engine e(seed);

                        for (int xb = 0; xb < N; xb++)
                        {
                            bool in_range = true;
                            for (int d = 0; d < num_attrs; d++)
                            {
                                int v = storage->attributes[xb][d];
                                if (v < static_cast<int>(wq.min_vals[d]) ||
                                    v > static_cast<int>(wq.max_vals[d]))
                                {
                                    in_range = false;
                                    break;
                                }
                            }
                            filter_ids_map[xb] = static_cast<char>(in_range);
                            // if (in_range)
                            //     sel_i++;
                        }
                        gettimeofday(&t1, NULL);

                        // 构建 N 维 Z-order 编码的边界
                        std::vector<int> coords_min(num_attrs), coords_max(num_attrs);
                        for (int d = 0; d < num_attrs; d++)
                        {
                            coords_min[d] = wq.min_vals[d];
                            coords_max[d] = wq.max_vals[d];
                        }
                        MultiAttributeZCode z_min = CalculateMultiAttributeZOrder(coords_min);
                        MultiAttributeZCode z_max = CalculateMultiAttributeZOrder(coords_max);

                        auto it1 = std::lower_bound(
                            storage->multi_zorder.begin(), storage->multi_zorder.end(),
                            z_min, MultiAttributeZOrderLess);
                        int l = std::distance(storage->multi_zorder.begin(), it1);
                        auto it2 = std::upper_bound(
                            storage->multi_zorder.begin(), storage->multi_zorder.end(),
                            z_max, MultiAttributeZOrderLess);
                        if (it2 != storage->multi_zorder.begin())
                            --it2;
                        int r = std::distance(storage->multi_zorder.begin(), it2);

                        std::vector<TreeNode *> node = tree->range_filter(tree->root, l, r);
                        std::priority_queue<PFI> res = TopDown_nodeentries_search_zorder_and(
                            node, storage->query_points[i].data(), l, r, ef,
                            storage->query_K, edge_limit, wq, filter_ids_map);

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
                    if (recall < 0.9)
                    {
                        break;
                    }
                }

                outfile.close();
            }
        }

        // Benchmark iRangeGraph's probabilistic multi-attribute traversal.
        // Predicate flags are cached per domain (N * query_count bytes) outside
        // the timed EF sweep. Morton bounds, entry sampling, and traversal are timed.
        // CSV columns: ef, recall, qps, distance computations/query, hops/query.
        void search_irange(std::vector<int> &SearchEF, std::string saveprefix, int edge_limit)
        {
            setprob();
            const int N = storage->data_nb;
            if (N <= 0 || storage->query_nb <= 0 || storage->query_K <= 0 || edge_limit <= 0)
                throw Exception("data count, query count, query_K and edge limit must be positive");
            if (storage->multi_zorder.size() != static_cast<size_t>(N))
                throw Exception("search_irange requires the existing Z-order keys loaded by the caller");
            for (int ef : SearchEF)
                if (ef <= 0)
                    throw Exception("ef must be positive");

            if (!std::is_sorted(storage->multi_zorder.begin(), storage->multi_zorder.end(),
                                MultiAttributeZOrderLess))
                throw Exception("search_irange requires Morton-sorted base rows");
            for (const auto &range : storage->query_range_multi)
            {
                const std::string &domain = range.first;
                const auto gt_it = storage->ground_truth.find(domain);
                if (gt_it == storage->ground_truth.end() ||
                    gt_it->second.size() != static_cast<size_t>(storage->query_nb) ||
                    range.second.size() != static_cast<size_t>(storage->query_nb))
                    throw Exception("query ranges and ground truth must match the query count");
                const auto &gt = gt_it->second;
                const std::string savepath = saveprefix + domain + "-irange.csv";
                CheckPath(savepath);
                std::ofstream outfile(savepath);
                if (!outfile.is_open())
                    throw Exception("cannot open " + savepath);
                std::cout << "Directory: " << savepath << std::endl;

                // Cache every query's predicate flags once, outside all timed ef runs.
                // This cache belongs to one domain and is released after its ef sweep.
                std::vector<std::vector<char>> query_flags(storage->query_nb);
                for (int i = 0; i < storage->query_nb; ++i)
                {
                    const auto &cons = range.second[i];
                    const int num_attrs = static_cast<int>(cons.attr_constraints.size());
                    if (num_attrs <= 0 || static_cast<size_t>(num_attrs) != attr_dim_)
                        throw Exception("query attribute dimension mismatch");
                    for (const auto &bounds : cons.attr_constraints)
                        if (bounds.first > bounds.second)
                            throw Exception("query lower bound exceeds upper bound");

                    auto &flags = query_flags[i];
                    flags.resize(N);
                    for (int pid = 0; pid < N; ++pid)
                    {
                        bool qualifying = true;
                        for (int d = 0; d < num_attrs; ++d)
                        {
                            const int value = storage->attributes[pid][d];
                            if (value < cons.attr_constraints[d].first ||
                                value > cons.attr_constraints[d].second)
                            {
                                qualifying = false;
                                break;
                            }
                        }
                        flags[pid] = static_cast<char>(qualifying);
                    }
                }
                for (int ef : SearchEF)
                {
                    size_t tp = 0;
                    double searchtime = 0;
                    metric_hops = 0;
                    metric_distance_computations = 0;
                    for (int i = storage->query_nb - 1; i >= 0; --i)
                    {
                        const auto &cons = range.second[i];
                        const int num_attrs = static_cast<int>(cons.attr_constraints.size());
                        if (num_attrs <= 0 || static_cast<size_t>(num_attrs) != attr_dim_)
                            throw Exception("query attribute dimension mismatch");

                        const std::vector<char> &filter_ids_map = query_flags[i];

                        timeval t1, t2;
                        gettimeofday(&t1, NULL);
                        std::vector<int> coords_min(num_attrs), coords_max(num_attrs);
                        for (int d = 0; d < num_attrs; ++d)
                        {
                            coords_min[d] = cons.attr_constraints[d].first;
                            coords_max[d] = cons.attr_constraints[d].second;
                        }
                        const auto z_min = CalculateMultiAttributeZOrder(coords_min);
                        const auto z_max = CalculateMultiAttributeZOrder(coords_max);
                        const auto first = std::lower_bound(storage->multi_zorder.begin(),
                                                            storage->multi_zorder.end(), z_min, MultiAttributeZOrderLess);
                        const auto end = std::upper_bound(storage->multi_zorder.begin(),
                                                          storage->multi_zorder.end(), z_max, MultiAttributeZOrderLess);
                        const int l = static_cast<int>(first - storage->multi_zorder.begin());
                        const int r = static_cast<int>(end - storage->multi_zorder.begin()) - 1;
                        std::priority_queue<PFI> res;
                        if (l <= r)
                        {
                            auto nodes = tree->range_filter(tree->root, l, r);
                            res = TopDown_nodeentries_search_mutil(
                                nodes, storage->query_points[i].data(), l, r, ef,
                                storage->query_K, edge_limit, filter_ids_map);
                        }
                        gettimeofday(&t2, NULL);
                        searchtime += GetTime(t1, t2);

                        std::map<int, int> record;
                        while (!res.empty())
                        {
                            const int id = res.top().second;
                            res.pop();
                            if (record.count(id))
                                throw Exception("repetitive search results");
                            record[id] = 1;
                            if (std::find(gt[i].begin(), gt[i].end(), id) != gt[i].end())
                                ++tp;
                        }
                    }
                    const double recall = double(tp) / storage->query_nb / storage->query_K;
                    const double qps = storage->query_nb / searchtime;
                    const double dco = double(metric_distance_computations) / storage->query_nb;
                    const double hop = double(metric_hops) / storage->query_nb;
                    outfile << ef << "," << recall << "," << qps << "," << dco << "," << hop << std::endl;
                    if (!outfile)
                        throw Exception("failed writing " + savepath);
                    // Preserve the existing benchmark's recall stopping rule.
                    if (recall < 0.9)
                        break;
                }
            }
        }

        void searchfilter(std::vector<int> &SearchEF, std::string saveprefix, int edge_limit)
        {
            const int N = storage->data_nb; // 数据点总数
            std::vector<char> filter_ids_map(N);
            for (auto range : storage->query_range_multi)
            {
                std::string domain = range.first;
                std::vector<std::vector<int>> &gt = storage->ground_truth[domain];
                std::string savepath = saveprefix + domain + "-filter.csv";
                std::cout << "Directory: " << savepath << std::endl;
                CheckPath(savepath);

                std::ofstream outfile(savepath);
                if (!outfile.is_open())
                {
                    throw Exception("cannot open " + savepath);
                }

                std::vector<int> HOP;
                std::vector<int> DCO;
                std::vector<float> QPS;
                std::vector<float> RECALL;

                for (auto ef : SearchEF)
                {
                    int cc = 0;
                    int cccount = 0;
                    int tp = 0;
                    float searchtime = 0;

                    metric_hops = 0;
                    metric_distance_computations = 0;

                    for (int i = storage->query_nb - 1; i >= 0; i--)
                    {
                        auto &cons = range.second[i];
                        int num_attrs = cons.attr_constraints.size(); // 属性个数，不再硬编码为2

                        // 构建 N 维 MBR
                        MBR wq;
                        wq.dim = num_attrs;
                        wq.min_vals.resize(num_attrs);
                        wq.max_vals.resize(num_attrs);
                        for (int d = 0; d < num_attrs; d++)
                        {
                            wq.min_vals[d] = cons.attr_constraints[d].first;
                            wq.max_vals[d] = cons.attr_constraints[d].second;
                        }

                        timeval t1, t2;
                        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
                        std::default_random_engine e(seed);

                        for (int xb = 0; xb < N; xb++)
                        {
                            bool in_range = true;
                            for (int d = 0; d < num_attrs; d++)
                            {
                                int v = storage->attributes[xb][d];
                                if (v < static_cast<int>(wq.min_vals[d]) ||
                                    v > static_cast<int>(wq.max_vals[d]))
                                {
                                    in_range = false;
                                    break;
                                }
                            }
                            filter_ids_map[xb] = static_cast<char>(in_range);
                        }
                        gettimeofday(&t1, NULL);

                        // 构建 N 维 Z-order 编码的边界
                        std::vector<int> coords_min(num_attrs), coords_max(num_attrs);
                        for (int d = 0; d < num_attrs; d++)
                        {
                            coords_min[d] = wq.min_vals[d];
                            coords_max[d] = wq.max_vals[d];
                        }
                        MultiAttributeZCode z_min = CalculateMultiAttributeZOrder(coords_min);
                        MultiAttributeZCode z_max = CalculateMultiAttributeZOrder(coords_max);

                        auto it1 = std::lower_bound(
                            storage->multi_zorder.begin(), storage->multi_zorder.end(),
                            z_min, MultiAttributeZOrderLess);
                        int l = std::distance(storage->multi_zorder.begin(), it1);
                        auto it2 = std::upper_bound(
                            storage->multi_zorder.begin(), storage->multi_zorder.end(),
                            z_max, MultiAttributeZOrderLess);
                        if (it2 != storage->multi_zorder.begin())
                            --it2;
                        int r = std::distance(storage->multi_zorder.begin(), it2);

                        std::vector<TreeNode *> node = tree->range_filter(tree->root, l, r);
                        std::priority_queue<PFI> res = TopDown_nodeentries_search_zorder_and_filter(node, storage->query_points[i].data(), l, r, ef, storage->query_K, edge_limit, wq, filter_ids_map);

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
                    if (recall < 0.9)
                    {
                        break;
                    }
                }

                outfile.close();
            }
        }

    };
}
