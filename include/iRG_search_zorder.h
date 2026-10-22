#pragma once

#include <vector>
// #include "utils.h"
#include "utils_multi.h"
#include "searcher.hpp"
#include "memory.hpp"
#include <bitset>
#include <stack>
#include <cmath>
#include <climits>

namespace iPreGraph_multi
{
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