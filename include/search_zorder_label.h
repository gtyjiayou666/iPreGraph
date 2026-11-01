#pragma once

#include <vector>
#include "multi_attribute_utils.h"
#include "search_containers.h"
#include "memory_utils.h"
#include <bitset>
#include <stack>
#include <cmath>
#include <climits>

// ============================================================================
// Filtered search over rows sorted lexicographically by (Morton code, label).
// Unlike the (label, Morton code) order in search_mixed.h, equal labels need
// not occupy contiguous row intervals. Corner Morton codes provide a
// conservative cover of the numerical query box, located by binary search.
// Node MBRs support pruning; point flags enforce both labels and numerical bounds.
// Query labels are combined with OR; numerical intervals are combined with AND.
// Attribute rows contain attr_nb values, with the label in label_col.
// validate_morton_label_order checks the required row ordering after loading.
// ============================================================================

namespace iPreGraph_multi
{
    // Import index ID and adjacency-count types from the shared utilities.
    using iPreGraph::linklistsizeint;
    using iPreGraph::tableint;

    // ------------------------------------------------------------------------
    // Tree node for the (Morton code, label) order, with a numerical MBR.
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
    // Binary segment tree with the same partitioning as SegmentTreeZorder.
    // Leaf MBRs contain one point; internal MBRs merge their children's bounds.
    // ------------------------------------------------------------------------
    class SegmentTreeZorderLabel
    {
    public:
        int ways_ = 2;
        int dim_ = 2; // Number of numerical attributes.
        TreeNodeZLabel *root{nullptr};
        int max_depth{-1};
        int node_count{0}; // Node count and next node ID.
        std::vector<TreeNodeZLabel *> treenodes;

        SegmentTreeZorderLabel(int data_nb, int dim) : dim_(dim)
        {
            root = new TreeNodeZLabel(0, data_nb - 1, 0, dim);
            root->node_id = node_count++;
        }

        // Inputs must already follow (Morton code, label) order.
        // attrs contains N rows of numerical attributes; labels uses the same order.
        // Only numerical attributes contribute to the node summaries.
        void build_tree(TreeNodeZLabel *u,
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
                // A leaf MBR contains exactly one point.
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
                build_tree(childnode, attrs, labels);
                l = r + 1;
            }
            if (L != R)
            {
                for (const auto *child : u->childs)
                    u->mbr.merge(child->mbr);
            }
        }

        // Return the canonical tree nodes covering the row interval [ql, qr].
        std::vector<TreeNodeZLabel *> find_covering_nodes(TreeNodeZLabel *u, int ql, int qr)
        {
            std::vector<TreeNodeZLabel *> res;
            if (u == nullptr)
                return res;
            if (u->lbound > qr || u->rbound < ql)
                return res; // Disjoint intervals.
            if (u->lbound >= ql && u->rbound <= qr)
            {
                res.push_back(u);
                return res;
            } // The query interval contains the entire node.

            for (auto child : u->childs)
            {
                auto t = find_covering_nodes(child, ql, qr);
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
    // Load labels, numerical attributes, and mixed predicates in Morton order.
    // Base vectors, labels, attributes, and Morton codes must share the same rows.
    // The base DataLoader provides vector and ground-truth loading.
    // ------------------------------------------------------------------------
    // Parse mixed predicates and enforce (Morton code, label) row ordering.
    class ZLabelDataLoader : public iPreGraph_multi::DataLoader
    {
    public:
        std::vector<int> labels;                                     // Labels in (Morton code, label) order.
        std::vector<std::vector<int>> z_attrs;                       // Numerical attributes, excluding the label column.
        std::vector<iPreGraph_multi::MultiAttributeZCode> z_codes; // Nondecreasing Morton codes.

        // Each query combines label alternatives with numerical range conjunctions.
        std::vector<std::vector<int>> query_labels;
        std::vector<Attr_Constraint> query_constraints;

        ZLabelDataLoader() {}
        ~ZLabelDataLoader() {}

        // ------------------------------------------------------------------
        // Minimal parser for the integer-array JSON formats used by the drivers.
        // ------------------------------------------------------------------
        static std::vector<int> parse_flat_json_integers(const std::string &s)
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
                        throw Exception("parse_flat_json_integers: int overflow");
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
            std::vector<int> ints;       // Integers directly contained in this array.
            std::vector<JsonArr> childs; // Nested arrays.
        };

        static long long parse_json_integer(const std::string &s, size_t &i, size_t n)
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

        // Precondition: s[i] == '['.
        static JsonArr parse_json_array(const std::string &s, size_t &i, size_t n)
        {
            JsonArr out;
            i++; // Skip '['.
            while (i < n && s[i] != ']')
            {
                if (s[i] == '[')
                {
                    out.childs.push_back(parse_json_array(s, i, n));
                }
                else if (s[i] == '-' || (s[i] >= '0' && s[i] <= '9'))
                {
                    long long v = parse_json_integer(s, i, n);
                    if (v > 2147483647LL || v < -2147483648LL)
                        throw Exception("parse_json_array: int out of range");
                    out.ints.push_back((int)v);
                }
                else
                {
                    i++; // Skip whitespace, commas, and other separators.
                }
            }
            if (i < n)
                i++; // Skip ']'.
            return out;
        }

        static JsonArr parse_json(const std::string &s)
        {
            size_t i = 0;
            if (i >= s.size() || s[i] != '[')
                throw Exception("parse_json: expected '[' at start");
            return parse_json_array(s, i, s.size());
        }

        // ------------------------------------------------------------------
        // Read row-major attributes with attr_nb columns and the label in label_col.
        // The remaining columns are numerical; rows must follow (Morton code, label) order.
        // Compute Morton codes and validate the ordering after loading.
        // If n_points <= 0, infer it from the integer count, which must be divisible by attr_nb.
        // ------------------------------------------------------------------
        void load_attributes_json(const std::string &filename, int n_points, int attr_nb, int label_col)
        {
            std::ifstream in(filename, std::ios::in | std::ios::binary);
            if (!in.is_open())
                throw Exception("cannot open " + filename);
            std::string content((std::istreambuf_iterator<char>(in)),
                                std::istreambuf_iterator<char>());
            in.close();

            std::vector<int> vals = parse_flat_json_integers(content);
            if (attr_nb <= 0)
                throw Exception("load_attributes_json: attr_nb must be > 0");
            if (vals.size() % attr_nb != 0)
                throw Exception("load_attributes_json: int count " + std::to_string(vals.size()) +
                                " not divisible by attr_nb " + std::to_string(attr_nb));
            if (n_points <= 0)
                n_points = (int)(vals.size() / attr_nb);
            else if ((int)vals.size() != n_points * attr_nb)
                throw Exception("load_attributes_json: expected " + std::to_string(n_points * attr_nb) +
                                " ints, got " + std::to_string(vals.size()));
            if (data_nb > 0 && data_nb != n_points)
                throw Exception("load_attributes_json: data_nb (" + std::to_string(data_nb) +
                                ") != attribute points (" + std::to_string(n_points) + ")");
            if (label_col < 0 || label_col >= attr_nb)
                throw Exception("load_attributes_json: label_col out of range");

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
                        throw Exception("load_attributes_json: numeric attrs must be non-negative for z-order");
                    z_attrs[p][k++] = v;
                }
                z_codes[p] = iPreGraph_multi::encode_morton_key(z_attrs[p]);
            }
            validate_morton_label_order();
        }

        // ------------------------------------------------------------------
        // Query format: [[labels, [lo1,hi1], [lo2,hi2], ...], ...].
        // Combine labels with OR and numerical intervals with AND.
        // ------------------------------------------------------------------
        void load_hybrid_predicates_json(const std::string &filename)
        {
            std::ifstream in(filename, std::ios::in | std::ios::binary);
            if (!in.is_open())
                throw Exception("cannot open " + filename);
            std::string content((std::istreambuf_iterator<char>(in)),
                                std::istreambuf_iterator<char>());
            in.close();

            JsonArr root = parse_json(content);
            int q_nb = (int)root.childs.size();
            if (query_nb > 0 && query_nb != q_nb)
                throw Exception("load_hybrid_predicates_json: query_nb (" + std::to_string(query_nb) +
                                ") != queries in file (" + std::to_string(q_nb) + ")");
            query_nb = q_nb;
            query_labels.resize(query_nb);
            query_constraints.resize(query_nb);
            for (int q = 0; q < query_nb; q++)
            {
                const JsonArr &query = root.childs[q]; // One query: [labels, [lo,hi], ...].
                if (query.childs.empty())
                    continue;
                // The first element contains query labels.
                query_labels[q] = query.childs[0].ints;
                // Remaining elements contain numerical intervals [lo, hi].
                for (size_t k = 1; k < query.childs.size(); k++)
                {
                    const auto &r = query.childs[k].ints;
                    if (r.size() < 2)
                        continue;
                    query_constraints[q].attr_constraints.emplace_back(r[0], r[1]);
                }
            }
        }

        // ------------------------------------------------------------------
        // Check nondecreasing lexicographic order by (Morton code, label).
        // ------------------------------------------------------------------
        void validate_morton_label_order() const
        {
            if (z_codes.size() != labels.size() || z_codes.size() != z_attrs.size())
                throw Exception("validate_morton_label_order: size mismatch among labels/z_attrs/z_codes");
            for (size_t i = 1; i < z_codes.size(); i++)
            {
                if (iPreGraph_multi::morton_key_less(z_codes[i], z_codes[i - 1]))
                    throw Exception("validate_morton_label_order: z_codes not non-decreasing "
                                    "(data must be sorted by (z_code, label)) at " +
                                    std::to_string(i));
                bool same_z = !iPreGraph_multi::morton_key_less(z_codes[i - 1], z_codes[i]) &&
                              !iPreGraph_multi::morton_key_less(z_codes[i], z_codes[i - 1]);
                if (same_z && labels[i] < labels[i - 1])
                    throw Exception("validate_morton_label_order: labels not non-decreasing "
                                    "within equal z_code at " +
                                    std::to_string(i));
            }
        }

    };

    // ------------------------------------------------------------------------
    // Search the segment graph in (Morton code, label) order using numerical MBR
    // pruning and point flags for the complete mixed predicate.
    // ------------------------------------------------------------------------
    template <typename dist_t>
    class iPreGraph_Search_Zorder_Label
    {
    public:
        ZLabelDataLoader *storage;
        SegmentTreeZorderLabel *tree;
        size_t max_elements_{0};
        size_t dim_{0};
        size_t attr_dim_{0}; // Numerical attribute count for Morton codes, MBRs, and filtering.
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
            tree->build_tree(tree->root, storage->z_attrs, storage->labels);

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
        }

        ~iPreGraph_Search_Zorder_Label()
        {
            free(data_memory_);
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

        // ------------------------------------------------------------------------
        // Cache the numerical MBR intersection test by node ID: 1 keeps, 2 prunes.
        // The test depends only on the query MBR, so the cache can be reused
        // across candidate intervals within the same query.
        // ------------------------------------------------------------------------
        inline bool may_match_numeric_bounds(const TreeNodeZLabel *u,
                                    const MBR &wp,
                                    std::vector<uint8_t> &node_cache) const
        {
            int nid = u->node_id;
            uint8_t c = node_cache[nid];
            if (c != 0)
                return c == 1; // 1: may contain a match; 2: can be pruned.

            bool res = u->mbr.intersects(wp);
            node_cache[nid] = res ? 1 : 2;
            return res;
        }

        // ------------------------------------------------------------------------
        // Select edges using the same traversal as select_direct_edges, with cached MBR tests.
        // Descend toward pid within [ql,qr], skipping levels whose sibling subtrees
        // are pruned. Keep neighbors only when their complete predicate flag is set.
        // ------------------------------------------------------------------------
        std::vector<tableint> select_direct_edges(int pid, int ql, int qr, int edge_limit,
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
                        if (interval_overlap_span(cur_node->lbound, cur_node->rbound, ql, qr) ==
                            interval_overlap_span(nxt_node->lbound, nxt_node->rbound, ql, qr))
                        {
                            cur_node = nxt_node;
                            contain = true;
                        }
                    }
                } while (contain);
                if (!may_match_numeric_bounds(cur_node, wp, node_cache))
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
                                if (!may_match_numeric_bounds(cur_node->childs[i], wp, node_cache))
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
                int *data = (int *)get_adjacency_list(pid, cur_node->depth);
                size_t size = get_neighbor_count((linklistsizeint *)data);
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
        // Bounded DFS with cached MBR pruning, used with a two-hop limit.
        // Collect only points satisfying both labels and numerical bounds.
        // ------------------------------------------------------------------------
        void select_cross_range_edges(int pid, int ql, int qr, int edge_limit, int deep,
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
                    if (interval_overlap_span(cur_node->lbound, cur_node->rbound, ql, qr) ==
                        interval_overlap_span(nxt_node->lbound, nxt_node->rbound, ql, qr))
                    {
                        cur_node = nxt_node;
                        contain = true;
                    }
                }
            } while (contain);

            while (cur_node)
            {
                if (!may_match_numeric_bounds(cur_node, wp, node_cache))
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
                            if (may_match_numeric_bounds(cur_node->childs[i], wp, node_cache))
                            {
                                inters = true;
                            }
                        }
                    }
                }

                if (inters)
                {
                    int *data = (int *)get_adjacency_list(pid, cur_node->depth);
                    size_t size = get_neighbor_count((linklistsizeint *)data);
                    for (size_t j = 1; j <= size; ++j)
                    {
                        int neighborId = *(data + j);
                        if (filter_ids_map[neighborId] != 0)
                        {
                            edges.emplace_back(neighborId);
                        }
                        else if (deep > 1)
                        {
                            select_cross_range_edges(neighborId, ql, qr, edge_limit, deep - 1, edges, wp, node_cache, filter_ids_map);
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
        // Top-down search over one conservative Morton interval cover.
        // select_direct_edges and select_cross_range_edges use MBR pruning and complete predicate flags.
        // ------------------------------------------------------------------------
        std::priority_queue<PFI> search_pdg(
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
                char *ep_data = get_vector_data(pid);
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
                auto selected_edges = select_direct_edges(current_pid, l, r, edge_limit, visited_set,
                                                       wp, node_cache, filter_ids_map);
                int num_edges = (int)selected_edges.size();
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
                select_cross_range_edges(current_pid, l, r, edge_limit / 2, 2, edges, wp, node_cache, filter_ids_map);
                int num_edges1 = (int)edges.size();
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
        // Benchmark ANN search with label alternatives and numerical range conjunctions.
        // Ground-truth IDs use (Morton code, label) row order; absent ground truth gives recall -1.
        // CSV columns: ef,recall,qps,dco,hop.
        // ------------------------------------------------------------------------
        void benchmark_pdg(std::vector<int> &SearchEF, std::string saveprefix, int edge_limit)
        {
            const int N = storage->data_nb;

            std::string savepath = saveprefix + "_zorder_label_memory.csv";
            std::cout << "Savepath: " << savepath << std::endl;
            ensure_parent_directory(savepath);
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
                    // 1. Sort and deduplicate query labels.
                    std::vector<int> labels = storage->query_labels[i];
                    std::sort(labels.begin(), labels.end());
                    labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
                    if (labels.empty())
                        continue;

                    // 2. Build the query-label bitmap.
                    iPreGraph::Bitmap q_labels;
                    for (int lb : labels)
                    {
                        if (lb < 0)
                            continue;
                        q_labels.set(lb);
                    }
                    if (q_labels.blocks.empty())
                        continue;

                    // 3. Build the query MBR for node pruning.
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

                    // 4. Prepare point flags for both labels and numerical bounds.
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
                    // 5. Locate a conservative row interval using the query box's
                    // corner Morton codes and binary search over sorted data codes.
                    int l = 0, r = N - 1;
                    std::vector<int> coords_min(num_attrs), coords_max(num_attrs);
                    for (int d = 0; d < num_attrs; d++)
                    {
                        coords_min[d] = cons.attr_constraints[d].first;
                        coords_max[d] = cons.attr_constraints[d].second;
                    }
                    MultiAttributeZCode z_min = encode_morton_key(coords_min);
                    MultiAttributeZCode z_max = encode_morton_key(coords_max);

                    auto it1 = std::lower_bound(
                        storage->z_codes.begin(), storage->z_codes.end(),
                        z_min, morton_key_less);
                    l = (int)std::distance(storage->z_codes.begin(), it1);
                    auto it2 = std::upper_bound(
                        storage->z_codes.begin(), storage->z_codes.end(),
                        z_max, morton_key_less);
                    if (it2 != storage->z_codes.begin())
                        --it2;
                    r = (int)std::distance(storage->z_codes.begin(), it2);
                    if (l > r)
                        continue;

                    std::vector<TreeNodeZLabel *> node = tree->find_covering_nodes(tree->root, l, r);
                    std::vector<uint8_t> node_cache((size_t)tree->node_count, 0);
                    std::priority_queue<PFI> res = search_pdg(
                        node, storage->query_points[i].data(), l, r, ef,
                        storage->query_K, edge_limit, wq, node_cache, filter_ids_map);
                    gettimeofday(&t2, NULL);
                    searchtime += elapsed_seconds(t1, t2);

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
