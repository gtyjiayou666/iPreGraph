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

// Mixed predicates combine label membership with numerical range conjunctions.
// Rows are ordered by (label, Morton code). Each query label contributes a
// conservative Morton cover; point-level flags enforce the complete predicate.
// Segment summaries are reconstructed when the index is loaded. Active numeric
// pruning reads the joint MBR; stored label bitmaps and per-label bounds remain
// part of the runtime representation.

namespace iPreGraph_mixed
{
    // Import the shared graph ID and adjacency-list types.
    using iPreGraph::linklistsizeint;
    using iPreGraph::tableint;

    // ------------------------------------------------------------------------
    // Mixed-attribute segment-tree node.
    // ------------------------------------------------------------------------
    class TreeNodeMixed
    {
    public:
        int node_id;
        int lbound, rbound;
        int depth;
        int range_id;
        std::vector<TreeNodeMixed *> childs;

        iPreGraph::Bitmap label_bmp; // Labels present in this segment.

        // Per-label numeric bounds, stored as packed int entries for labels present.
        // Each entry contains 1 + 2*z_attr_dim integers:
        // [label, min[0..dim), max[0..dim)], ordered by label.
        std::vector<int> label_mbr_data;
        std::vector<int> joint_mn; // Joint numeric MBR used for segment rejection.
        std::vector<int> joint_mx;

        TreeNodeMixed(int l, int r, int d, int z_attr_dim)
            : lbound(l), rbound(r), depth(d),
              joint_mn(z_attr_dim, std::numeric_limits<int>::max()),
              joint_mx(z_attr_dim, std::numeric_limits<int>::lowest())
        {
        }
    };

    // ------------------------------------------------------------------------
    // Binary segment tree matching the graph index partitioning.
    // Node depths identify the corresponding graph layers.
    // ------------------------------------------------------------------------
    class SegmentTreeMixed
    {
    public:
        int ways_ = 2;
        int z_attr_dim_ = 2; // Number of numeric attribute columns.
        int node_count{0};   // Node count and next node ID.
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

        // Input rows must be sorted by (label, Morton code).
        // labels: N nondecreasing label IDs.
        // z_attrs: N rows of nonnegative integer numeric attributes in the same order.
        void build_tree(TreeNodeMixed *u,
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
                // A leaf contains one object.
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
                build_tree(childnode, labels, z_attrs);
                l = r + 1;
            }

            // merge child label bitmaps and per-label bounds.
            u->label_bmp.reset();
            for (const auto *ch : u->childs)
                u->label_bmp.union_with(ch->label_bmp);
            // merge child joint MBRs.
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
            // merge the two sorted per-label bound arrays.
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

        // Return the minimal disjoint tree-node cover of the position range [ql,qr].
        std::vector<TreeNodeMixed *> find_covering_nodes(TreeNodeMixed *u, int ql, int qr)
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
                auto t = find_covering_nodes(static_cast<TreeNodeMixed *>(child), ql, qr);
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
    // Load mixed attributes, query vectors, constraints, and ground truth.
    // Base rows and attributes share (label, Morton code) ordering.
    // ------------------------------------------------------------------------
    // Parse mixed predicates and enforce (label, Morton code) row ordering.
    class MixedDataLoader
    {
    public:
        int Dim{0}, query_nb{0}, query_K{0}, data_nb{0};
        std::vector<std::vector<float>> query_points;
        std::vector<std::vector<float>> data_points;                 // Base vectors in mixed order.
        std::vector<int> labels;                                     // Nondecreasing labels in mixed order.
        std::vector<std::vector<int>> z_attrs;                       // Numeric attributes in mixed order (N by D).
        std::vector<iPreGraph_multi::MultiAttributeZCode> z_codes; // Morton codes in mixed order.
        std::vector<int> label_block_start;                          // First row of each label block; -1 if absent.
        std::vector<int> label_block_end;                            // Last row of each label block; -1 if absent.

        // Each query combines label membership (OR) with numeric intervals (AND).
        std::vector<std::vector<int>> query_labels;
        std::vector<Attr_Constraint> query_constraints;

        MixedDataLoader() {}
        ~MixedDataLoader() {}

        // ------------------------------------------------------------------
        // Parse the integer-array JSON formats used by the input files.
        // Strings, objects, and fractional numbers are not supported.
        // ------------------------------------------------------------------
        // Extract integers from a flat array such as [v0, v1, ..., vn-1].
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

        // Parse nested integer arrays.
        // Arrays may contain integers or further arrays at any depth.
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
        // Load attr_nb integer columns per row; label_col selects the label.
        // All other columns are numeric attributes. Rows must be in mixed order.
        // Compute Morton codes and label-block boundaries; the index validates order.
        // Infer n_points from the integer count when n_points <= 0.
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
            build_label_blocks();
        }

        // ------------------------------------------------------------------
        // Query format: [[label_list, [lo1,hi1], [lo2,hi2], ...], ...].
        // Labels are alternatives; all numeric intervals must hold.
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
            query_nb = (int)root.childs.size();
            query_labels.resize(query_nb);
            query_constraints.resize(query_nb);
            for (int q = 0; q < query_nb; q++)
            {
                const JsonArr &query = root.childs[q]; // One query: [label_list, [lo,hi], ...].
                if (query.childs.empty())
                    continue;
                // The first array contains requested labels.
                query_labels[q] = query.childs[0].ints;
                // Remaining arrays contain inclusive numeric bounds [lo,hi].
                for (size_t k = 1; k < query.childs.size(); k++)
                {
                    const auto &r = query.childs[k].ints;
                    if (r.size() < 2)
                        continue;
                    query_constraints[q].attr_constraints.emplace_back(r[0], r[1]);
                }
            }
        }

        // Read integer payloads from JSON/text files.
        // Read the complete file into a string.
        static std::string read_text_file(const std::string &filename)
        {
            std::ifstream in(filename);
            if (!in.is_open())
                throw Exception("cannot open " + filename);
            std::stringstream buf;
            buf << in.rdbuf();
            return buf.str();
        }

        // Parse nested integer lists while preserving query groups.
        static std::vector<std::vector<int>> read_nested_integer_arrays_json(const std::string &filename)
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
                    ++i; // Allow an empty inner list.
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

        // Extract integers from flat ground-truth input.
        static std::vector<int> parse_integer_values(const std::string &text, const std::string &who)
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
                    // Skip a standalone sign.
                    ++i;
                    continue;
                }
                // Reject fractional or scientific notation.
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
        // Load ground-truth IDs, grouped by query or flattened.
        // IDs refer to base rows in mixed order.
        // ------------------------------------------------------------------
        static std::vector<std::vector<int>> load_ground_truth_json(const std::string &filename, int query_nb, int query_K)
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
        // ------------------------------------------------------------------
        // Each label occupies one contiguous block in mixed order.
        // Cache block boundaries for constant-time lookup during search.
        // Return immediately if boundaries are already populated.
        // ------------------------------------------------------------------
        void build_label_blocks()
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
        // Validate lexicographic (label, Morton code) ordering:
        // labels must be nondecreasing, as must Morton codes within each label.
        // Reject invalid ordering before computing query covers.
        // ------------------------------------------------------------------
        void validate_label_morton_order() const
        {
            const int n = (int)labels.size();
            if (n == 0)
                return;
            if ((int)z_codes.size() != n || (int)z_attrs.size() != n)
                throw Exception("validate_label_morton_order: labels/z_attrs/z_codes size mismatch");
            for (int i = 1; i < n; i++)
            {
                if (labels[i] < labels[i - 1])
                    throw Exception("validate_label_morton_order: labels not non-decreasing "
                                    "(data must be sorted by label first)");
                if (labels[i] == labels[i - 1] &&
                    iPreGraph_multi::morton_key_less(z_codes[i], z_codes[i - 1]))
                    throw Exception("validate_label_morton_order: z-codes not non-decreasing within a label "
                                    "(data must be z-order sorted within each label)");
            }
        }

        void load_query_vectors(std::string filename)
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

    };


    // ============================================================================
    // Mixed-attribute ANN search.
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

        std::vector<bool> flags; // Complete point-level predicate: label membership and numeric bounds.

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
            tree->build_tree(tree->root, storage->labels, storage->z_attrs);
            // Validate the mixed row ordering required by search.
            storage->validate_label_morton_order();

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
                    linklistsizeint *data = get_adjacency_list(pid, layer);
                    edgefile.read((char *)data, sizeof(tableint));
                    int size = get_neighbor_count(data);
                    if (size > (int)M_out)
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


        // Reject segments whose joint numeric MBR misses the query box.
        inline bool may_match_numeric_bounds(const TreeNodeMixed *u,
                                     const iPreGraph_multi::MBR &wq) const
        {
            const size_t dim = u->joint_mn.size();
            for (size_t d = 0; d < dim; d++)
            {
                if (u->joint_mn[d] > (int)wq.max_vals[d] ||
                    (int)wq.min_vals[d] > u->joint_mx[d])
                {
                    return false;
                }
            }
            return true;
        }

        // Point flags combine label membership and all numeric constraints.
        inline bool point_matches_predicate(int pid,
                              const std::vector<bool> &filter_map) const
        {
            return filter_map[pid];
        }

        // ------------------------------------------------------------------------
        // Unified search over per-label Morton covers. Direct and DFS edge
        // selection retain endpoints that pass the complete predicate.
        // ------------------------------------------------------------------------

        // Min-heap comparator for iPreGraph_multi::PFIORPRO.
        // The shared iPreGraph::Compare accepts a different PFIORPRO type.
        struct ComparePFIORPRO
        {
            bool operator()(const iPreGraph_multi::PFIORPRO &a,
                            const iPreGraph_multi::PFIORPRO &b) const
            {
                return a.dis > b.dis;
            }
        };

        // Use the position cover for layer skipping and numeric MBRs for pruning.
        // Retain unvisited endpoints inside this cover that satisfy the complete predicate.
        // The caller supplies one label cover for each expansion.
        std::vector<tableint> select_direct_edges(int pid, int ql, int qr, int edge_limit,
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
                // Skip layers whose child preserves the cover intersection.
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

                // No numeric match exists in this segment or its descendants.
                if (!may_match_numeric_bounds(cur_node, wq))
                    return selected_edges;

                // Skip this layer when the sibling MBR misses the query box.
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
                                if (!may_match_numeric_bounds(cur_node->childs[i], wq))
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
                    if (neighborId < ql || neighborId > qr)
                        continue;
                    if (visited_set.get(neighborId))
                        continue;
                    if (!point_matches_predicate(neighborId, filter_map))
                        continue;
                    selected_edges.emplace_back(neighborId);
                    if (selected_edges.size() == (size_t)edge_limit)
                        return selected_edges;
                }
            } while (nxt_node);
            return selected_edges;
        }

        std::vector<tableint> select_intra_range_edges(int pid, int ql, int qr, int edge_limit, searcher::Bitset<uint64_t> &visited_set)
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

        std::priority_queue<PFI> search_post_filter(std::vector<TreeNodeMixed *> &filterednodes, const void *query_data, int l, int r, int ef, int query_k, int edge_limit)
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
        // DFS explores nonqualifying intermediates across the supplied covers.
        // A branch reduced to one intersecting cover uses numeric MBR checks.
        void select_cross_range_edges(std::vector<int> &QL, std::vector<int> &QR, int pid,
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
                // Skip layers whose child preserves the current covering-range intersection.
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
                        if (interval_overlap_span(cur_node->lbound, cur_node->rbound, left, right) ==
                            interval_overlap_span(nxt_node->lbound, nxt_node->rbound, left, right))
                        {
                            cur_node = nxt_node;
                            contain = true;
                        }
                    }
                } while (contain);

                if (single_mode)
                {
                    // Inspect this layer when the sibling's numeric MBR may
                    // contain qualifying points, then continue from this node.
                    if (!may_match_numeric_bounds(cur_node, wq))
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
                            else if (may_match_numeric_bounds(cur_node->childs[i], wq))
                            {
                                inters = true;
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
                            if (neighborId < left || neighborId > right)
                                continue;
                            if (point_matches_predicate(neighborId, filter_map))
                            {
                                edges.emplace_back(neighborId);
                            }
                            else if (deep > 1)
                            {
                                // Recurse with the original cover arrays and one fewer hop.
                                select_cross_range_edges(QL, QR, neighborId, edge_limit, deep - 1,
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
                    int *data = (int *)get_adjacency_list(pid, cur_node->depth);
                    size_t size = get_neighbor_count((linklistsizeint *)data);
                    for (size_t j = 1; j <= size; ++j)
                    {
                        int neighborId = *(data + j);
                        if (neighborId < left || neighborId > right)
                            continue;
                        if (point_matches_predicate(neighborId, filter_map))
                        {
                            edges.emplace_back(neighborId);
                        }
                        else if (deep > 1)
                        {
                            select_cross_range_edges(QL, QR, neighborId, edge_limit, deep - 1,
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
                        // Continue this branch with numeric pruning after its
                        // intersecting covers shrink to one label block.
                        single_mode = true;
                    }
                }
            }
        }

        // Unified search over the supplied navigation covers.
        // Q_L/Q_R contain one conservative cover per requested label.
        // Sample entries from the tree-node cover of the largest range.
        // Queue entries carry a range ID for direct edge selection.
        // Only qualifying objects enter top_candidates.
        // select_cross_range_edges supplies additional cross-range endpoints.
        // id identifies the largest range used to obtain filterednodes.
        std::priority_queue<PFI> search_pdg(
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
                char *ep_data = get_vector_data(pid);
                float dis = fstdistfunc_(query_data, ep_data, dist_func_param_);
                ++metric_distance_computations;
                if (top_candidates.size() < ef)
                {
                    candidate_set.emplace(dis, pid, id);
                    if (point_matches_predicate(pid, filter_map))
                    {
                        top_candidates.emplace(dis, pid);
                        lowerBound = top_candidates.top().first;
                    }
                }
                else if (dis < lowerBound)
                {
                    candidate_set.emplace(dis, pid, id);
                    if (point_matches_predicate(pid, filter_map))
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
                    // Locate the range containing current_pid.
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
                auto selected_edges = select_direct_edges(current_pid, Q_L[range_id],
                                                        Q_R[range_id], edge_limit,
                                                        visited_set, label_bmp, wq,
                                                        node_cache, filter_map);
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
                        candidate_set.emplace(dis, neighbor_id, range_id);
                        top_candidates.emplace(dis, neighbor_id);
                        lowerBound = top_candidates.top().first;
                    }
                    else if (dis < lowerBound)
                    {
                        candidate_set.emplace(dis, neighbor_id, range_id);
                        top_candidates.emplace(dis, neighbor_id);
                        top_candidates.pop();
                        lowerBound = top_candidates.top().first;
                    }
                }

                std::vector<tableint> edges;
                edges.reserve(edge_limit / 2);
                // Use all per-label covers for cross-range DFS.
                select_cross_range_edges(Q_L, Q_R, current_pid, edge_limit / 2, 2, edges,
                           label_bmp, wq, node_cache, filter_map, single_mode);
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
        // Benchmark label membership AND numeric interval predicates.
        // gt stores mixed-order IDs per query; an empty gt reports recall = -1.
        // CSV columns: ef, recall, qps, dco, hop.
        // ------------------------------------------------------------------------
        void benchmark_pdg(std::vector<int> &SearchEF, std::string saveprefix, int edge_limit,
                          const std::vector<std::vector<int>> &gt = {})
        {
            const int N = storage->data_nb;

            std::string savepath = saveprefix + "_mixed.csv";
            std::cout << "Savepath: " << savepath << std::endl;
            ensure_parent_directory(savepath);
            std::ofstream outfile(savepath);
            if (!outfile.is_open())
                throw Exception("cannot open " + savepath);

            if (flags.size() < (size_t)N)
                flags.assign(N, false);
            storage->build_label_blocks(); // Ensure label-block boundaries are available.

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
                    // Sort and deduplicate requested labels.
                    std::vector<int> labels = storage->query_labels[i];
                    std::sort(labels.begin(), labels.end());
                    labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
                    if (labels.empty())
                        continue;

                    // Build the query-label bitmap.
                    iPreGraph::Bitmap label_bitmap;
                    for (int label : labels)
                    {
                        if (label < 0)
                            continue;
                        label_bitmap.set(label);
                    }
                    if (label_bitmap.blocks.empty())
                        continue;

                    // Compute the numeric query box and Morton codes of its corners.
                    // The corner codes bound a conservative cover, not the exact qualifying set.
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
                        iPreGraph_multi::encode_morton_key(coords_min);
                    iPreGraph_multi::MultiAttributeZCode q_z_max =
                        iPreGraph_multi::encode_morton_key(coords_max);

                    // Compute complete point flags within the requested label blocks.
                    // Each point must satisfy every numeric interval.
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

                    // Generate one conservative Morton cover per requested label.
                    // Rows are sorted by (label, Morton code), and query labels are sorted.
                    // Within each label block, binary-search the corner-code bounds:
                    // the first code >= q_z_min and the last code <= q_z_max.
                    // Every numeric match lies within these bounds.
                    // Nonmatching points inside a cover are rejected by point flags.
                    // Unrequested label blocks are excluded from the covers.
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
                                                        iPreGraph_multi::morton_key_less) -
                                       storage->z_codes.begin());
                        int rr = (int)(std::upper_bound(storage->z_codes.begin() + bl,
                                                        storage->z_codes.begin() + br + 1, q_z_max,
                                                        iPreGraph_multi::morton_key_less) -
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
                        searchtime += elapsed_seconds(t1, t2);
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
                    std::vector<TreeNodeMixed *> node = tree->find_covering_nodes(tree->root, Q_L[id], Q_R[id]);
                    std::vector<uint8_t> node_cache((size_t)tree->node_count, 0);

                    std::priority_queue<PFI> res = search_pdg(
                        node, storage->query_points[i].data(), Q_L, Q_R, ef,
                        storage->query_K, edge_limit, id,
                        label_bitmap, wq,
                        node_cache, flags);
                    gettimeofday(&t2, NULL);
                    searchtime += elapsed_seconds(t1, t2);

                    std::map<int, int> record;
                    while (!res.empty())
                    {
                        auto x = res.top().second;
                        res.pop();
                        if (record.count(x))
                            continue;
                        record[x] = 1;
                        total_cnt++;
                        if (point_matches_predicate(x, flags))
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
        void benchmark_post_filter(std::vector<int> &SearchEF, std::string saveprefix, int edge_limit,
                           const std::vector<std::vector<int>> &gt = {})
        {
            const int N = storage->data_nb;

            std::string savepath = saveprefix + "_filter.csv";
            std::cout << "Savepath: " << savepath << std::endl;
            ensure_parent_directory(savepath);
            std::ofstream outfile(savepath);
            if (!outfile.is_open())
                throw Exception("cannot open " + savepath);

            if (flags.size() < (size_t)N)
                flags.assign(N, false);
            storage->build_label_blocks(); // Ensure label-block boundaries are available.

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
                    // Sort and deduplicate requested labels.
                    std::vector<int> labels = storage->query_labels[i];
                    std::sort(labels.begin(), labels.end());
                    labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
                    if (labels.empty())
                        continue;

                    // Build the query-label bitmap.
                    iPreGraph::Bitmap label_bitmap;
                    for (int label : labels)
                    {
                        if (label < 0)
                            continue;
                        label_bitmap.set(label);
                    }
                    if (label_bitmap.blocks.empty())
                        continue;

                    // Compute the numeric query box and Morton codes of its corners.
                    // The corner codes narrow the first and last requested label blocks.
                    // Point flags enforce the complete predicate within the resulting cover.
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
                        iPreGraph_multi::encode_morton_key(coords_min);
                    iPreGraph_multi::MultiAttributeZCode q_z_max =
                        iPreGraph_multi::encode_morton_key(coords_max);

                    // Compute complete point flags within the requested label blocks.
                    // Each point must satisfy every numeric interval.
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
                                                   iPreGraph_multi::morton_key_less) -
                                  storage->z_codes.begin());
                    bl = storage->label_block_start[labels.back()];
                    br = storage->label_block_end[labels.back()];

                    int r = (int)(std::upper_bound(storage->z_codes.begin() + bl,
                                                   storage->z_codes.begin() + br + 1, q_z_max,
                                                   iPreGraph_multi::morton_key_less) -
                                  storage->z_codes.begin()) -
                            1;

                    std::vector<TreeNodeMixed *> node = tree->find_covering_nodes(tree->root, l, r);

                    timeval t1, t2;
                    gettimeofday(&t1, NULL);
                    std::priority_queue<PFI> res = search_post_filter(
                        node, storage->query_points[i].data(), l, r, ef,
                        storage->query_K, edge_limit);
                    gettimeofday(&t2, NULL);
                    searchtime += elapsed_seconds(t1, t2);

                    std::map<int, int> record;
                    while (!res.empty())
                    {
                        auto x = res.top().second;
                        res.pop();
                        if (record.count(x))
                            continue;
                        record[x] = 1;
                        total_cnt++;
                        if (point_matches_predicate(x, flags))
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
