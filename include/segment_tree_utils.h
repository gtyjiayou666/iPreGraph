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

void ensure_parent_directory(std::string filename)
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

float elapsed_seconds(timeval &begin, timeval &end)
{
    return end.tv_sec - begin.tv_sec + (end.tv_usec - begin.tv_usec) * 1.0 / 1000000.0;
}

namespace iPreGraph
{
    typedef std::pair<float, int> PFI;
    struct PFIORPRO
    {
        float dis;
        int id;
        int range_id;
        PFIORPRO(float d, int i, int range_id)
            : dis(d), id(i), range_id(range_id) {}
    };
    struct Compare
    {
        bool operator()(const PFIORPRO &a, const PFIORPRO &b)
        {
            return a.dis > b.dis; // Min-heap: the smallest distance comes first.
        }
    };

    typedef unsigned int tableint;
    typedef unsigned int linklistsizeint;

    // Vector loader used by segment-tree index construction.
    class DataLoader
    {
    public:
        int Dim, data_nb;
        std::vector<std::vector<float>> data_points;

        void load_base_vectors(std::string filename)
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

    // Read integer payloads from JSON/text files.
    // Read the entire file into a string.
    inline std::string read_text_file(const std::string &filename)
    {
        std::ifstream in(filename);
        if (!in.is_open())
            throw Exception("cannot open " + filename);
        std::stringstream buf;
        buf << in.rdbuf();
        return buf.str();
    }

    // Extract integer tokens in input order.
    inline std::vector<int> parse_integer_values(const std::string &text, const std::string &who)
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
                // Skip a standalone plus or minus sign.
                ++i;
                continue;
            }
            // Reject decimal or exponent suffixes instead of truncating non-integers.
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
        std::vector<int> label_arr;

        DataLoaderOr(const std::string &filename,
                     int attr_num = 3,
                     int label_col = 0)
        {
            if (attr_num <= 0 || label_col < 0 || label_col >= attr_num)
                throw Exception("load_label_array: need 0 <= label_col < attr_num (got attr_num=" +
                                std::to_string(attr_num) + ", label_col=" + std::to_string(label_col) + ")");

            std::vector<int> v = parse_integer_values(read_text_file(filename), filename);
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
        void load_query_vectors(std::string filename)
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

        void build_tree(TreeNode *u)
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
                build_tree(childnode);
                l = r + 1;
            }
        }

    };

    // Sparse bitmap: nonzero 64-bit words stored as (block_idx, word), sorted by block_idx.
    // Membership and intersection tests operate on sparse nonzero blocks.
    class Bitmap
    {
    public:
        std::vector<std::pair<uint32_t, uint64_t>> blocks; // sorted by first, second != 0

        Bitmap() {}
        explicit Bitmap(int nbits) {} // Sparse storage needs no capacity reservation.

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

        // Return whether the two bitmaps share any set bit.
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
        // Form the union by merging the sorted block lists.
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

    // Segment-tree node with label-presence bitmaps for its row interval.
    class TreeNodeLabel
    {
    public:
        int node_id;
        int lbound, rbound;
        int depth;
        int range_id;
        std::vector<TreeNodeLabel *> childs;
        std::vector<Bitmap> bmps;
        static int next_id; // Shared counter for node IDs.
        TreeNodeLabel(int l, int r, int d) : lbound(l), rbound(r), depth(d)
        {
            node_id = next_id++;
        }
    };

    int TreeNodeLabel::next_id = 0;
    // Binary segment tree matching SegmentTree's split rule and graph depths.
    // Each node stores the union of labels present in its row interval.
    class SegmentTreeLabel
    {
    public:
        int ways_ = 2;
        int dim_ = 1;              // Number of bitmap slots; label search uses one.
        std::vector<int> dim_max_; // Label-domain upper bounds for bitmap initialization.
        TreeNodeLabel *root{nullptr};
        int max_depth{-1};
        std::vector<TreeNodeLabel *> treenodes;

        // attribute[pid] is the selected label for sorted data row pid.
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
            initialize_node_bitmaps(root);
            build_tree(root, attribute);
        }

        void initialize_node_bitmaps(TreeNodeLabel *u)
        {
            u->bmps.clear();
            for (int d = 0; d < dim_; d++)
                u->bmps.emplace_back(dim_max_[d]);
        }

        void build_tree(TreeNodeLabel *u, const std::vector<int> &attributes)
        {
            if (u == nullptr)
                throw Exception("Tree node is a nullptr");

            treenodes.emplace_back(u);
            max_depth = std::max(max_depth, u->depth);
            int L = u->lbound, R = u->rbound;

            if (L == R)
            {
                // A leaf records its single point's label in each bitmap.
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
                initialize_node_bitmaps(childnode);
                u->childs.emplace_back(childnode);
                build_tree(childnode, attributes);
                l = r + 1;
            }

            // Internal-node bitmaps are the bitwise union of child bitmaps.
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

        // Return the smallest set of tree nodes covering row interval [ql, qr].
        std::vector<TreeNodeLabel *> find_covering_nodes(TreeNodeLabel *u, int ql, int qr)
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
                auto t = find_covering_nodes(static_cast<TreeNodeLabel *>(child), ql, qr);
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
