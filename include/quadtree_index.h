#pragma once

// A point-balanced two-attribute quadtree with one approximate RNG (L2) per node.
// Independent of the legacy Linux-only loaders and segment-tree index format.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <sys/time.h>
#include <unordered_set>
#include <vector>
#include "quadtree_graph_builder.h"
#include "hnswlib.h"
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <xmmintrin.h>
#endif

namespace iPreGraph
{

    struct QuadPoint
    {
        double x, y;
    };
    struct QuadRect
    {
        double xmin, xmax, ymin, ymax;
        bool valid() const
        {
            return std::isfinite(xmin) && std::isfinite(xmax) &&
                   std::isfinite(ymin) && std::isfinite(ymax) && xmin <= xmax && ymin <= ymax;
        }
        bool contains(const QuadPoint &p) const
        {
            return xmin <= p.x && p.x <= xmax && ymin <= p.y && p.y <= ymax;
        }
        bool contains(const QuadRect &r) const
        {
            return xmin <= r.xmin && r.xmax <= xmax && ymin <= r.ymin && r.ymax <= ymax;
        }
        bool intersects(const QuadRect &r) const
        {
            return xmin <= r.xmax && r.xmin <= xmax && ymin <= r.ymax && r.ymin <= ymax;
        }
    };

    float elapsed_seconds(timeval &begin, timeval &end)
    {
        return end.tv_sec - begin.tv_sec + (end.tv_usec - begin.tv_usec) * 1.0 / 1000000.0;
    }
    struct QuadConfig
    {
        uint32_t M = 16;
        uint32_t ef_construction = 100;
        uint32_t leaf_size = 32;
        uint32_t max_depth = 32;
        uint32_t threads = 1; // construction workers; not part of the on-disk topology
    };

    struct QuadSearchOptions
    {
        size_t entry_points = 5;
        size_t edge_limit = 0; // 0 means the index's M, shared across all levels per expansion
        uint32_t seed = 42;    // reproducible random entries, independent of ef
    };

    class QuadTreeIndex
    {
    public:
        using Id = uint32_t;
        using dist_t = float;                           // hnswlib::L2Space and the original iPreGraph search
        using EdgeLayer = std::vector<std::vector<Id>>; // indexed by original point ID
        using FilterFlags = std::vector<uint8_t>;       // flags[original ID]: 1 satisfies the rectangle
        using Result = std::pair<double, Id>;           // squared L2 distance, original input row ID
        struct Node
        {
            QuadRect bounds; // tight member MBR for new trees; legacy files may use spatial cells
            uint32_t depth;
            std::array<int32_t, 4> children{{-1, -1, -1, -1}};
            // Sorted original IDs. Edges live in edges_[depth][id], not in this node.
            std::vector<Id> ids;
            bool leaf() const
            {
                return std::all_of(children.begin(), children.end(), [](int32_t c)
                                   { return c < 0; });
            }
        };

        // Preserve input row IDs while partitioning attributes by point count.
        // Child graphs are built first, then merged into parent graph layers.
        void build(std::vector<std::vector<float>> vectors, std::vector<QuadPoint> attributes,
                   QuadConfig config = {}, std::ostream *progress = nullptr)
        {
            if (progress)
                *progress << "[quadtree] validating input\n"
                          << std::flush;
            validate_config(config);
            if (vectors.empty() || vectors.size() != attributes.size() ||
                vectors.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
                vectors[0].empty() || vectors[0].size() > std::numeric_limits<uint32_t>::max())
                fail("nonempty vectors and attributes must have matching row counts");
            const size_t dim = vectors[0].size();
            for (const auto &v : vectors)
            {
                if (v.size() != dim)
                    fail("inconsistent vector dimensions");
                for (float x : v)
                    if (!std::isfinite(x))
                        fail("non-finite vector component");
            }
            for (const auto &p : attributes)
                if (!std::isfinite(p.x) || !std::isfinite(p.y))
                    fail("non-finite attribute");
            config_ = config;
            vectors_ = std::move(vectors);
            attributes_ = std::move(attributes);
            dim_ = static_cast<uint32_t>(dim);
            initialize_distance();
            if (progress)
                *progress << "[quadtree] building balanced tree: " << vectors_.size()
                          << " points, dim=" << dim_ << ", leaf_size=" << config_.leaf_size << '\n'
                          << std::flush;
            const auto tree_start = std::chrono::steady_clock::now();
            make_tree();
            if (progress)
                *progress << "[quadtree] tree ready: " << nodes_.size() << " nodes, "
                          << std::chrono::duration<double>(std::chrono::steady_clock::now() - tree_start).count()
                          << " s; allocating layer storage\n"
                          << std::flush;
            for (size_t ni = 0; ni < nodes_.size(); ++ni)
                register_level_members(ni);
            if (progress)
                *progress << "[quadtree] " << edges_.size() << " layers; building graphs bottom-up, M="
                          << config_.M << ", ef_construction=" << config_.ef_construction << '\n'
                          << std::flush;
            quad_detail::build_graphs(nodes_, edges_, vectors_, config_.M, config_.ef_construction, config_.threads, progress);
        }

        // Prepare eligibility before graph traversal. Fully covered tree nodes
        // need no per-point attribute checks; boundary leaves are checked once.
        FilterFlags prepare_flags(const QuadRect &range) const
        {
            check_range(range);
            FilterFlags flags(size(), 0);
            std::vector<size_t> cover;
            if (!nodes_.empty())
                cover_nodes(0, range, cover);
            for (size_t ni : cover)
            {
                const Node &node = nodes_[ni];
                const bool complete = range.contains(node.bounds);
                for (Id id : node.ids)
                    if (complete || range.contains(attributes_[id]))
                        flags[id] = 1;
            }
            return flags;
        }

        // Timing starts after entry sampling and covers the graph search only.
        // Timed search with externally prepared flags, passed without copying.
        // Caller must use prepare_flags(range) from this index for this rectangle.
        std::vector<Result> search(const std::vector<float> &query, const QuadRect &range,
                                   const FilterFlags &flags, size_t k, float &searchtime, size_t ef = 100,
                                   const QuadSearchOptions &options = {}) const
        {
            check_range(range);
            if (flags.size() != size())
                fail("filter flags size mismatch");
            if (query.size() != dim_)
                fail("query dimension mismatch");
            for (float x : query)
                if (!std::isfinite(x))
                    fail("non-finite query component");
            if (!ef)
                fail("ef must be positive");
            if (!options.entry_points)
                fail("entry_points must be positive");
            if (!k || nodes_.empty())
                return {};
            k = std::min(k, vectors_.size());
            ef = std::min(vectors_.size(), std::max(ef, k));
            const size_t limit = options.edge_limit ? options.edge_limit : config_.M;
            const auto entries = sample_entries(range, options.entry_points, options.seed);
            timeval t1, t2;
            gettimeofday(&t1, NULL);
            auto best = search_pdg(query, range, flags, ef, k, limit, options, entries);
            gettimeofday(&t2, NULL);
            searchtime += elapsed_seconds(t1, t2);
            std::vector<Result> out;
            while (!best.empty())
            {
                out.push_back(best.top());
                best.pop();
            }
            std::reverse(out.begin(), out.end());
            return out;
        }

        // Uniform sampling without replacement over eligible points. Fully covered
        // nodes contribute their member counts; only boundary leaves are filtered.
        // These blocks are used only to sample entries, never as separate ANN searches.
        std::vector<Id> sample_entries(const QuadRect &range, size_t count, uint32_t seed = 42) const
        {
            check_range(range);
            if (!count || nodes_.empty())
                return {};
            struct Block
            {
                size_t node;
                bool complete;
                std::vector<Id> partial;
                size_t end;
            };
            std::vector<size_t> cover;
            cover_nodes(0, range, cover);
            std::vector<Block> blocks;
            size_t total = 0;
            for (size_t ni : cover)
            {
                Block b{ni, range.contains(nodes_[ni].bounds), {}, 0};
                if (!b.complete)
                    for (Id id : nodes_[ni].ids)
                        if (range.contains(attributes_[id]))
                            b.partial.push_back(id);
                size_t matches = b.complete ? nodes_[ni].ids.size() : b.partial.size();
                if (!matches)
                    continue;
                total += matches;
                b.end = total;
                blocks.push_back(std::move(b));
            }
            count = std::min(count, total);
            std::vector<Id> entries;
            entries.reserve(count);
            std::unordered_set<size_t> ranks;
            std::mt19937 generator(seed);
            // Floyd's algorithm avoids unbounded rejection loops on small ranges.
            for (size_t j = total - count; j < total; ++j)
            {
                size_t rank = std::uniform_int_distribution<size_t>(0, j)(generator);
                if (!ranks.insert(rank).second)
                {
                    rank = j;
                    ranks.insert(rank);
                }
                auto it = std::upper_bound(blocks.begin(), blocks.end(), rank,
                                           [](size_t r, const Block &b)
                                           { return r < b.end; });
                size_t start = it == blocks.begin() ? 0 : (it - 1)->end;
                entries.push_back(it->complete ? nodes_[it->node].ids[rank - start] : it->partial[rank - start]);
            }
            return entries;
        }

        // Version 2 stores topology, node membership and every node's approximate RNG
        // explicitly. Version 1 remains readable (it reconstructs topology on load).
        void save(const std::string &path, std::ostream *progress = nullptr) const
        {
            if (nodes_.empty())
                fail("index has not been built");
            const auto parent = std::filesystem::path(path).parent_path();
            if (!parent.empty())
                std::filesystem::create_directories(parent);
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            if (!out)
                fail("cannot create index: " + path);
            out.write("IRQDIDX1", 8);
            write_u32(out, 2);
            write_u32(out, static_cast<uint32_t>(vectors_.size()));
            write_u32(out, dim_);
            write_u32(out, config_.M);
            write_u32(out, config_.ef_construction);
            write_u32(out, config_.leaf_size);
            write_u32(out, config_.max_depth);
            write_u32(out, static_cast<uint32_t>(nodes_.size()));
            quad_detail::BuildProgress data_progress(progress, "save vectors + attributes", vectors_.size(), "points");
            for (size_t id = 0; id < vectors_.size(); ++id)
            {
                write_double(out, attributes_[id].x);
                write_double(out, attributes_[id].y);
                for (float x : vectors_[id])
                {
                    uint32_t bits;
                    std::memcpy(&bits, &x, 4);
                    write_u32(out, bits);
                }
                if (!out)
                    fail("failed writing index: " + path);
                data_progress.tick();
            }
            data_progress.finish();
            size_t graph_rows = 0;
            for (const auto &node : nodes_)
                graph_rows += node.ids.size();
            quad_detail::BuildProgress graph_progress(progress, "save topology + graph rows", graph_rows, "rows");
            for (const auto &n : nodes_)
            {
                write_double(out, n.bounds.xmin);
                write_double(out, n.bounds.xmax);
                write_double(out, n.bounds.ymin);
                write_double(out, n.bounds.ymax);
                write_u32(out, n.depth);
                for (int32_t c : n.children)
                    write_u32(out, c < 0 ? UINT32_MAX : uint32_t(c));
                write_u32(out, static_cast<uint32_t>(n.ids.size()));
                for (Id id : n.ids)
                    write_u32(out, id);
                for (Id pid : n.ids)
                {
                    const auto &row = edges_[n.depth][pid];
                    write_u32(out, static_cast<uint32_t>(row.size()));
                    for (Id id : row)
                        write_u32(out, id);
                    graph_progress.tick();
                }
                if (!out)
                    fail("failed writing index: " + path);
            }
            out.flush();
            if (!out)
                fail("failed writing index: " + path);
            graph_progress.finish();
        }

        static QuadTreeIndex load(const std::string &path)
        {
            std::ifstream in(path, std::ios::binary);
            if (!in)
                fail("cannot open index: " + path);
            char magic[8];
            in.read(magic, 8);
            if (!in || std::memcmp(magic, "IRQDIDX1", 8))
                fail("not a quadtree index");
            uint32_t version = read_u32(in);
            if (version != 1 && version != 2)
                fail("unsupported quadtree index version");
            QuadTreeIndex index;
            uint32_t count = read_u32(in);
            index.dim_ = read_u32(in);
            index.config_.M = read_u32(in);
            index.config_.ef_construction = read_u32(in);
            index.config_.leaf_size = read_u32(in);
            index.config_.max_depth = read_u32(in);
            validate_config(index.config_);
            uint32_t node_count = read_u32(in);
            if (!count || !index.dim_ || count > uint32_t(std::numeric_limits<int32_t>::max()) || !node_count)
                fail("invalid index dimensions");
            if (node_count > uint32_t(std::numeric_limits<int32_t>::max()) ||
                node_count > 1 + uint64_t(count) * index.config_.max_depth)
                fail("invalid quadtree node count");
            // Check file size before allocating based on an untrusted header.
            const uint64_t minimum = uint64_t(count) * (16 + uint64_t(index.dim_) * 4);
            auto start = in.tellg();
            in.seekg(0, std::ios::end);
            auto end = in.tellg();
            if (end < start || uint64_t(end - start) < minimum)
                fail("truncated index payload");
            in.seekg(start);
            index.vectors_.resize(count);
            index.attributes_.resize(count);
            for (Id id = 0; id < count; ++id)
            {
                auto &p = index.attributes_[id];
                p.x = read_double(in);
                p.y = read_double(in);
                if (!std::isfinite(p.x) || !std::isfinite(p.y))
                    fail("invalid stored attribute");
                auto &v = index.vectors_[id];
                v.resize(index.dim_);
                for (float &x : v)
                {
                    uint32_t bits = read_u32(in);
                    std::memcpy(&x, &bits, 4);
                    if (!std::isfinite(x))
                        fail("invalid stored vector");
                }
            }
            if (version == 1)
            {
                // Version 1 omitted topology: preserve its original midpoint splits.
                index.make_tree(true);
                if (node_count != index.nodes_.size())
                    fail("quadtree topology mismatch");
            }
            else
            {
                // Each nonempty node needs >=56 metadata bytes, one ID and a degree.
                if (uint64_t(end - in.tellg()) < uint64_t(node_count) * 64)
                    fail("truncated quadtree nodes");
                index.nodes_.resize(node_count);
            }
            for (size_t ni = 0; ni < index.nodes_.size(); ++ni)
            {
                auto &n = index.nodes_[ni];
                if (version == 2)
                {
                    n.bounds.xmin = read_double(in);
                    n.bounds.xmax = read_double(in);
                    n.bounds.ymin = read_double(in);
                    n.bounds.ymax = read_double(in);
                    n.depth = read_u32(in);
                    if (!n.bounds.valid() || n.depth > index.config_.max_depth)
                        fail("invalid node bounds/depth");
                    for (int32_t &c : n.children)
                    {
                        uint32_t id = read_u32(in);
                        if (id != UINT32_MAX && id >= node_count)
                            fail("invalid child node ID");
                        c = id == UINT32_MAX ? -1 : static_cast<int32_t>(id);
                    }
                    uint32_t members = read_u32(in);
                    if (!members || members > count || uint64_t(end - in.tellg()) < uint64_t(members) * 8)
                        fail("invalid or truncated node membership");
                    n.ids.resize(members);
                    for (size_t row = 0; row < members; ++row)
                    {
                        n.ids[row] = read_u32(in);
                        if (n.ids[row] >= count || (row && n.ids[row] <= n.ids[row - 1]))
                            fail("invalid node member IDs");
                    }
                }
                index.register_level_members(ni);
                for (size_t row = 0; row < n.ids.size(); ++row)
                {
                    uint32_t degree = read_u32(in);
                    if (degree > index.config_.M || degree >= n.ids.size())
                        fail("invalid graph degree");
                    auto &edges = index.edges_[n.depth][n.ids[row]];
                    edges.resize(degree);
                    for (Id &id : edges)
                    {
                        id = read_u32(in);
                        if (id == n.ids[row] || !std::binary_search(n.ids.begin(), n.ids.end(), id))
                            fail("graph edge outside its quadtree node");
                    }
                    auto sorted = edges;
                    std::sort(sorted.begin(), sorted.end());
                    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
                        fail("duplicate graph edge");
                }
            }
            index.validate_tree();
            if (in.peek() != std::char_traits<char>::eof())
                fail("unexpected trailing index data");
            index.initialize_distance();
            return index;
        }

        size_t size() const { return vectors_.size(); }
        uint32_t dimension() const { return dim_; }
        const std::vector<Node> &nodes() const { return nodes_; }
        size_t level_count() const { return edges_.size(); }
        const QuadConfig &config() const { return config_; }
        const std::vector<float> &point_vector(Id id) const { return vectors_.at(id); }
        QuadPoint point_attributes(Id id) const { return attributes_.at(id); }

    private:
        // Same get/set semantics as searcher::Bitset, with portable allocation:
        // search_containers.h depends on the legacy Linux-only memory_utils.h.
        class QueryVisited
        {
            std::vector<uint64_t> words_;

        public:
            explicit QueryVisited(size_t count) : words_((count + 63) / 64, 0) {}
            bool get(Id id) const { return (words_[id / 64] >> (id % 64)) & 1; }
            void set(Id id) { words_[id / 64] |= uint64_t(1) << (id % 64); }
        };

        void prefetch_point(Id id) const
        {
            const auto &data = vectors_[id];
            // Only prefetch addresses within the actual vector allocation.
            for (size_t offset = 0; offset < data.size(); offset += 64 / sizeof(float))
            {
#if defined(__GNUC__) || defined(__clang__)
                __builtin_prefetch(data.data() + offset, 0, 3);
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
                _mm_prefetch(reinterpret_cast<const char *>(data.data() + offset), _MM_HINT_T0);
#endif
            }
        }

        void initialize_distance()
        {
            space_ = std::make_shared<hnswlib::L2Space>(dim_);
            fstdistfunc_ = space_->get_dist_func();
            dist_func_param_ = space_->get_dist_func_param();
        }
        // Heap control follows iPreGraph; direct selection is followed by the
        // two-hop cross-range DFS from search_zorder.h, on quadtree paths.
        std::priority_queue<Result> search_pdg(
            const std::vector<float> &query, const QuadRect &range, const FilterFlags &flags, size_t ef,
            size_t query_k, size_t edge_limit,
            const QuadSearchOptions &options, const std::vector<Id> &entries) const
        {
            std::priority_queue<Result, std::vector<Result>, std::greater<Result>> candidate_set;
            std::priority_queue<Result> top_candidates;
            QueryVisited visited_set(size());
            for (Id pid : entries)
            {
                visited_set.set(pid);
                const dist_t dis = fstdistfunc_(query.data(), vectors_[pid].data(), dist_func_param_);
                candidate_set.emplace(dis, pid);
                top_candidates.emplace(dis, pid);
            }
            // A rectangle can contain no objects. Do not call top() on an empty heap.
            if (top_candidates.empty())
                return top_candidates;
            double lowerBound = top_candidates.top().first;

            auto process_edges = [&](const std::vector<Id> &selected_edges)
            {
                for (size_t i = 0; i < std::min(selected_edges.size(), size_t(3)); ++i)
                    prefetch_point(selected_edges[i]);
                for (Id neighbor_id : selected_edges)
                {
                    // select_intra_range_edges has already marked these distinct IDs visited;
                    // process every returned neighbor once, without skipping it.
                    const dist_t dis = fstdistfunc_(query.data(), vectors_[neighbor_id].data(), dist_func_param_);
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
            };
            while (!candidate_set.empty())
            {
                const auto current_point_pair = candidate_set.top();
                if (current_point_pair.first > lowerBound)
                    break;
                candidate_set.pop();
                const Id current_pid = current_point_pair.second;
                process_edges(select_edges_impl(current_pid, range, flags, edge_limit, visited_set));
                // std::vector<Id> cross_edges;
                // const size_t cross_limit = edge_limit / 2;
                // cross_edges.reserve(std::min(cross_limit, size()));
                // select_cross_range_edges(current_pid, range, flags, cross_limit, 2, cross_edges, visited_set);
                // process_edges(cross_edges);
            }
            while (top_candidates.size() > query_k)
                top_candidates.pop();
            return top_candidates;
        }

        QuadConfig config_;
        uint32_t dim_ = 0;
        std::shared_ptr<hnswlib::L2Space> space_;
        hnswlib::DISTFUNC<dist_t> fstdistfunc_ = nullptr;
        void *dist_func_param_ = nullptr;
        std::vector<std::vector<float>> vectors_;
        std::vector<QuadPoint> attributes_;
        std::vector<Node> nodes_;
        std::vector<EdgeLayer> edges_;                    // [depth][point ID] -> neighbor IDs
        std::vector<std::vector<int32_t>> node_at_level_; // [depth][point ID] -> tree node ID

        void register_level_members(size_t ni)
        {
            const Node &node = nodes_[ni];
            while (edges_.size() <= node.depth)
            {
                edges_.emplace_back(vectors_.size());
                node_at_level_.emplace_back(vectors_.size(), -1);
            }
            for (Id id : node.ids)
            {
                auto &owner = node_at_level_[node.depth][id];
                if (owner != -1)
                    fail("duplicate point membership at tree level");
                owner = static_cast<int32_t>(ni);
            }
        }

        [[noreturn]] static void fail(const std::string &s) { throw std::runtime_error(s); }
        static void validate_config(const QuadConfig &c)
        {
            if (!c.M || c.ef_construction < c.M || !c.leaf_size || !c.max_depth || c.max_depth > 64 || !c.threads)
                fail("require M > 0, ef_construction >= M, leaf_size > 0, 1 <= max_depth <= 64, threads > 0");
        }
        static void check_range(const QuadRect &r)
        {
            if (!r.valid())
                fail("invalid query rectangle");
        }
        void validate_tree() const
        {
            if (nodes_.empty() || nodes_[0].depth != 0 || nodes_[0].ids.size() != vectors_.size())
                fail("invalid quadtree root");
            std::vector<unsigned> parents(nodes_.size(), 0);
            for (size_t ni = 0; ni < nodes_.size(); ++ni)
            {
                const auto &n = nodes_[ni];
                if (ni && parents[ni] != 1)
                    fail("orphan quadtree node");
                for (Id id : n.ids)
                    if (!n.bounds.contains(attributes_[id]))
                        fail("point outside node bounds");
                std::vector<Id> children_ids;
                if (!n.leaf())
                    children_ids.reserve(n.ids.size());
                for (int32_t c : n.children)
                    if (c >= 0)
                    {
                        if (size_t(c) <= ni || size_t(c) >= nodes_.size() || ++parents[c] != 1)
                            fail("invalid quadtree parent/child relationship");
                        const auto &child = nodes_[c];
                        if (child.depth != n.depth + 1 || !n.bounds.contains(child.bounds))
                            fail("invalid child bounds/depth");
                        if (child.ids.size() > n.ids.size() - children_ids.size())
                            fail("invalid child membership count");
                        children_ids.insert(children_ids.end(), child.ids.begin(), child.ids.end());
                    }
                if (!n.leaf())
                {
                    std::sort(children_ids.begin(), children_ids.end());
                    if (children_ids != n.ids)
                        fail("children do not partition their parent");
                }
            }
        }
        QuadRect member_bounds(const std::vector<Id> &ids) const
        {
            const auto first = attributes_[ids.front()];
            QuadRect bounds{first.x, first.x, first.y, first.y};
            for (Id id : ids)
            {
                const auto p = attributes_[id];
                bounds.xmin = std::min(bounds.xmin, p.x);
                bounds.xmax = std::max(bounds.xmax, p.x);
                bounds.ymin = std::min(bounds.ymin, p.y);
                bounds.ymax = std::max(bounds.ymax, p.y);
            }
            return bounds;
        }
        void make_tree(bool legacy = false)
        {
            nodes_.clear();
            edges_.clear();
            node_at_level_.clear();
            std::vector<Id> ids(vectors_.size());
            std::iota(ids.begin(), ids.end(), 0);
            if (legacy)
            {
                const auto bounds = member_bounds(ids);
                split_legacy(std::move(ids), bounds, 0);
            }
            else
                split(std::move(ids), 0);
        }
        int32_t split(std::vector<Id> ids, uint32_t depth)
        {
            if (nodes_.size() >= size_t(std::numeric_limits<int32_t>::max()))
                fail("too many quadtree nodes");
            const int32_t ni = static_cast<int32_t>(nodes_.size());
            Node node;
            node.bounds = member_bounds(ids);
            node.depth = depth;
            node.ids = std::move(ids);
            nodes_.push_back(std::move(node));
            const size_t count = nodes_[ni].ids.size();
            if (count <= config_.leaf_size || depth >= config_.max_depth)
                return ni;

            // Rank splits guarantee progress even if all coordinates coincide. Each
            // side has its own y median: this is a balanced four-way spatial tree,
            // not a fixed-center subdivision. Equal coordinates are ordered by ID.
            std::vector<Id> ordered = nodes_[ni].ids;
            const auto by_x = [&](Id a, Id b)
            {
                const auto x = attributes_[a], y = attributes_[b];
                if (x.x != y.x)
                    return x.x < y.x;
                if (x.y != y.y)
                    return x.y < y.y;
                return a < b;
            };
            const auto by_y = [&](Id a, Id b)
            {
                const auto x = attributes_[a], y = attributes_[b];
                if (x.y != y.y)
                    return x.y < y.y;
                if (x.x != y.x)
                    return x.x < y.x;
                return a < b;
            };
            const size_t mid = (count + 1) / 2;
            std::nth_element(ordered.begin(), ordered.begin() + mid, ordered.end(), by_x);
            std::array<std::vector<Id>, 4> groups;
            for (unsigned side = 0; side < 2; ++side)
            {
                const size_t first = side ? mid : 0, last = side ? count : mid;
                const size_t cut = first + (last - first + 1) / 2;
                if (cut < last)
                    std::nth_element(ordered.begin() + first, ordered.begin() + cut, ordered.begin() + last, by_y);
                groups[side].assign(ordered.begin() + first, ordered.begin() + cut);
                groups[side + 2].assign(ordered.begin() + cut, ordered.begin() + last);
            }
            for (unsigned q = 0; q < 4; ++q)
            {
                if (groups[q].empty())
                    continue;
                // Membership and graph row lookup use original ID order everywhere.
                std::sort(groups[q].begin(), groups[q].end());
                const int32_t child = split(std::move(groups[q]), depth + 1);
                nodes_[ni].children[q] = child;
            }
            return ni;
        }
        int32_t split_legacy(std::vector<Id> ids, QuadRect bounds, uint32_t depth)
        {
            if (nodes_.size() >= size_t(std::numeric_limits<int32_t>::max()))
                fail("too many quadtree nodes");
            const int32_t ni = static_cast<int32_t>(nodes_.size());
            Node n;
            n.bounds = bounds;
            n.depth = depth;
            n.ids = std::move(ids);
            nodes_.push_back(std::move(n));
            if (nodes_[ni].ids.size() <= config_.leaf_size || depth >= config_.max_depth)
                return ni;
            // Half-sums avoid overflow even for large finite coordinates.
            double mx = bounds.xmin * .5 + bounds.xmax * .5;
            double my = bounds.ymin * .5 + bounds.ymax * .5;
            bool sx = bounds.xmin < mx && mx < bounds.xmax;
            bool sy = bounds.ymin < my && my < bounds.ymax;
            if (!sx && !sy)
                return ni;
            std::array<std::vector<Id>, 4> groups;
            for (Id id : nodes_[ni].ids)
            {
                auto p = attributes_[id];
                unsigned q = unsigned(sx && p.x >= mx) | (unsigned(sy && p.y >= my) << 1);
                groups[q].push_back(id);
            }
            for (unsigned q = 0; q < 4; ++q)
            {
                if (groups[q].empty())
                    continue;
                QuadRect b = bounds;
                if (sx)
                {
                    if (q & 1)
                        b.xmin = mx;
                    else
                        b.xmax = mx;
                }
                if (sy)
                {
                    if (q & 2)
                        b.ymin = my;
                    else
                        b.ymax = my;
                }
                int32_t child = split_legacy(std::move(groups[q]), b, depth + 1);
                nodes_[ni].children[q] = child;
            }
            return ni;
        }
        void cover_nodes(size_t ni, const QuadRect &r, std::vector<size_t> &out) const
        {
            const Node &n = nodes_[ni];
            if (!r.intersects(n.bounds))
                return;
            if (r.contains(n.bounds) || n.leaf())
            {
                out.push_back(ni);
                return;
            }
            for (int32_t c : n.children)
                if (c >= 0)
                    cover_nodes(c, r, out);
        }
        // Descend using the CURRENT data object's numeric attributes, not the
        // query vector or the query rectangle. Closed MBRs can overlap for equal
        // coordinates; only that case needs the membership table as a tie-break.
        int32_t child_containing_point(const Node &node, Id pid, const QuadPoint &point) const
        {
            if (node.leaf())
                return -1;
            std::array<int32_t, 4> matches{};
            size_t count = 0;
            for (int32_t child : node.children)
                if (child >= 0)
                {
                    if (nodes_[child].bounds.contains(point))
                        matches[count++] = child;
                }
            if (count == 1)
                return matches[0];
            if (count > 1)
            {
                const int32_t owner = node_at_level_[node.depth + 1][pid];
                if (std::find(matches.begin(), matches.begin() + count, owner) != matches.begin() + count)
                    return owner;
            }
            fail("current point has no valid child on attribute descent");
        }
        std::vector<Id> select_edges_impl(Id pid, const QuadRect &range, const FilterFlags &flags, size_t limit,
                                          QueryVisited &visited) const
        {
            std::vector<Id> selected;
            selected.reserve(std::min(limit, size()));
            const QuadPoint point = attributes_[pid];
            size_t ni = 0; // Restart at root for every expanded ANN candidate.
            for (;;)
            {
                const Node &n = nodes_[ni];
                const int32_t own_child = child_containing_point(n, pid, point);
                bool other_child_intersects = false;
                for (int32_t c : n.children)
                    if (c >= 0)
                    {
                        const auto &child = nodes_[c];
                        // Query bounds decide whether to skip this level; the current
                        // object's attributes decide which child to descend into.
                        if (c != own_child && range.intersects(child.bounds))
                            other_child_intersects = true;
                    }
                // Skip a parent only if ALL other children miss the query. An area
                // comparison would be incorrect for zero-width/zero-height queries.
                if (own_child < 0 || other_child_intersects)
                {
                    for (Id id : edges_[n.depth][pid])
                    {
                        if (visited.get(id) || !flags[id])
                            continue;
                        // Mark at selection time so another level cannot select
                        // the same neighbor and consume its budget a second time.
                        visited.set(id);
                        selected.push_back(id);
                        if (selected.size() == limit)
                            return selected;
                    }
                }
                if (own_child < 0)
                    break;
                ni = static_cast<size_t>(own_child);
            }
            return selected;
        }

        // Adapt search_zorder.h::select_cross_range_edges: each transit point starts at the root,
        // descends along its own attribute path, and skips levels whose other
        // children all miss the query. Leaves are handled by direct selection.
        void select_cross_range_edges(Id pid, const QuadRect &range, const FilterFlags &flags, size_t limit,
                 unsigned hop, std::vector<Id> &selected, QueryVisited &visited) const
        {
            if (!hop || selected.size() >= limit)
                return;
            const QuadPoint point = attributes_[pid];
            int32_t ni = 0;
            while (ni >= 0)
            {
                const Node &node = nodes_[ni];
                if (!range.intersects(node.bounds))
                    break; // Prune only this point's path.
                const int32_t own_child = child_containing_point(node, pid, point);
                bool other_child_intersects = false;
                for (int32_t child : node.children)
                    if (child >= 0 && child != own_child && range.intersects(nodes_[child].bounds))
                        other_child_intersects = true;
                if (other_child_intersects)
                {
                    for (Id id : edges_[node.depth][pid])
                    {
                        if (visited.get(id))
                            continue;
                        if (flags[id])
                        {
                            visited.set(id);
                            selected.push_back(id);
                        }
                        else if (hop > 1)
                        {
                            // Do not mark transit points globally: a later route
                            // may reach them with a larger remaining hop budget.
                            select_cross_range_edges(id, range, flags, limit, hop - 1, selected, visited);
                        }
                        if (selected.size() >= limit)
                            return;
                    }
                }
                ni = own_child;
            }
        }
        static void write_u32(std::ostream &out, uint32_t v)
        {
            for (unsigned j = 0; j < 4; ++j)
                out.put(static_cast<char>((v >> (8 * j)) & 255));
        }
        static uint32_t read_u32(std::istream &in)
        {
            uint32_t v = 0;
            for (unsigned j = 0; j < 4; ++j)
            {
                int c = in.get();
                if (c == std::char_traits<char>::eof())
                    fail("truncated index");
                v |= uint32_t(static_cast<unsigned char>(c)) << (8 * j);
            }
            return v;
        }
        static void write_double(std::ostream &out, double x)
        {
            static_assert(sizeof(double) == 8 && sizeof(float) == 4 && std::numeric_limits<double>::is_iec559 &&
                              std::numeric_limits<float>::is_iec559,
                          "IEEE floating point required");
            uint64_t bits;
            std::memcpy(&bits, &x, 8);
            write_u32(out, uint32_t(bits));
            write_u32(out, uint32_t(bits >> 32));
        }
        static double read_double(std::istream &in)
        {
            uint64_t bits = read_u32(in);
            bits |= uint64_t(read_u32(in)) << 32;
            double x;
            std::memcpy(&x, &bits, 8);
            return x;
        }
    };
} // namespace iPreGraph
