#pragma once

// Bottom-up construction for the quadtree's approximate RNG-style graphs.
// The merge follows segment_graph_builder.h: search an already merged graph, combine
// candidates with inherited child edges, then add/prune reverse edges. Child
// phases have barriers, so no search reads a graph row while it is modified.
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#include "quadtree_progress.h"

namespace iPreGraph {
namespace quad_detail {

// One bounded pool is reused for every tree level and every merge phase.
// Worker zero runs on the caller, avoiding an extra idle construction thread.
class BuildWorkers {
public:
    explicit BuildWorkers(size_t count) : count_(std::max(size_t(1), count)) {
        try {
            for (size_t worker = 1; worker < count_; ++worker)
                threads_.emplace_back([this, worker] { worker_loop(worker); });
        } catch (...) {
            stop();
            throw;
        }
    }
    ~BuildWorkers() { stop(); }
    BuildWorkers(const BuildWorkers&) = delete;
    BuildWorkers& operator=(const BuildWorkers&) = delete;
    size_t size() const { return count_; }

    template<class Function>
    void run(size_t count, Function&& function) {
        if (!count) return;
        if (count_ == 1) {
            for (size_t i = 0; i < count; ++i) function(size_t(0), i);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            job_ = std::forward<Function>(function);
            jobs_ = count;
            grain_ = std::max(size_t(1), std::min(size_t(16), count / (count_ * 4)));
            next_.store(0, std::memory_order_relaxed);
            cancelled_.store(false, std::memory_order_relaxed);
            error_ = nullptr;
            remaining_ = threads_.size();
            ++generation_;
        }
        ready_.notify_all();
        work(0);
        std::unique_lock<std::mutex> lock(mutex_);
        finished_.wait(lock, [this] { return remaining_ == 0; });
        job_ = nullptr;
        if (error_) std::rethrow_exception(error_);
    }

private:
    size_t count_;
    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable ready_, finished_;
    std::function<void(size_t, size_t)> job_;
    std::atomic<size_t> next_{0};
    std::atomic<bool> cancelled_{false};
    size_t jobs_ = 0, grain_ = 1, remaining_ = 0, generation_ = 0;
    bool stopping_ = false;
    std::exception_ptr error_;

    void work(size_t worker) {
        try {
            while (!cancelled_.load(std::memory_order_relaxed)) {
                const size_t first = next_.fetch_add(grain_, std::memory_order_relaxed);
                if (first >= jobs_) break;
                const size_t last = std::min(jobs_, first + grain_);
                for (size_t i = first; i < last; ++i) job_(worker, i);
            }
        } catch (...) {
            cancelled_.store(true, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(mutex_);
            if (!error_) error_ = std::current_exception();
        }
    }
    void worker_loop(size_t worker) {
        size_t seen = 0;
        for (;;) {
            std::unique_lock<std::mutex> lock(mutex_);
            ready_.wait(lock, [this, seen] { return stopping_ || generation_ != seen; });
            if (stopping_) return;
            seen = generation_;
            lock.unlock();
            work(worker);
            lock.lock();
            if (--remaining_ == 0) finished_.notify_one();
        }
    }
    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        for (auto& thread : threads_) if (thread.joinable()) thread.join();
    }
};

template<class Node>
class GraphBuilder {
    using Id = typename std::decay_t<decltype(std::declval<Node>().ids)>::value_type;
    using EdgeLevels = std::vector<std::vector<std::vector<Id>>>;
    using Result = std::pair<double, Id>;
    struct Scratch {
        std::vector<uint32_t> visited;
        uint32_t epoch = 0;
        void begin(size_t count) {
            if (visited.size() != count) { visited.assign(count, 0); epoch = 0; }
            if (++epoch == 0) {
                std::fill(visited.begin(), visited.end(), 0);
                epoch = 1;
            }
        }
    };
    struct Candidate { double distance; Id id; bool inherited; };

public:
    GraphBuilder(std::vector<Node>& nodes, EdgeLevels& edges, const std::vector<std::vector<float>>& vectors,
                 size_t degree, size_t ef, size_t threads, std::ostream* progress)
        : nodes_(nodes), edges_(edges), vectors_(vectors), degree_(degree), ef_(ef),
          workers_(std::min(std::max(size_t(1), threads), std::max(size_t(1), vectors.size()))),
          scratch_(workers_.size()), row_of_id_(vectors.size()), progress_(progress) {}

    void build() {
        size_t max_depth = 0;
        for (const auto& node : nodes_) max_depth = std::max(max_depth, size_t(node.depth));
        std::vector<std::vector<size_t>> levels(max_depth + 1);
        for (size_t ni = 0; ni < nodes_.size(); ++ni) levels[nodes_[ni].depth].push_back(ni);
        for (size_t depth = levels.size(); depth-- > 0;) {
            const auto& level = levels[depth];
            if (progress_) *progress_ << "[quadtree] building layer " << depth << '/' << max_depth
                << " (0=root), " << level.size() << " nodes, " << workers_.size() << " workers\n" << std::flush;
            BuildProgress layer_progress(progress_, "layer " + std::to_string(depth), level.size(), "nodes");
            if (level.size() > 1 && level.size() >= workers_.size()) {
                // Nodes at one depth have disjoint membership. Their row maps,
                // adjacency lists and per-worker visitation state are independent.
                workers_.run(level.size(), [&](size_t worker, size_t i) {
                    build_node(level[i], worker, false);
                    layer_progress.tick();
                });
            } else {
                // In particular, root construction still uses all workers.
                for (size_t ni : level) {
                    if (progress_) *progress_ << "[quadtree] node " << ni << ", layer " << depth
                        << ", " << nodes_[ni].ids.size() << " points\n" << std::flush;
                    build_node(ni, 0, true);
                    layer_progress.tick();
                }
            }
            layer_progress.finish();
        }
    }

private:
    std::vector<Node>& nodes_;
    EdgeLevels& edges_; // [tree depth][original point ID], no per-node copy
    const std::vector<std::vector<float>>& vectors_;
    size_t degree_, ef_;
    BuildWorkers workers_;
    std::vector<Scratch> scratch_;
    // Reused between levels. Same-depth nodes write disjoint global IDs.
    std::vector<size_t> row_of_id_;
    std::ostream* progress_;

    template<class Function>
    void rows(size_t count, size_t worker, bool parallel, const char* phase, Function&& function) {
        if (parallel) {
            BuildProgress progress(progress_, phase, count, "points");
            workers_.run(count, [&](size_t tid, size_t row) {
                function(tid, row);
                progress.tick();
            });
            progress.finish();
        }
        else for (size_t row = 0; row < count; ++row) function(worker, row);
    }
    double distance(Id a, Id b) const {
        const auto& x = vectors_[a];
        const auto& y = vectors_[b];
        double sum = 0;
        for (size_t j = 0; j < x.size(); ++j) {
            const double delta = double(x[j]) - double(y[j]);
            sum += delta * delta;
        }
        return sum;
    }
    static uint64_t mix(uint64_t x) {
        x += UINT64_C(0x9e3779b97f4a7c15);
        x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
        return x ^ (x >> 31);
    }
    std::vector<Result> search(const Node& node, Id query, const std::vector<Id>& active,
                               size_t worker) {
        Scratch& scratch = scratch_[worker];
        scratch.begin(vectors_.size());
        std::priority_queue<Result, std::vector<Result>, std::greater<Result>> frontier;
        std::priority_queue<Result> best;
        const size_t ef = std::min(ef_, active.size());
        auto visit = [&](Id id) {
            if (scratch.visited[id] == scratch.epoch) return;
            scratch.visited[id] = scratch.epoch;
            const Result candidate{distance(query, id), id};
            if (best.size() < ef || candidate < best.top()) {
                frontier.push(candidate);
                best.push(candidate);
                if (best.size() > ef) best.pop();
            }
        };
        // Three reproducible, query-specific entries, independent of scheduling.
        uint64_t seed = mix(uint64_t(query) ^ (uint64_t(node.ids.front()) << 32));
        const size_t entry_count = std::min(size_t(3), active.size());
        size_t entries[3] = {};
        for (size_t j = 0; j < entry_count; ++j) {
            seed = mix(seed);
            size_t entry = static_cast<size_t>(seed % active.size());
            while (std::find(entries, entries + j, entry) != entries + j)
                entry = (entry + 1) % active.size();
            entries[j] = entry;
            visit(active[entry]);
        }
        while (!frontier.empty()) {
            const Result current = frontier.top();
            if (best.size() >= ef && current > best.top()) break;
            frontier.pop();
            for (Id neighbor : edges_[node.depth][current.second]) visit(neighbor);
        }
        std::vector<Result> result;
        result.reserve(best.size());
        while (!best.empty()) { result.push_back(best.top()); best.pop(); }
        std::reverse(result.begin(), result.end());
        return result;
    }

    std::vector<Id> prune(Id query, const std::vector<Id>& inherited,
                          const std::vector<Result>& extra) const {
        std::vector<Candidate> candidates;
        candidates.reserve(inherited.size() + extra.size());
        for (Id id : inherited) if (id != query)
            candidates.push_back({distance(query, id), id, true});
        for (const auto& result : extra) if (result.second != query)
            candidates.push_back({result.first, result.second, false});
        std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            if (a.id != b.id) return a.id < b.id;
            return a.inherited > b.inherited;
        });
        candidates.erase(std::unique(candidates.begin(), candidates.end(),
            [](const Candidate& a, const Candidate& b) { return a.id == b.id; }), candidates.end());
        std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            return a.distance != b.distance ? a.distance < b.distance : a.id < b.id;
        });
        std::vector<Id> selected;
        std::vector<bool> selected_inherited;
        selected.reserve(std::min(degree_, candidates.size()));
        selected_inherited.reserve(std::min(degree_, candidates.size()));
        for (const auto& candidate : candidates) {
            bool keep = true;
            // As in PruneByHeuristic2, already-pruned inherited edges do not
            // suppress one another. Small lists are kept up to the degree cap.
            if (candidates.size() > degree_) {
                for (size_t i = 0; i < selected.size(); ++i) {
                    if (candidate.inherited && selected_inherited[i]) continue;
                    if (distance(candidate.id, selected[i]) < candidate.distance) {
                        keep = false;
                        break;
                    }
                }
            }
            if (keep) {
                selected.push_back(candidate.id);
                selected_inherited.push_back(candidate.inherited);
                if (selected.size() == degree_) break;
            }
        }
        return selected;
    }

    void merge(Node& node, std::vector<Id>& active, const std::vector<Id>& incoming,
               std::vector<unsigned char>& active_rows, std::vector<std::vector<Id>>& reverse,
               size_t worker, bool parallel) {
        if (incoming.empty()) return;
        // Existing rows contain only active vertices until the reverse phase.
        // Incoming rows can therefore be searched/pruned concurrently, without
        // reading any row being changed by a different worker.
        rows(incoming.size(), worker, parallel, "merge: candidate search + pruning", [&](size_t tid, size_t i) {
            const Id id = incoming[i];
            const auto candidates = search(node, id, active, tid);
            edges_[node.depth][id] = prune(id, edges_[node.depth][id], candidates);
        });
        for (Id id : active) reverse[row_of_id_[id]].clear();
        // A deterministic gather avoids concurrent push_back on reverse lists.
        for (Id id : incoming) {
            for (Id neighbor : edges_[node.depth][id]) {
                const size_t row = row_of_id_[neighbor];
                if (active_rows[row]) reverse[row].push_back(id);
            }
        }
        rows(active.size(), worker, parallel, "merge: reverse-edge pruning", [&](size_t, size_t i) {
            const Id id = active[i];
            const size_t row = row_of_id_[id];
            if (reverse[row].empty()) return;
            std::vector<Result> candidates;
            candidates.reserve(reverse[row].size());
            for (Id neighbor : reverse[row]) candidates.emplace_back(distance(id, neighbor), neighbor);
            edges_[node.depth][id] = prune(id, edges_[node.depth][id], candidates);
        });
        for (Id id : incoming) active_rows[row_of_id_[id]] = 1;
        active.insert(active.end(), incoming.begin(), incoming.end());
    }

    void build_leaf(Node& node, size_t worker, bool parallel) {
        // Only bounded base blocks enumerate all point pairs. A leaf forced by
        // max_depth can be arbitrarily large, so it uses the same graph merges.
        constexpr size_t block_size = 32;
        const size_t count = node.ids.size();
        rows(count, worker, parallel, "leaf: block graph construction", [&](size_t, size_t row) {
            const size_t first = row / block_size * block_size;
            const size_t last = std::min(count, first + block_size);
            std::vector<Result> candidates;
            candidates.reserve(last - first - 1);
            for (size_t other = first; other < last; ++other) if (other != row)
                candidates.emplace_back(distance(node.ids[row], node.ids[other]), node.ids[other]);
            edges_[node.depth][node.ids[row]] = prune(node.ids[row], {}, candidates);
        });
        if (count <= block_size) return;
        std::vector<unsigned char> active_rows(count, 0);
        std::vector<std::vector<Id>> reverse(count);
        for (size_t width = block_size; width < count;) {
            const size_t group_width = width > count / 4 ? count : width * 4;
            for (size_t first = 0; first < count; first += group_width) {
                const size_t last = std::min(count, first + group_width);
                std::fill(active_rows.begin() + first, active_rows.begin() + last, 0);
                const size_t initial_end = std::min(last, first + width);
                std::vector<Id> active(node.ids.begin() + first, node.ids.begin() + initial_end);
                active.reserve(last - first);
                std::fill(active_rows.begin() + first, active_rows.begin() + initial_end, 1);
                for (size_t begin = initial_end; begin < last; begin += width) {
                    const size_t end = std::min(last, begin + width);
                    const std::vector<Id> incoming(node.ids.begin() + begin, node.ids.begin() + end);
                    merge(node, active, incoming, active_rows, reverse, worker, parallel);
                }
            }
            if (group_width == count) break;
            width = group_width;
        }
    }

    void build_node(size_t ni, size_t worker, bool parallel) {
        Node& node = nodes_[ni];
        // All rows were allocated before launching workers. Same-depth nodes
        // own disjoint IDs, so each worker writes only its own graph rows.
        for (size_t row = 0; row < node.ids.size(); ++row) row_of_id_[node.ids[row]] = row;
        if (node.leaf()) { build_leaf(node, worker, parallel); return; }
        std::vector<unsigned char> active_rows(node.ids.size(), 0);
        std::vector<std::vector<Id>> reverse(node.ids.size());
        std::vector<Id> active;
        active.reserve(node.ids.size());
        for (int32_t ci : node.children) if (ci >= 0) {
            const Node& child = nodes_[static_cast<size_t>(ci)];
            if (parallel && progress_) *progress_ << "[quadtree] inherit child " << ci
                << ", merged " << active.size() << '/' << node.ids.size() << " points\n" << std::flush;
            rows(child.ids.size(), worker, parallel, "copy child edges", [&](size_t, size_t row) {
                const Id id = child.ids[row];
                edges_[node.depth][id] = edges_[child.depth][id];
            });
            if (active.empty()) {
                active = child.ids;
                for (Id id : active) active_rows[row_of_id_[id]] = 1;
            } else {
                merge(node, active, child.ids, active_rows, reverse, worker, parallel);
            }
        }
    }
};

// Preconditions: valid disjoint child memberships; sorted global node IDs;
// finite vectors with one shared dimension; valid nonempty node topology.
// edges[depth] is preallocated with one empty row per global point ID for every
// tree depth. Rows belonging to different nodes at one depth are disjoint.
// Each returned row stores global IDs, with at most M outgoing edges. This is
// candidate-search/RNG-heuristic construction, not an exact all-pairs RNG.
template<class Node, class EdgeLevels>
void build_graphs(std::vector<Node>& nodes,
                  EdgeLevels& edges,
                  const std::vector<std::vector<float>>& vectors,
                  size_t M, size_t ef_construction, size_t threads, std::ostream* progress = nullptr) {
    if (!M || ef_construction < M || !threads)
        throw std::invalid_argument("require M > 0, ef_construction >= M, threads > 0");
    if (nodes.empty()) return;
    GraphBuilder<Node>(nodes, edges, vectors, M, ef_construction, threads, progress).build();
}

} // namespace quad_detail
} // namespace iPreGraph
