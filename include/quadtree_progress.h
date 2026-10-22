#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <mutex>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>

namespace iPreGraph {
namespace quad_detail {

// A scope-local counter: emit start/end and at most one intermediate line per
// two seconds. Worker threads share the counter; output is serialized/flushed.
class BuildProgress {
public:
    BuildProgress(std::ostream* out, std::string label, size_t total, const char* unit)
        : out_(out), label_(std::move(label)), total_(total), unit_(unit),
          start_(Clock::now()), last_(start_) {
        if (out_) print(0, start_);
    }
    void tick() {
        if (!out_) return;
        const size_t done = done_.fetch_add(1, std::memory_order_relaxed) + 1;
        // Avoid clock reads/locks on every point in large construction phases.
        if (done == total_ || (total_ >= 1024 && done % 64 != 0)) return;
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (!lock.owns_lock()) return;
        const auto now = Clock::now();
        if (now - last_ < std::chrono::seconds(2)) return;
        print(done_.load(std::memory_order_relaxed), now);
        last_ = now;
    }
    // Called only after all workers have joined this phase. Never print success
    // from a destructor: a failed construction must not appear complete.
    void finish() {
        if (out_) print(total_, Clock::now());
    }
private:
    using Clock = std::chrono::steady_clock;
    std::ostream* out_;
    std::string label_;
    size_t total_;
    const char* unit_;
    Clock::time_point start_, last_;
    std::atomic<size_t> done_{0};
    std::mutex mutex_;
    void print(size_t done, Clock::time_point now) {
        std::ostringstream line;
        line << "[quadtree] " << label_ << ": " << done << '/' << total_ << ' ' << unit_
             << " (" << std::fixed << std::setprecision(1)
             << (total_ ? 100.0 * double(done) / double(total_) : 100.0) << "%), "
             << std::chrono::duration<double>(now - start_).count() << " s\n";
        *out_ << line.str() << std::flush;
    }
};

} // namespace quad_detail
} // namespace iPreGraph
