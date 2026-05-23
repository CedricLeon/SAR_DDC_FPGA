#pragma once
// stage_timer.hpp — per-stage wall-clock accumulator for benchmark_hardware.
//
// C++ analog of the Python StepTimer in scripts/fpga/benchmark_fpga.py:
// a mark/commit pattern records the time *between* consecutive marks within one
// iteration, and summary() aggregates each labelled interval across all iterations.
//
// Usage:
//   StageTimer t;
//   for (int it = 0; it < n_iters; ++it) {
//       t.mark("g_a");   run_g_a();
//       t.mark("h_a");   run_h_a();
//       t.mark("_end");          // sentinel: closes the last real interval
//       t.commit();              // store this iteration's interval durations
//   }
//   for (const auto& [label, s] : t.summary())
//       printf("%-16s %.3f ms\n", label.c_str(), s.mean_s * 1e3);
//
// Labels beginning with '_' are sentinels: they delimit intervals but are
// excluded from summary() (mirrors the Python "_end" convention). Header-only,
// no dependencies beyond the standard library, so it has zero coupling.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace ddc {

class StageTimer {
public:
    using clock = std::chrono::steady_clock;

    // Per-stage statistics (seconds), computed across all committed iterations.
    struct Stats {
        double mean_s   = 0.0;
        double std_s    = 0.0;  // sample standard deviation (n-1), matches statistics.stdev
        double median_s = 0.0;
        double min_s    = 0.0;
        double max_s    = 0.0;
        double p95_s    = 0.0;
        std::size_t n   = 0;
    };

    // Record a timestamp labelled `label`. The interval [this mark, next mark)
    // is attributed to `label` on commit().
    void mark(const std::string& label) {
        marks_.emplace_back(label, clock::now());
    }

    // Compute durations between consecutive marks of the current iteration and
    // append them to the per-label history. Clears the marks for the next round.
    void commit() {
        for (std::size_t i = 0; i + 1 < marks_.size(); ++i) {
            const double dt =
                std::chrono::duration<double>(marks_[i + 1].second - marks_[i].second).count();
            history_[marks_[i].first].push_back(dt);
        }
        marks_.clear();
    }

    bool empty() const { return history_.empty(); }

    // Per-label stats. Sentinel labels (leading '_') are skipped.
    std::map<std::string, Stats> summary() const {
        std::map<std::string, Stats> out;
        for (const auto& kv : history_) {
            if (!kv.first.empty() && kv.first.front() == '_')
                continue;
            out.emplace(kv.first, compute_stats(kv.second));
        }
        return out;
    }

    // Sum of per-stage mean durations (seconds) over all non-sentinel stages.
    double total_mean_s() const {
        double sum = 0.0;
        for (const auto& kv : summary())
            sum += kv.second.mean_s;
        return sum;
    }

private:
    static Stats compute_stats(std::vector<double> v) {
        Stats s;
        s.n = v.size();
        if (v.empty())
            return s;

        std::sort(v.begin(), v.end());

        double sum = 0.0;
        for (double x : v)
            sum += x;
        s.mean_s = sum / static_cast<double>(v.size());
        s.min_s  = v.front();
        s.max_s  = v.back();

        const std::size_t mid = v.size() / 2;
        s.median_s = (v.size() % 2 == 1) ? v[mid] : 0.5 * (v[mid - 1] + v[mid]);

        // 95th percentile via the same index rule as the Python benchmark
        // (sorted[int(0.95 * n)]), clamped to a valid index.
        std::size_t p95_idx = static_cast<std::size_t>(0.95 * static_cast<double>(v.size()));
        if (p95_idx >= v.size())
            p95_idx = v.size() - 1;
        s.p95_s = v[p95_idx];

        if (v.size() > 1) {
            double sq = 0.0;
            for (double x : v)
                sq += (x - s.mean_s) * (x - s.mean_s);
            s.std_s = std::sqrt(sq / static_cast<double>(v.size() - 1));
        }
        return s;
    }

    std::vector<std::pair<std::string, clock::time_point>> marks_;
    std::map<std::string, std::vector<double>>             history_;
};

} // namespace ddc
