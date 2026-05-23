#pragma once
// bench_configs.hpp — executor functions for each benchmark scenario.
//
// Each run_* function runs warmup + timed iterations over test patches,
// instruments stage timings with StageTimer, and returns a BenchResult.
//
// Milestone build order:
//   M1: run_s0  — sequential, 1 DPU core, 1 thread (this file)
//   M3: run_s1, run_nn_only, run_entropy_only
//   M4: run_p0  — coarse 2-stage DPU/entropy pipeline
//   M5: run_p2  — multi-entropy consumers

#include <filesystem>
#include <map>
#include <string>

#include "bench_pipeline.hpp"
#include "stage_timer.hpp"

namespace ddc {

// Scenario: which stages to run.
enum class Scenario {
    compress,   // normalize → g_a → [h_a →] EB → [h_s → GC]  (no reconstruction)
    full,       // compress + g_s + denorm
};

// RunConfig — parameters shared across all bench configs.
struct RunConfig {
    Scenario              scenario        = Scenario::compress;
    int                   warmup          = 5;    // untimed warmup iterations
    int                   iters           = 50;   // timed iterations
    std::filesystem::path data_path;              // test .npy (N, H, W, 4)
    int                   subset          = 20;   // patches loaded (cycled if iters > subset)
    int                   dpu_cores       = 1;    // runner replicas (S0 ignores >1)
    int                   entropy_threads = 1;    // entropy workers (S0 ignores >1)
};

// BenchResult — output of a benchmark run.
struct BenchResult {
    std::map<std::string, StageTimer::Stats> stage_stats; // per-stage aggregated stats
    double total_latency_mean_ms = 0.0;   // sum of all stage means (ms)
    double throughput_fps        = 0.0;   // iters / wall_time_s  (sequential ≈ 1/latency)
    double wall_time_s           = 0.0;   // total timed window
    int    num_iters             = 0;
    bool   is_shyp               = false; // true = ScaleHyperprior; false = FactorizedPrior
};

// ---------------------------------------------------------------------------
// M1: S0 sequential baseline
// ---------------------------------------------------------------------------
// Single DPU core, single thread.  Runs each patch fully before starting the
// next.  Reports per-stage mean / std / p95 latency (ms).
// RunConfig::dpu_cores and ::entropy_threads are accepted but silently ignored.
BenchResult run_s0(BenchPipeline& pipeline, const RunConfig& cfg);

} // namespace ddc
