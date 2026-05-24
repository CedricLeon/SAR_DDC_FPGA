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
    // dpu_cores semantics differ by config:
    //   S0/S1:       ignored (S0=1 runner; S1=2 structural runners per role)
    //   nn_only:     N independent BenchPipeline instances (data-parallel ceiling)
    //   P0/P2:       N DPU pipeline lanes
    int                   dpu_cores       = 1;
    // entropy_threads semantics:
    //   S0/S1:       ignored
    //   entropy_only: N concurrent entropy workers (CPU ceiling)
    //   P2:          N entropy consumers
    int                   entropy_threads = 1;
    // Paths used by configs that construct their own BenchPipeline instances
    // (nn_only, entropy_only).  Set by main_benchmark.cpp before dispatch.
    std::filesystem::path xmodel_path;
    std::filesystem::path params_path;
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

// ---------------------------------------------------------------------------
// M3: S1, nn_only, entropy_only
// ---------------------------------------------------------------------------

// S1 channel-parallel: pipeline must have init_s1() called before this.
// stage_ga and stage_gs are replaced by their _s1 (concurrent) variants.
// Reports per-stage latency (wall time of the concurrent g_a / g_s = max of two).
BenchResult run_s1(BenchPipeline& pipeline, const RunConfig& cfg);

// nn_only DPU ceiling: creates cfg.dpu_cores independent BenchPipeline instances,
// each processing a different patch concurrently.  Headline = throughput_fps.
// cfg.xmodel_path and cfg.params_path must be set.
BenchResult run_nn_only(const RunConfig& cfg);

// entropy_only CPU ceiling: creates cfg.entropy_threads BenchPipeline instances
// (DPU used only in warmup to populate state); timed loop runs entropy stages only.
// cfg.xmodel_path and cfg.params_path must be set.
BenchResult run_entropy_only(const RunConfig& cfg);

} // namespace ddc
