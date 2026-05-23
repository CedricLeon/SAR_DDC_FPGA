/* bench_configs.cpp — S0 sequential executor.
 *
 * Loads up to RunConfig::subset patches from the test NPY, cycles through them
 * for warmup and timed iterations, and records per-stage timings with StageTimer.
 *
 * Timer labelling convention (matches stage_timer.hpp usage example):
 *   mark placed BEFORE the operation → interval attributed to that mark label.
 *   Sentinel marks (leading '_') delimit boundaries but are excluded from summary.
 *
 * Stage sequence per scenario (SHyp):
 *   compress: normalize → g_a → h_a → eb → h_s → gc → _end
 *   full:     normalize → g_a → h_a → eb → h_s → gc → g_s → denorm → _end
 *
 * Stage sequence per scenario (FP):
 *   compress: normalize → g_a → eb → _end
 *   full:     normalize → g_a → eb → g_s → denorm → _end
 */

#include "bench_configs.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>

#include "npy_io.hpp"

namespace ddc {

// ---------------------------------------------------------------------------
// Helper: run one full timed iteration given the fixed scenario and arch.
// Inlined by the compiler once the booleans are known at the call site.
// ---------------------------------------------------------------------------
static void run_one(BenchPipeline& pl, PatchState& s,
                    StageTimer& timer, bool is_shyp, bool is_compress)
{
    if (is_shyp) {
        if (is_compress) {
            timer.mark("normalize"); pl.stage_normalize(s);
            timer.mark("g_a");      pl.stage_ga(s);
            timer.mark("h_a");      pl.stage_ha(s);
            timer.mark("eb");       pl.stage_eb(s);
            timer.mark("h_s");      pl.stage_hs(s);
            timer.mark("gc");       pl.stage_gc(s);
            timer.mark("_end");
        } else {
            timer.mark("normalize"); pl.stage_normalize(s);
            timer.mark("g_a");      pl.stage_ga(s);
            timer.mark("h_a");      pl.stage_ha(s);
            timer.mark("eb");       pl.stage_eb(s);
            timer.mark("h_s");      pl.stage_hs(s);
            timer.mark("gc");       pl.stage_gc(s);
            timer.mark("g_s");      pl.stage_gs(s);
            timer.mark("denorm");   pl.stage_denorm(s);
            timer.mark("_end");
        }
    } else {
        if (is_compress) {
            timer.mark("normalize"); pl.stage_normalize(s);
            timer.mark("g_a");      pl.stage_ga(s);
            timer.mark("eb");       pl.stage_eb(s);
            timer.mark("_end");
        } else {
            timer.mark("normalize"); pl.stage_normalize(s);
            timer.mark("g_a");      pl.stage_ga(s);
            timer.mark("eb");       pl.stage_eb(s);
            timer.mark("g_s");      pl.stage_gs(s);
            timer.mark("denorm");   pl.stage_denorm(s);
            timer.mark("_end");
        }
    }
    timer.commit();
}

// ---------------------------------------------------------------------------
// Helper: run one iteration without timing (warmup path)
// ---------------------------------------------------------------------------
static void run_one_notimed(BenchPipeline& pl, PatchState& s,
                            bool is_shyp, bool is_compress)
{
    pl.stage_normalize(s);
    pl.stage_ga(s);
    if (is_shyp) {
        pl.stage_ha(s);
        pl.stage_eb(s);
        pl.stage_hs(s);
        pl.stage_gc(s);
    } else {
        pl.stage_eb(s);
    }
    if (!is_compress) {
        pl.stage_gs(s);
        pl.stage_denorm(s);
    }
}

// ---------------------------------------------------------------------------
// run_s0
// ---------------------------------------------------------------------------
BenchResult run_s0(BenchPipeline& pipeline, const RunConfig& cfg)
{
    // Load test patches (N, H, W, 4): channels 0=real, 1=imag, 2=adam, 3=merlin
    NpyArray arr = npy_load(cfg.data_path.string());
    if (arr.shape.size() != 4 || arr.shape[3] != 4)
        throw std::runtime_error("run_s0: expected data shape (N,H,W,4)");

    const int N_avail = static_cast<int>(arr.shape[0]);
    const int H       = static_cast<int>(arr.shape[1]);
    const int W       = static_cast<int>(arr.shape[2]);
    const int P       = std::min(cfg.subset, N_avail);

    if (P == 0)
        throw std::runtime_error("run_s0: no patches available");

    // Extract [P][H*W*2] raw complex patches (channels 0 and 1 from the 4-channel NPY)
    auto raw = arr.as_float32();
    std::vector<std::vector<float>> patches(static_cast<size_t>(P),
                                            std::vector<float>(static_cast<size_t>(H * W * 2)));
    for (int i = 0; i < P; ++i) {
        const float* src = raw + i * H * W * 4;
        for (int p = 0; p < H * W; ++p) {
            patches[i][p * 2 + 0] = src[p * 4 + 0]; // real
            patches[i][p * 2 + 1] = src[p * 4 + 1]; // imag
        }
    }

    PatchState state   = pipeline.make_patch_state(H, W);
    const bool is_shyp = pipeline.uses_hyper();
    const bool is_cmp  = (cfg.scenario == Scenario::compress);

    // Warmup — fills pipeline caches / JIT state; not timed
    for (int w = 0; w < cfg.warmup; ++w) {
        std::memcpy(state.noisy_hwc.data(), patches[w % P].data(),
                    static_cast<size_t>(H * W * 2) * sizeof(float));
        run_one_notimed(pipeline, state, is_shyp, is_cmp);
    }

    // Timed loop
    StageTimer timer;
    auto t_start = std::chrono::steady_clock::now();

    for (int it = 0; it < cfg.iters; ++it) {
        // Copy patch before timing starts — this is benchmark overhead,
        // not pipeline compute.  In production the data arrives via DMA.
        std::memcpy(state.noisy_hwc.data(), patches[it % P].data(),
                    static_cast<size_t>(H * W * 2) * sizeof(float));
        run_one(pipeline, state, timer, is_shyp, is_cmp);
    }

    auto t_end     = std::chrono::steady_clock::now();
    double wall_s  = std::chrono::duration<double>(t_end - t_start).count();

    BenchResult r;
    r.stage_stats            = timer.summary();
    r.total_latency_mean_ms  = timer.total_mean_s() * 1e3;
    r.throughput_fps         = static_cast<double>(cfg.iters) / wall_s;
    r.wall_time_s            = wall_s;
    r.num_iters              = cfg.iters;
    r.is_shyp                = is_shyp;
    return r;
}

} // namespace ddc
