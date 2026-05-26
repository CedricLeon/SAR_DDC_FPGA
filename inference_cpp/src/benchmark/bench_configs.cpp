/* bench_configs.cpp — benchmark executors: S0, S1, nn_only, entropy_only.
 *
 * Timer labelling convention: mark placed BEFORE the operation → interval
 * attributed to that label on commit(). Sentinel marks ('_' prefix) delimit
 * boundaries but are excluded from summary().
 *
 * Stage sequences (SHyp):
 *   S0/S1 compress: normalize→g_a→h_a→eb_compress→eb_decompress→h_s→gc_compress→_end
 *   S0/S1 full:     …→gc_compress→gc_decompress→g_s→denorm→_end
 *   entropy_only compress: eb_compress→gc_compress→_end   (scales pre-cached)
 *   entropy_only full:     eb_compress→eb_decompress→gc_compress→gc_decompress→_end
 *
 * Stage sequences (FP):
 *   S0/S1 compress: normalize→g_a→eb_compress→_end
 *   S0/S1 full:     normalize→g_a→eb_compress→eb_decompress→g_s→denorm→_end
 *   entropy_only compress: eb_compress→_end
 *   entropy_only full:     eb_compress→eb_decompress→_end
 */

#include "bench_configs.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <thread>

#include "npy_io.hpp"

namespace ddc {

// ---------------------------------------------------------------------------
// PatchData — loaded patches shared by all configs
// ---------------------------------------------------------------------------
struct PatchData {
    std::vector<std::vector<float>> patches;  // [P][H*W*2], channels 0=real 1=imag
    int H = 0, W = 0, P = 0;
};

static PatchData load_patches(const RunConfig& cfg)
{
    NpyArray arr = npy_load(cfg.data_path.string());
    if (arr.shape.size() != 4 || arr.shape[3] != 4)
        throw std::runtime_error("benchmark: expected data shape (N,H,W,4)");

    PatchData pd;
    pd.H = static_cast<int>(arr.shape[1]);
    pd.W = static_cast<int>(arr.shape[2]);
    const int N_avail = static_cast<int>(arr.shape[0]);
    pd.P = std::min(cfg.subset, N_avail);
    if (pd.P == 0) throw std::runtime_error("benchmark: no patches available");

    auto raw = arr.as_float32();
    pd.patches.resize(static_cast<size_t>(pd.P),
                      std::vector<float>(static_cast<size_t>(pd.H * pd.W * 2)));
    for (int i = 0; i < pd.P; ++i) {
        const float* src = raw + i * pd.H * pd.W * 4;
        for (int p = 0; p < pd.H * pd.W; ++p) {
            pd.patches[i][p * 2 + 0] = src[p * 4 + 0]; // real
            pd.patches[i][p * 2 + 1] = src[p * 4 + 1]; // imag
        }
    }
    return pd;
}

// ---------------------------------------------------------------------------
// Helper: run one full timed iteration given the fixed scenario and arch.
// Inlined by the compiler once the booleans are known at the call site.
// ---------------------------------------------------------------------------
static void run_one(BenchPipeline& pl, PatchState& s,
                    StageTimer& timer, bool is_shyp, bool is_compress)
{
    if (is_shyp) {
        if (is_compress) {
            timer.mark("normalize");     pl.stage_normalize(s);
            timer.mark("g_a");           pl.stage_ga(s);
            timer.mark("h_a");           pl.stage_ha(s);
            timer.mark("eb_compress");   pl.stage_eb_compress(s);
            timer.mark("eb_decompress"); pl.stage_eb_decompress(s);
            timer.mark("h_s");           pl.stage_hs(s);
            timer.mark("gc_compress");   pl.stage_gc_compress(s);
            timer.mark("_end");
        } else {
            timer.mark("normalize");      pl.stage_normalize(s);
            timer.mark("g_a");            pl.stage_ga(s);
            timer.mark("h_a");            pl.stage_ha(s);
            timer.mark("eb_compress");    pl.stage_eb_compress(s);
            timer.mark("eb_decompress");  pl.stage_eb_decompress(s);
            timer.mark("h_s");            pl.stage_hs(s);
            timer.mark("gc_compress");    pl.stage_gc_compress(s);
            timer.mark("gc_decompress");  pl.stage_gc_decompress(s);
            timer.mark("g_s");            pl.stage_gs(s);
            timer.mark("denorm");         pl.stage_denorm(s);
            timer.mark("_end");
        }
    } else {
        if (is_compress) {
            timer.mark("normalize");   pl.stage_normalize(s);
            timer.mark("g_a");         pl.stage_ga(s);
            timer.mark("eb_compress"); pl.stage_eb_compress(s);
            timer.mark("_end");
        } else {
            timer.mark("normalize");     pl.stage_normalize(s);
            timer.mark("g_a");           pl.stage_ga(s);
            timer.mark("eb_compress");   pl.stage_eb_compress(s);
            timer.mark("eb_decompress"); pl.stage_eb_decompress(s);
            timer.mark("g_s");           pl.stage_gs(s);
            timer.mark("denorm");        pl.stage_denorm(s);
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
        pl.stage_eb_compress(s);
        pl.stage_eb_decompress(s);
        pl.stage_hs(s);
        pl.stage_gc_compress(s);
        if (!is_compress)
            pl.stage_gc_decompress(s);
    } else {
        pl.stage_eb_compress(s);
        if (!is_compress)
            pl.stage_eb_decompress(s);
    }
    if (!is_compress) {
        pl.stage_gs(s);
        pl.stage_denorm(s);
    }
}

// ---------------------------------------------------------------------------
// Helper: S1 warmup — same execution path as the timed loop (stage_ga_s1 /
// stage_gs_s1) so that std::thread launch and VART concurrent-runner paths
// are exercised before the timing window opens.
// ---------------------------------------------------------------------------
static void run_one_s1_notimed(BenchPipeline& pl, PatchState& s,
                               bool is_shyp, bool is_compress)
{
    pl.stage_normalize(s);
    pl.stage_ga_s1(s);
    if (is_shyp) {
        pl.stage_ha(s);
        pl.stage_eb_compress(s);
        pl.stage_eb_decompress(s);
        pl.stage_hs(s);
        pl.stage_gc_compress(s);
        if (!is_compress)
            pl.stage_gc_decompress(s);
    } else {
        pl.stage_eb_compress(s);
        if (!is_compress)
            pl.stage_eb_decompress(s);
    }
    if (!is_compress) {
        pl.stage_gs_s1(s);
        pl.stage_denorm(s);
    }
}

// ---------------------------------------------------------------------------
// Helper: one S1 timed iteration (uses stage_ga_s1 / stage_gs_s1)
// ---------------------------------------------------------------------------
static void run_one_s1(BenchPipeline& pl, PatchState& s,
                       StageTimer& timer, bool is_shyp, bool is_compress)
{
    if (is_shyp) {
        if (is_compress) {
            timer.mark("normalize");     pl.stage_normalize(s);
            timer.mark("g_a");           pl.stage_ga_s1(s);
            timer.mark("h_a");           pl.stage_ha(s);
            timer.mark("eb_compress");   pl.stage_eb_compress(s);
            timer.mark("eb_decompress"); pl.stage_eb_decompress(s);
            timer.mark("h_s");           pl.stage_hs(s);
            timer.mark("gc_compress");   pl.stage_gc_compress(s);
            timer.mark("_end");
        } else {
            timer.mark("normalize");     pl.stage_normalize(s);
            timer.mark("g_a");           pl.stage_ga_s1(s);
            timer.mark("h_a");           pl.stage_ha(s);
            timer.mark("eb_compress");   pl.stage_eb_compress(s);
            timer.mark("eb_decompress"); pl.stage_eb_decompress(s);
            timer.mark("h_s");           pl.stage_hs(s);
            timer.mark("gc_compress");   pl.stage_gc_compress(s);
            timer.mark("gc_decompress"); pl.stage_gc_decompress(s);
            timer.mark("g_s");           pl.stage_gs_s1(s);
            timer.mark("denorm");        pl.stage_denorm(s);
            timer.mark("_end");
        }
    } else {
        if (is_compress) {
            timer.mark("normalize");   pl.stage_normalize(s);
            timer.mark("g_a");         pl.stage_ga_s1(s);
            timer.mark("eb_compress"); pl.stage_eb_compress(s);
            timer.mark("_end");
        } else {
            timer.mark("normalize");     pl.stage_normalize(s);
            timer.mark("g_a");           pl.stage_ga_s1(s);
            timer.mark("eb_compress");   pl.stage_eb_compress(s);
            timer.mark("eb_decompress"); pl.stage_eb_decompress(s);
            timer.mark("g_s");           pl.stage_gs_s1(s);
            timer.mark("denorm");        pl.stage_denorm(s);
            timer.mark("_end");
        }
    }
    timer.commit();
}

// ---------------------------------------------------------------------------
// Helper: entropy-only stages without timing (warmup path)
// ---------------------------------------------------------------------------
static void run_entropy_notimed(BenchPipeline& pl, PatchState& s,
                                bool is_shyp, bool is_compress)
{
    if (is_shyp) {
        pl.stage_eb_compress(s);
        if (is_compress) {
            pl.stage_gc_compress(s);
        } else {
            pl.stage_eb_decompress(s);
            pl.stage_gc_compress(s);
            pl.stage_gc_decompress(s);
        }
    } else {
        pl.stage_eb_compress(s);
        if (!is_compress) pl.stage_eb_decompress(s);
    }
}

// ---------------------------------------------------------------------------
// Helper: entropy-only stages with timing (timed loop)
// ---------------------------------------------------------------------------
static void run_entropy_timed(BenchPipeline& pl, PatchState& s,
                               StageTimer& timer, bool is_shyp, bool is_compress)
{
    if (is_shyp) {
        if (is_compress) {
            timer.mark("eb_compress");  pl.stage_eb_compress(s);
            timer.mark("gc_compress");  pl.stage_gc_compress(s);
            timer.mark("_end");
        } else {
            timer.mark("eb_compress");   pl.stage_eb_compress(s);
            timer.mark("eb_decompress"); pl.stage_eb_decompress(s);
            timer.mark("gc_compress");   pl.stage_gc_compress(s);
            timer.mark("gc_decompress"); pl.stage_gc_decompress(s);
            timer.mark("_end");
        }
    } else {
        if (is_compress) {
            timer.mark("eb_compress"); pl.stage_eb_compress(s);
            timer.mark("_end");
        } else {
            timer.mark("eb_compress");   pl.stage_eb_compress(s);
            timer.mark("eb_decompress"); pl.stage_eb_decompress(s);
            timer.mark("_end");
        }
    }
    timer.commit();
}

// ---------------------------------------------------------------------------
// run_s0
// ---------------------------------------------------------------------------
BenchResult run_s0(BenchPipeline& pipeline, const RunConfig& cfg)
{
    auto pd = load_patches(cfg);
    const int H = pd.H, W = pd.W, P = pd.P;

    PatchState state   = pipeline.make_patch_state(H, W);
    const bool is_shyp = pipeline.uses_hyper();
    const bool is_cmp  = (cfg.scenario == Scenario::compress);

    // Warmup — fills pipeline caches / JIT state; not timed
    for (int w = 0; w < cfg.warmup; ++w) {
        std::memcpy(state.noisy_hwc.data(), pd.patches[w % P].data(),
                    static_cast<size_t>(H * W * 2) * sizeof(float));
        run_one_notimed(pipeline, state, is_shyp, is_cmp);
    }

    // Timed loop
    StageTimer timer;
    std::vector<int> bytes_record;
    bytes_record.reserve(static_cast<size_t>(cfg.iters));
    auto t_start = std::chrono::steady_clock::now();

    for (int it = 0; it < cfg.iters; ++it) {
        // Copy patch before timing starts — benchmark overhead; not pipeline compute.
        std::memcpy(state.noisy_hwc.data(), pd.patches[it % P].data(),
                    static_cast<size_t>(H * W * 2) * sizeof(float));
        run_one(pipeline, state, timer, is_shyp, is_cmp);
        bytes_record.push_back(state.num_bytes);
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
    r.bytes_per_iter         = std::move(bytes_record);
    return r;
}

// ---------------------------------------------------------------------------
// run_s1 — S1 channel-parallel (init_s1() must be called on pipeline first)
// ---------------------------------------------------------------------------
BenchResult run_s1(BenchPipeline& pipeline, const RunConfig& cfg)
{
    if (!pipeline.has_s1())
        throw std::runtime_error("run_s1: call pipeline.init_s1() before run_s1()");

    auto pd = load_patches(cfg);
    const int H = pd.H, W = pd.W, P = pd.P;

    PatchState state   = pipeline.make_patch_state(H, W);
    const bool is_shyp = pipeline.uses_hyper();
    const bool is_cmp  = (cfg.scenario == Scenario::compress);

    // Warmup uses the same concurrent path as the timed loop (stage_ga_s1 /
    // stage_gs_s1) so thread launch and VART concurrent-runner paths are
    // exercised before the timing window opens.
    for (int w = 0; w < cfg.warmup; ++w) {
        std::memcpy(state.noisy_hwc.data(), pd.patches[w % P].data(),
                    static_cast<size_t>(H * W * 2) * sizeof(float));
        run_one_s1_notimed(pipeline, state, is_shyp, is_cmp);
    }

    StageTimer timer;
    std::vector<int> bytes_record;
    bytes_record.reserve(static_cast<size_t>(cfg.iters));
    auto t_start = std::chrono::steady_clock::now();

    for (int it = 0; it < cfg.iters; ++it) {
        std::memcpy(state.noisy_hwc.data(), pd.patches[it % P].data(),
                    static_cast<size_t>(H * W * 2) * sizeof(float));
        run_one_s1(pipeline, state, timer, is_shyp, is_cmp);
        bytes_record.push_back(state.num_bytes);
    }

    auto t_end    = std::chrono::steady_clock::now();
    double wall_s = std::chrono::duration<double>(t_end - t_start).count();

    BenchResult r;
    r.stage_stats           = timer.summary();
    r.total_latency_mean_ms = timer.total_mean_s() * 1e3;
    r.throughput_fps        = static_cast<double>(cfg.iters) / wall_s;
    r.wall_time_s           = wall_s;
    r.num_iters             = cfg.iters;
    r.is_shyp               = is_shyp;
    r.bytes_per_iter        = std::move(bytes_record);
    return r;
}

// ---------------------------------------------------------------------------
// run_nn_only — DPU data-parallel ceiling (cfg.dpu_cores independent pipelines)
// ---------------------------------------------------------------------------
BenchResult run_nn_only(const RunConfig& cfg)
{
    auto pd = load_patches(cfg);
    const int H = pd.H, W = pd.W, P = pd.P;
    const int N = std::max(1, cfg.dpu_cores);

    // Create N BenchPipeline instances — VART round-robin distributes runners
    // across cores as they are created, so pipeline[i]'s g_a tends to land on
    // core (i % 3) when N <= 3.
    std::vector<std::unique_ptr<BenchPipeline>> pipelines;
    pipelines.reserve(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i)
        pipelines.push_back(std::make_unique<BenchPipeline>(cfg.xmodel_path, cfg.params_path));

    const bool is_shyp = pipelines[0]->uses_hyper();
    const bool is_cmp  = (cfg.scenario == Scenario::compress);

    std::vector<PatchState> states;
    states.reserve(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i)
        states.push_back(pipelines[i]->make_patch_state(H, W));

    // Warmup each pipeline independently (sequential to avoid VART contention)
    for (int i = 0; i < N; ++i) {
        for (int w = 0; w < cfg.warmup; ++w) {
            std::memcpy(states[i].noisy_hwc.data(),
                        pd.patches[(w + i * cfg.warmup) % P].data(),
                        static_cast<size_t>(H * W * 2) * sizeof(float));
            run_one_notimed(*pipelines[i], states[i], is_shyp, is_cmp);
        }
    }

    // Launch N threads — each runs cfg.iters patches on its own pipeline
    std::vector<StageTimer> timers(static_cast<size_t>(N));
    auto t_start = std::chrono::steady_clock::now();
    {
        std::vector<std::thread> threads;
        threads.reserve(static_cast<size_t>(N));
        for (int i = 0; i < N; ++i) {
            threads.emplace_back([&, i]() {
                for (int it = 0; it < cfg.iters; ++it) {
                    std::memcpy(states[i].noisy_hwc.data(),
                                pd.patches[it % P].data(),
                                static_cast<size_t>(H * W * 2) * sizeof(float));
                    run_one(*pipelines[i], states[i], timers[i], is_shyp, is_cmp);
                }
            });
        }
        for (auto& t : threads) t.join();
    }
    auto t_end    = std::chrono::steady_clock::now();
    double wall_s = std::chrono::duration<double>(t_end - t_start).count();

    // Thread 0's per-stage latency as representative; throughput = N × iters / wall
    BenchResult r;
    r.stage_stats           = timers[0].summary();
    r.total_latency_mean_ms = timers[0].total_mean_s() * 1e3;
    r.throughput_fps        = static_cast<double>(N * cfg.iters) / wall_s;
    r.wall_time_s           = wall_s;
    r.num_iters             = N * cfg.iters;
    r.is_shyp               = is_shyp;
    return r;
}

// ---------------------------------------------------------------------------
// run_entropy_only — CPU entropy ceiling (cfg.entropy_threads workers)
// ---------------------------------------------------------------------------
BenchResult run_entropy_only(const RunConfig& cfg)
{
    auto pd = load_patches(cfg);
    const int H = pd.H, W = pd.W, P = pd.P;
    const int N = std::max(1, cfg.entropy_threads);

    // Each thread gets its own BenchPipeline (mutable entropy model state).
    // DPU is used only in the warmup to populate intermediate buffers
    // (z, y, scales) that entropy stages read from.
    std::vector<std::unique_ptr<BenchPipeline>> pipelines;
    pipelines.reserve(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i)
        pipelines.push_back(std::make_unique<BenchPipeline>(cfg.xmodel_path, cfg.params_path));

    const bool is_shyp = pipelines[0]->uses_hyper();
    const bool is_cmp  = (cfg.scenario == Scenario::compress);

    // Init state: one full pipeline run per thread (sequential) to populate
    // z, y, scales used by entropy stages in the timed loop.
    std::vector<PatchState> states;
    states.reserve(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) {
        states.push_back(pipelines[i]->make_patch_state(H, W));
        std::memcpy(states[i].noisy_hwc.data(),
                    pd.patches[i % P].data(),
                    static_cast<size_t>(H * W * 2) * sizeof(float));
        run_one_notimed(*pipelines[i], states[i], is_shyp, is_cmp);
    }

    // Entropy-only warmup (sequential — DPU not called here)
    for (int i = 0; i < N; ++i)
        for (int w = 0; w < cfg.warmup; ++w)
            run_entropy_notimed(*pipelines[i], states[i], is_shyp, is_cmp);

    // Launch N threads — each runs entropy stages only on fixed pre-computed state
    std::vector<StageTimer> timers(static_cast<size_t>(N));
    auto t_start = std::chrono::steady_clock::now();
    {
        std::vector<std::thread> threads;
        threads.reserve(static_cast<size_t>(N));
        for (int i = 0; i < N; ++i) {
            threads.emplace_back([&, i]() {
                for (int it = 0; it < cfg.iters; ++it)
                    run_entropy_timed(*pipelines[i], states[i], timers[i], is_shyp, is_cmp);
            });
        }
        for (auto& t : threads) t.join();
    }
    auto t_end    = std::chrono::steady_clock::now();
    double wall_s = std::chrono::duration<double>(t_end - t_start).count();

    BenchResult r;
    r.stage_stats           = timers[0].summary();
    r.total_latency_mean_ms = timers[0].total_mean_s() * 1e3;
    r.throughput_fps        = static_cast<double>(N * cfg.iters) / wall_s;
    r.wall_time_s           = wall_s;
    r.num_iters             = N * cfg.iters;
    r.is_shyp               = is_shyp;
    return r;
}

} // namespace ddc
