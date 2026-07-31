/* bench_pipeline.cpp — BenchPipeline stage implementations.
 *
 * The math in each stage is taken verbatim from InferencePipeline::_run_shyp /
 * _run_fp in inference_runner.cpp.  The only change is that intermediate results
 * live in PatchState instead of local variables, so callers can time individual
 * stages and (in future milestones) pass state across thread boundaries.
 *
 * Board re-verify: after first on-board build, run inference_hybrid on
 * test_sub500_seed42.npy (100 patches) and confirm results are unchanged
 * before building pipelined configs on top of these stages.
 */

#include "bench_pipeline.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>

#include "constants.hpp"
#include "patch_transforms.hpp"

#ifdef HAVE_DPU
#include "dpu_runners.hpp"
#include "logger.hpp"
#endif

namespace ddc {

// ---------------------------------------------------------------------------
// Private helpers — pure transforms on PatchState fields, shared between the
// sequential and S1-concurrent stage variants.
// ---------------------------------------------------------------------------

static void channel_split(PatchState& s)
{
    for (int i = 0; i < s.H * s.W; ++i) {
        s.real_ch[i] = s.norm_hwc[i * 2 + 0];
        s.imag_ch[i] = s.norm_hwc[i * 2 + 1];
    }
}

static void interleave_and_abs(PatchState& s)
{
    for (int hw = 0; hw < s.yh * s.yw; ++hw) {
        for (int c = 0; c < s.yc; ++c) {
            s.y[hw * 2 * s.yc + c]        = s.y_real[hw * s.yc + c];
            s.y[hw * 2 * s.yc + s.yc + c] = s.y_imag[hw * s.yc + c];
        }
    }
    const int y_total = s.yh * s.yw * s.yc * 2;
    for (int i = 0; i < y_total; ++i)
        s.y_abs[i] = std::abs(s.y[i]);
}

static void deinterleave_yhat(PatchState& s)
{
    for (int hw = 0; hw < s.yh * s.yw; ++hw) {
        for (int c = 0; c < s.yc; ++c) {
            s.yh_real[hw * s.yc + c] = s.y_hat[hw * 2 * s.yc + c];
            s.yh_imag[hw * s.yc + c] = s.y_hat[hw * 2 * s.yc + s.yc + c];
        }
    }
}

static void pack_recon(PatchState& s)
{
    for (int i = 0; i < s.H * s.W; ++i) {
        s.recon_norm_logI[i * 2 + 0] = s.recon_real[i];
        s.recon_norm_logI[i * 2 + 1] = s.recon_imag[i];
    }
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
BenchPipeline::BenchPipeline(const std::filesystem::path& xmodel_path,
                               const std::filesystem::path& params_dir)
{
    eb_.load_params(params_dir);

    auto gc_scale = params_dir / "gc_scale_table.npy";
    if (std::filesystem::exists(gc_scale)) {
        gc_.load_params(params_dir);
        has_gc_ = gc_.is_loaded();
    }

#ifdef HAVE_DPU
    auto meta_json = xmodel_path.parent_path() / "meta.json";
    loader_.load(xmodel_path.string(), meta_json.string());
#else
    (void)xmodel_path;
#endif
}

// ---------------------------------------------------------------------------
// make_patch_state
// ---------------------------------------------------------------------------
PatchState BenchPipeline::make_patch_state(int H, int W) const
{
    PatchState s;
    s.H = H;
    s.W = W;

    s.noisy_hwc.resize(static_cast<size_t>(H * W * 2));
    s.norm_hwc .resize(static_cast<size_t>(H * W * 2));
    s.real_ch  .resize(static_cast<size_t>(H * W));
    s.imag_ch  .resize(static_cast<size_t>(H * W));

#ifdef HAVE_DPU
    const auto& ga        = loader_.runner("g_a");
    const auto& ga_shape  = ga.output_shape();   // [N, H', W', C]
    s.yh = ga_shape[1];
    s.yw = ga_shape[2];
    s.yc = ga_shape[3];

    int ga_out = ga.output_numel();
    int y_len  = s.yh * s.yw * s.yc;

    s.y_real.resize(static_cast<size_t>(ga_out));
    s.y_imag.resize(static_cast<size_t>(ga_out));
    s.y    .resize(static_cast<size_t>(y_len * 2));
    s.y_abs.resize(static_cast<size_t>(y_len * 2));

    if (has_gc_) {
        const auto& ha       = loader_.runner("h_a");
        const auto& ha_shape = ha.output_shape();
        s.zh = ha_shape[1];
        s.zw = ha_shape[2];
        s.z    .resize(static_cast<size_t>(ha.output_numel()));
        s.z_hat.resize(static_cast<size_t>(ha.output_numel()));

        const auto& hs = loader_.runner("h_s");
        s.scales.resize(static_cast<size_t>(hs.output_numel()));
        s.y_hat .resize(static_cast<size_t>(y_len * 2));
        s.means .assign(static_cast<size_t>(y_len * 2), 0.0f);
    } else {
        // FP: EB decompresses directly into y_hat (same size as y)
        s.y_hat.resize(static_cast<size_t>(y_len * 2));
    }

    s.yh_real.resize(static_cast<size_t>(y_len));
    s.yh_imag.resize(static_cast<size_t>(y_len));

    const auto& gs = loader_.runner("g_s");
    s.recon_real .resize(static_cast<size_t>(gs.output_numel()));
    s.recon_imag .resize(static_cast<size_t>(gs.output_numel()));
#else
    // Fallback sizing from constants so host builds compile.
    // Stages will throw at runtime if called without HAVE_DPU.
    s.yh = H_LATENT; s.yw = W_LATENT; s.yc = C_MAIN;
    int y_len = H_LATENT * W_LATENT * C_MAIN;
    s.y_real.resize(static_cast<size_t>(y_len));
    s.y_imag.resize(static_cast<size_t>(y_len));
    s.y    .resize(static_cast<size_t>(y_len * 2));
    s.y_abs.resize(static_cast<size_t>(y_len * 2));
    if (has_gc_) {
        s.zh = H_HYPER; s.zw = W_HYPER;
        int z_len = H_HYPER * W_HYPER * C_HYPER;
        s.z    .resize(static_cast<size_t>(z_len));
        s.z_hat.resize(static_cast<size_t>(z_len));
        s.scales.resize(static_cast<size_t>(y_len * 2));
        s.y_hat .resize(static_cast<size_t>(y_len * 2));
        s.means .assign(static_cast<size_t>(y_len * 2), 0.0f);
    } else {
        s.y_hat.resize(static_cast<size_t>(y_len * 2));
    }
    s.yh_real.resize(static_cast<size_t>(y_len));
    s.yh_imag.resize(static_cast<size_t>(y_len));
    s.recon_real .resize(static_cast<size_t>(H * W));
    s.recon_imag .resize(static_cast<size_t>(H * W));
#endif

    s.recon_norm_logI.resize(static_cast<size_t>(H * W * 2));
    s.recon_lina     .resize(static_cast<size_t>(H * W));
    return s;
}

// ---------------------------------------------------------------------------
// init_s1 — create duplicate g_a / g_s runners for channel-parallel S1
// ---------------------------------------------------------------------------
void BenchPipeline::init_s1()
{
    if (has_s1_)
        throw std::runtime_error("init_s1: already initialised");
#ifdef HAVE_DPU
    // VART assigns cores via global round-robin at runner creation time.
    // We need both concurrent pairs (g_a‖g_a_1, g_s‖g_s_1) on distinct cores.
    // The correct creation order depends on how many primary runners were loaded:
    //
    //   SHyp (4 primaries: g_a→0, h_a→1, g_s→2, h_s→0):
    //     next slot = core 1.  Create g_s_1 first (→1 ≠ g_s core 2 ✓),
    //     then g_a_1 (→2 ≠ g_a core 0 ✓).  Both pairs on distinct cores.
    //
    //   FP (2 primaries: g_a→0, g_s→1):
    //     next slot = core 2.  Create g_a_1 first (→2 ≠ g_a core 0 ✓),
    //     then g_s_1 (→0 ≠ g_s core 1 ✓).  Both pairs on distinct cores.
    //
    // WARNING: assumes 3 DPU cores (ZCU102 B4096×3).  On fewer cores the
    // round-robin wraps sooner and a collision may occur (see design doc §9-A).
    // To verify the physical core count on the board run: xdputil query
    if (has_gc_) {
        // SHyp: g_s_1 first → core 1, then g_a_1 → core 2
        runner_gs2_.emplace(loader_.create_duplicate_runner("g_s", "g_s_1"));
        runner_ga2_.emplace(loader_.create_duplicate_runner("g_a", "g_a_1"));
    } else {
        // FP: g_a_1 first → core 2, then g_s_1 → core 0
        runner_ga2_.emplace(loader_.create_duplicate_runner("g_a", "g_a_1"));
        runner_gs2_.emplace(loader_.create_duplicate_runner("g_s", "g_s_1"));
    }
    has_s1_ = true;
#else
    throw std::runtime_error("init_s1: compiled without HAVE_DPU");
#endif
}

// ---------------------------------------------------------------------------
// Stage 0 — CPU normalise
// ---------------------------------------------------------------------------
void BenchPipeline::stage_normalize(PatchState& s)
{
    if (use_neon_)
        normalize_patch_neon(s.noisy_hwc.data(), s.norm_hwc.data(), s.H, s.W);
    else
        normalize_patch(s.noisy_hwc.data(), s.norm_hwc.data(), s.H, s.W);
}

// ---------------------------------------------------------------------------
// Stage 1 — DPU g_a + CPU interleave
// ---------------------------------------------------------------------------
void BenchPipeline::stage_ga(PatchState& s)
{
    channel_split(s);
#ifdef HAVE_DPU
    const auto& ga = loader_.runner("g_a");
    ga.run(s.real_ch.data(), s.y_real.data());
    ga.run(s.imag_ch.data(), s.y_imag.data());
#else
    throw std::runtime_error("stage_ga: compiled without HAVE_DPU");
#endif
    interleave_and_abs(s);
}

// ---------------------------------------------------------------------------
// Stage 1 S1 — concurrent g_a(real) ‖ g_a(imag)
// ---------------------------------------------------------------------------
void BenchPipeline::stage_ga_s1(PatchState& s)
{
    channel_split(s);
#ifdef HAVE_DPU
    if (!runner_ga2_)
        throw std::runtime_error("stage_ga_s1: call init_s1() first");
    const auto& ga  = loader_.runner("g_a");
    const auto& ga2 = *runner_ga2_;
    std::thread t([&]{ ga2.run(s.imag_ch.data(), s.y_imag.data()); });
    ga.run(s.real_ch.data(), s.y_real.data());
    t.join();
#else
    throw std::runtime_error("stage_ga_s1: compiled without HAVE_DPU");
#endif
    interleave_and_abs(s);
}

// ---------------------------------------------------------------------------
// Stage 2 — DPU h_a  (SHyp only)
// ---------------------------------------------------------------------------
void BenchPipeline::stage_ha(PatchState& s)
{
#ifdef HAVE_DPU
    const auto& ha = loader_.runner("h_a");
    ha.run(s.y_abs.data(), s.z.data());
#else
    (void)s;
    throw std::runtime_error("stage_ha: compiled without HAVE_DPU");
#endif
}

// ---------------------------------------------------------------------------
// Stage 3a — CPU EB compress
//   SHyp: z → z_bits / z_bytes
//   FP:   y → y_bits / y_bytes / num_bytes
// ---------------------------------------------------------------------------
void BenchPipeline::stage_eb_compress(PatchState& s)
{
    if (has_gc_) {
        s.z_bits  = eb_.compress(s.z.data(), s.zh, s.zw);
        s.z_bytes = static_cast<int>(s.z_bits.size());
    } else {
        s.y_bits    = eb_.compress(s.y.data(), s.yh, s.yw);
        s.y_bytes   = static_cast<int>(s.y_bits.size());
        s.num_bytes = s.y_bytes;
    }
}

// ---------------------------------------------------------------------------
// Stage 3b — CPU EB decompress
//   SHyp: z_bits → z_hat
//   FP:   y_bits → y_hat
// ---------------------------------------------------------------------------
void BenchPipeline::stage_eb_decompress(PatchState& s)
{
    if (has_gc_) {
        s.z_hat = eb_.decompress(s.z_bits, s.zh, s.zw);
    } else {
        s.y_hat = eb_.decompress(s.y_bits, s.yh, s.yw);
    }
}

// ---------------------------------------------------------------------------
// Stage 4 — DPU h_s  (SHyp only)
// ---------------------------------------------------------------------------
void BenchPipeline::stage_hs(PatchState& s)
{
#ifdef HAVE_DPU
    const auto& hs = loader_.runner("h_s");
    hs.run(s.z_hat.data(), s.scales.data());
#else
    (void)s;
    throw std::runtime_error("stage_hs: compiled without HAVE_DPU");
#endif
}

// ---------------------------------------------------------------------------
// Stage 5a — CPU GC compress  (SHyp only)
// ---------------------------------------------------------------------------
void BenchPipeline::stage_gc_compress(PatchState& s)
{
    // means is pre-zeroed in make_patch_state; stays zero across iterations
    s.y_bits    = gc_.compress(s.y.data(), s.scales.data(), s.means.data(),
                               s.yh, s.yw, C_MAIN * 2);
    s.y_bytes   = static_cast<int>(s.y_bits.size());
    s.num_bytes = s.z_bytes + s.y_bytes;
}

// ---------------------------------------------------------------------------
// Stage 5b — CPU GC decompress  (SHyp only)
// ---------------------------------------------------------------------------
void BenchPipeline::stage_gc_decompress(PatchState& s)
{
    s.y_hat = gc_.decompress(s.y_bits, s.scales.data(), s.means.data(),
                              s.yh, s.yw, C_MAIN * 2);
}

// ---------------------------------------------------------------------------
// Stage 6 — CPU de-interleave + DPU g_s + CPU pack
// ---------------------------------------------------------------------------
void BenchPipeline::stage_gs(PatchState& s)
{
    deinterleave_yhat(s);
#ifdef HAVE_DPU
    const auto& gs = loader_.runner("g_s");
    gs.run(s.yh_real.data(), s.recon_real.data());
    gs.run(s.yh_imag.data(), s.recon_imag.data());
#else
    throw std::runtime_error("stage_gs: compiled without HAVE_DPU");
#endif
    pack_recon(s);
}

// ---------------------------------------------------------------------------
// Stage 6 S1 — concurrent g_s(real) ‖ g_s(imag)
// ---------------------------------------------------------------------------
void BenchPipeline::stage_gs_s1(PatchState& s)
{
    deinterleave_yhat(s);
#ifdef HAVE_DPU
    if (!runner_gs2_)
        throw std::runtime_error("stage_gs_s1: call init_s1() first");
    const auto& gs  = loader_.runner("g_s");
    const auto& gs2 = *runner_gs2_;
    std::thread t([&]{ gs2.run(s.yh_imag.data(), s.recon_imag.data()); });
    gs.run(s.yh_real.data(), s.recon_real.data());
    t.join();
#else
    throw std::runtime_error("stage_gs_s1: compiled without HAVE_DPU");
#endif
    pack_recon(s);
}

// ---------------------------------------------------------------------------
// Stage 7 — CPU denorm  (full / decompress scenarios)
// ---------------------------------------------------------------------------
void BenchPipeline::stage_denorm(PatchState& s)
{
    if (use_neon_)
        denorm_to_lina_neon(s.recon_norm_logI.data(), s.recon_lina.data(), s.H, s.W);
    else
        denorm_to_lina(s.recon_norm_logI.data(), s.recon_lina.data(), s.H, s.W);
}

} // namespace ddc
