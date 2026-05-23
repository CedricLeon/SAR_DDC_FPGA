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

#include "constants.hpp"
#include "patch_transforms.hpp"

#ifdef HAVE_DPU
#include "dpu_runners.hpp"
#include "logger.hpp"
#endif

namespace ddc {

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
// Stage 0 — CPU normalise
// ---------------------------------------------------------------------------
void BenchPipeline::stage_normalize(PatchState& s)
{
    normalize_patch(s.noisy_hwc.data(), s.norm_hwc.data(), s.H, s.W);
}

// ---------------------------------------------------------------------------
// Stage 1 — DPU g_a + CPU interleave
// ---------------------------------------------------------------------------
void BenchPipeline::stage_ga(PatchState& s)
{
    // Split norm_hwc [H*W*2] → real_ch [H*W] + imag_ch [H*W]
    for (int i = 0; i < s.H * s.W; ++i) {
        s.real_ch[i] = s.norm_hwc[i * 2 + 0];
        s.imag_ch[i] = s.norm_hwc[i * 2 + 1];
    }

#ifdef HAVE_DPU
    const auto& ga = loader_.runner("g_a");
    ga.run(s.real_ch.data(), s.y_real.data());
    ga.run(s.imag_ch.data(), s.y_imag.data());
#else
    throw std::runtime_error("stage_ga: compiled without HAVE_DPU");
#endif

    // Interleave: y[hw][c_real..., c_imag...] — each spatial pos gets
    // the real channel block followed by the imag channel block.
    // Identical to np.concatenate((y_real, y_imag), axis=-1) in Python.
    for (int hw = 0; hw < s.yh * s.yw; ++hw) {
        for (int c = 0; c < s.yc; ++c) {
            s.y[hw * 2 * s.yc + c]        = s.y_real[hw * s.yc + c];
            s.y[hw * 2 * s.yc + s.yc + c] = s.y_imag[hw * s.yc + c];
        }
    }

    // |y| for h_a (always computed; zero cost for FP since h_a is skipped)
    int y_total = s.yh * s.yw * s.yc * 2;
    for (int i = 0; i < y_total; ++i)
        s.y_abs[i] = std::abs(s.y[i]);
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
// Stage 3 — CPU entropy bottleneck
//   SHyp: compress+decompress z  → z_hat / z_bytes
//   FP:   compress+decompress y  → y_hat / y_bytes / num_bytes
// ---------------------------------------------------------------------------
void BenchPipeline::stage_eb(PatchState& s)
{
    if (has_gc_) {
        // ScaleHyperprior: EB on the hyper latent z
        std::vector<uint8_t> z_bits = eb_.compress(s.z.data(), s.zh, s.zw);
        s.z_hat   = eb_.decompress(z_bits, s.zh, s.zw);
        s.z_bytes = static_cast<int>(z_bits.size());
    } else {
        // FactorizedPrior: EB directly on the main latent y
        std::vector<uint8_t> y_bits = eb_.compress(s.y.data(), s.yh, s.yw);
        s.y_hat   = eb_.decompress(y_bits, s.yh, s.yw);
        s.y_bytes = static_cast<int>(y_bits.size());
        s.num_bytes = s.y_bytes;
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
// Stage 5 — CPU Gaussian conditional  (SHyp only)
// ---------------------------------------------------------------------------
void BenchPipeline::stage_gc(PatchState& s)
{
    // means is pre-zeroed in make_patch_state; stays zero across iterations
    std::vector<uint8_t> y_bits = gc_.compress(
        s.y.data(), s.scales.data(), s.means.data(),
        s.yh, s.yw, C_MAIN * 2);
    s.y_hat = gc_.decompress(
        y_bits, s.scales.data(), s.means.data(),
        s.yh, s.yw, C_MAIN * 2);
    s.y_bytes   = static_cast<int>(y_bits.size());
    s.num_bytes = s.z_bytes + s.y_bytes;
}

// ---------------------------------------------------------------------------
// Stage 6 — CPU de-interleave + DPU g_s + CPU pack
// ---------------------------------------------------------------------------
void BenchPipeline::stage_gs(PatchState& s)
{
    // Reverse the interleave from stage_ga
    for (int hw = 0; hw < s.yh * s.yw; ++hw) {
        for (int c = 0; c < s.yc; ++c) {
            s.yh_real[hw * s.yc + c] = s.y_hat[hw * 2 * s.yc + c];
            s.yh_imag[hw * s.yc + c] = s.y_hat[hw * 2 * s.yc + s.yc + c];
        }
    }

#ifdef HAVE_DPU
    const auto& gs = loader_.runner("g_s");
    gs.run(s.yh_real.data(), s.recon_real.data());
    gs.run(s.yh_imag.data(), s.recon_imag.data());
#else
    (void)s;
    throw std::runtime_error("stage_gs: compiled without HAVE_DPU");
#endif

    // Pack into HW2
    for (int i = 0; i < s.H * s.W; ++i) {
        s.recon_norm_logI[i * 2 + 0] = s.recon_real[i];
        s.recon_norm_logI[i * 2 + 1] = s.recon_imag[i];
    }
}

// ---------------------------------------------------------------------------
// Stage 7 — CPU denorm  (full / decompress scenarios)
// ---------------------------------------------------------------------------
void BenchPipeline::stage_denorm(PatchState& s)
{
    denorm_to_lina(s.recon_norm_logI.data(), s.recon_lina.data(), s.H, s.W);
}

} // namespace ddc
