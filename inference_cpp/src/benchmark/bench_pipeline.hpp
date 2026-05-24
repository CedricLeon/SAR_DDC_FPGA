#pragma once
// bench_pipeline.hpp — stage-decomposed compress/decompress pipeline.
//
// BenchPipeline loads the xmodel and entropy models once, then exposes each
// pipeline step as a void method on a PatchState.  Callers (bench_configs)
// call stages in the desired order and instrument them with StageTimer.
//
// PatchState holds all intermediate buffers for one patch.  For S0 a single
// state is reused across iterations (avoiding per-iteration allocation).
// For pipelined configs (P0, P2) a small pool of states moves through queues.
//
// Thread safety: stages read from / write to a PatchState exclusively.
// Two threads calling stages on *different* PatchState objects is safe.
// Two threads calling stages on the *same* PatchState is a data race — don't.
// DPU runner safety under concurrency is addressed in the P0/P2 milestone.

#include <filesystem>
#include <optional>
#include <vector>

#include "entropy_models.hpp"

#ifdef HAVE_DPU
#include "dpu_runners.hpp"
#endif

namespace ddc {

// ---------------------------------------------------------------------------
// PatchState — all intermediate buffers for one 256×256 patch
// ---------------------------------------------------------------------------
struct PatchState {
    // Patch spatial dimensions (set by BenchPipeline::make_patch_state)
    int H = 0, W = 0;

    // Input (caller fills noisy_hwc before calling stage_normalize)
    std::vector<float> noisy_hwc;        // [H*W*2] raw complex, interleaved re/im

    // After stage_normalize
    std::vector<float> norm_hwc;         // [H*W*2]

    // stage_ga: split norm_hwc, run g_a on each channel, interleave
    std::vector<float> real_ch;          // [H*W]
    std::vector<float> imag_ch;          // [H*W]
    std::vector<float> y_real;           // [ga_out_numel]
    std::vector<float> y_imag;           // [ga_out_numel]
    int yh = 0, yw = 0, yc = 0;         // g_a output spatial dims and channels
    std::vector<float> y;                // [yh*yw*yc*2] interleaved NHWC
    std::vector<float> y_abs;            // |y|  (for h_a input in SHyp path)

    // stage_ha (SHyp only)
    std::vector<float> z;                // [zh*zw*C_HYPER]
    int zh = 0, zw = 0;

    // stage_eb_compress / stage_eb_decompress
    std::vector<uint8_t> z_bits;         // EB-compressed z (SHyp)
    std::vector<float>   z_hat;          // [zh*zw*C_HYPER] (SHyp EB output)
    int z_bytes = 0;

    // stage_hs (SHyp only): h_s(z_hat) → scales
    std::vector<float> scales;           // [yh*yw*yc*2]

    // stage_gc_compress / stage_eb_compress (FP): compressed main latent + decompressed y
    std::vector<uint8_t> y_bits;         // GC-compressed y (SHyp) or EB-compressed y (FP)
    std::vector<float>   y_hat;          // [yh*yw*yc*2]
    std::vector<float>   means;          // [yh*yw*yc*2], pre-zeroed (avoids per-call alloc)
    int y_bytes = 0;

    // Summary (set by the last entropy stage)
    int num_bytes = 0;

    // stage_gs: de-interleave y_hat, run g_s, pack result
    std::vector<float> yh_real;          // [yh*yw*yc]
    std::vector<float> yh_imag;          // [yh*yw*yc]
    std::vector<float> recon_real;       // [gs_out_numel]
    std::vector<float> recon_imag;       // [gs_out_numel]

    // After stage_gs: normalised log-intensity [H*W*2]
    std::vector<float> recon_norm_logI;

    // After stage_denorm: linear amplitude [H*W]
    std::vector<float> recon_lina;
};

// ---------------------------------------------------------------------------
// BenchPipeline
// ---------------------------------------------------------------------------
class BenchPipeline {
public:
    // Load entropy models and (if HAVE_DPU) the xmodel.
    // Throws std::runtime_error on failure.
    explicit BenchPipeline(const std::filesystem::path& xmodel_path,
                           const std::filesystem::path& params_dir);

    // True = ScaleHyperprior path (g_a→h_a→EB→h_s→GC→g_s).
    // False = FactorizedPrior path (g_a→EB→g_s).
    bool uses_hyper() const { return has_gc_; }

    // S1 channel-parallel: creates duplicate g_a_1 and g_s_1 runners so that
    // stage_ga_s1 / stage_gs_s1 can dispatch real and imag concurrently.
    // Must be called before any _s1 stage method.
    // Throws on host builds (HAVE_DPU=OFF) or if init_s1() already called.
    void init_s1();
    bool has_s1() const { return has_s1_; }

    // Allocate a PatchState with all buffers pre-sized for this model's output
    // shapes and H×W input.  Call once per worker; reuse across iterations.
    PatchState make_patch_state(int H, int W) const;

    // ------------------------------------------------------------------
    // Pipeline stages — call in the order appropriate for the scenario.
    // Each reads fields populated by the previous stage and writes its
    // own output fields.  Stages are not const because DPU runners mutate
    // internal VART state on execute_async/wait.
    // ------------------------------------------------------------------

    // Stage 0 (CPU): log-amplitude normalise  noisy_hwc → norm_hwc
    void stage_normalize(PatchState& s);

    // Stage 1 (DPU + CPU): split norm_hwc → real/imag, run g_a on each,
    //   interleave y, compute |y| for h_a.
    //   Throws if compiled without HAVE_DPU.
    void stage_ga(PatchState& s);

    // Stage 1 S1 variant: same as stage_ga but dispatches g_a(real) and
    //   g_a(imag) on two separate DPU runners concurrently (one std::thread).
    //   Requires init_s1() called first.
    void stage_ga_s1(PatchState& s);

    // Stage 2 (DPU, SHyp only): h_a(|y|) → z
    void stage_ha(PatchState& s);

    // Stage 3a (CPU): EB compress.
    //   SHyp — z → z_bits, sets z_bytes.
    //   FP   — y → y_bits, sets y_bytes / num_bytes.
    void stage_eb_compress(PatchState& s);

    // Stage 3b (CPU): EB decompress.
    //   SHyp — z_bits → z_hat.
    //   FP   — y_bits → y_hat.
    void stage_eb_decompress(PatchState& s);

    // Stage 4 (DPU, SHyp only): h_s(z_hat) → scales
    void stage_hs(PatchState& s);

    // Stage 5a (CPU, SHyp only): GC compress y → y_bits, sets y_bytes / num_bytes.
    void stage_gc_compress(PatchState& s);

    // Stage 5b (CPU, SHyp only): GC decompress y_bits → y_hat.
    void stage_gc_decompress(PatchState& s);

    // Stage 6 (DPU + CPU): de-interleave y_hat, run g_s, pack recon_norm_logI.
    void stage_gs(PatchState& s);

    // Stage 6 S1 variant: concurrent g_s(real) and g_s(imag).
    // Requires init_s1() called first.
    void stage_gs_s1(PatchState& s);

    // Stage 7 (CPU): denorm recon_norm_logI → recon_lina  (linA, float32)
    void stage_denorm(PatchState& s);

private:
#ifdef HAVE_DPU
    XModelLoader loader_;
    std::optional<DPUSubgraphRunner> runner_ga2_;  // S1 duplicate for g_a(imag)
    std::optional<DPUSubgraphRunner> runner_gs2_;  // S1 duplicate for g_s(imag)
#endif
    EntropyBottleneck   eb_;
    GaussianConditional gc_;
    bool has_gc_  = false;
    bool has_s1_  = false;
};

} // namespace ddc
