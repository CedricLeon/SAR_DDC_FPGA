#pragma once
// patch_transforms.hpp — pure per-patch transforms shared by inference_runner
// and bench_pipeline.  Header-only, no deps beyond <cmath> and constants.hpp.
//
// Extracted from the anonymous namespace in inference_runner.cpp so that
// bench_pipeline.cpp can include them without duplicating the math.

#include <algorithm>
#include <cmath>

#include "constants.hpp"
#include "neon_mathfun.h"  // vectorised log_ps/exp_ps (ARM only; empty on other hosts)

namespace ddc {

// Normalise raw complex HW2 patch -> normalised log-intensity HW2.
//   norm = (log(amp^2 + EPS) - 2*AMP_MIN) / (2*AMP_MAX - 2*AMP_MIN)
// `in_hwc` and `out_hwc` are both H*W*2 interleaved (real, imag).
// [[maybe_unused]]: only called from HAVE_DPU-gated code in inference_runner;
// bench_pipeline always uses it regardless of HAVE_DPU.
[[maybe_unused]] inline void normalize_patch(const float* in_hwc, float* out_hwc, int H, int W)
{
    const float denom = 2.0f * (static_cast<float>(AMP_MAX) - static_cast<float>(AMP_MIN));
    for (int i = 0; i < H * W; ++i) {
        for (int c = 0; c < 2; ++c) {
            float amp  = in_hwc[i * 2 + c];
            float logI = std::log(amp * amp + static_cast<float>(EPS));
            out_hwc[i * 2 + c] = (logI - 2.0f * static_cast<float>(AMP_MIN)) / denom;
        }
    }
}

// Denormalise recon_norm_logI HW2 -> linear amplitude HW (single channel).
//   linI = 0.5 * (exp(logI_ch0)^2 + exp(logI_ch1)^2);  linA = sqrt(linI)
// Double intermediates match Python's float64 pathway (numpy promotes float32
// constants to float64; must keep identical to stay within the 0.1 dB gate).
inline void denorm_to_lina(const float* norm_hwc, float* out_hw, int H, int W)
{
    const double amp_range = AMP_MAX - AMP_MIN;
    const double amp_min   = AMP_MIN;
    for (int i = 0; i < H * W; ++i) {
        double logI0 = static_cast<double>(norm_hwc[i * 2 + 0]) * amp_range + amp_min;
        double logI1 = static_cast<double>(norm_hwc[i * 2 + 1]) * amp_range + amp_min;
        double linI0 = std::exp(logI0);
        double linI1 = std::exp(logI1);
        double linI  = 0.5 * (linI0 * linI0 + linI1 * linI1);
        out_hw[i]    = static_cast<float>(std::sqrt(linI));
    }
}

// NEON --neon variant of normalize_patch. Element-wise (interleaved re/im), 4 lanes at a time with
// a vectorised log; scalar tail. Falls back to the scalar path on non-NEON builds. The polynomial
// log differs from libm at ~1 ULP — well below the DPU's INT8 input step, so the .ddc is ~unchanged.
[[maybe_unused]] inline void normalize_patch_neon(const float* in_hwc, float* out_hwc, int H, int W)
{
#if defined(__ARM_NEON) || defined(__aarch64__)
    const int n = H * W * 2;
    const float eps = static_cast<float>(EPS);
    const float amin2 = 2.0f * static_cast<float>(AMP_MIN);
    const float inv_denom =
        1.0f / (2.0f * (static_cast<float>(AMP_MAX) - static_cast<float>(AMP_MIN)));
    const float32x4_t veps = vdupq_n_f32(eps);
    const float32x4_t vamin2 = vdupq_n_f32(amin2);
    const float32x4_t vinv = vdupq_n_f32(inv_denom);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t amp = vld1q_f32(in_hwc + i);
        float32x4_t logI = log_ps(vmlaq_f32(veps, amp, amp));  // log(amp*amp + eps)
        vst1q_f32(out_hwc + i, vmulq_f32(vsubq_f32(logI, vamin2), vinv));
    }
    for (; i < n; ++i) {
        float amp = in_hwc[i];
        out_hwc[i] = (std::log(amp * amp + eps) - amin2) * inv_denom;
    }
#else
    normalize_patch(in_hwc, out_hwc, H, W);
#endif
}

// NEON --neon variant of denorm_to_lina. Deinterleaves the two channels (vld2q), vectorised exp,
// float32 throughout (the scalar path uses double, so this is a deliberate precision drop measured
// in the scalar-vs-NEON comparison). Falls back to scalar on non-NEON builds. Decode-side only.
[[maybe_unused]] inline void denorm_to_lina_neon(const float* norm_hwc, float* out_hw, int H, int W)
{
#if defined(__ARM_NEON) || defined(__aarch64__)
    const int n = H * W;
    const float range = static_cast<float>(AMP_MAX - AMP_MIN);
    const float amin = static_cast<float>(AMP_MIN);
    const float32x4_t vrange = vdupq_n_f32(range);
    const float32x4_t vamin = vdupq_n_f32(amin);
    const float32x4_t vhalf = vdupq_n_f32(0.5f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4x2_t nn = vld2q_f32(norm_hwc + i * 2);  // val[0]=ch0, val[1]=ch1
        float32x4_t e0 = exp_ps(vmlaq_f32(vamin, nn.val[0], vrange));
        float32x4_t e1 = exp_ps(vmlaq_f32(vamin, nn.val[1], vrange));
        float32x4_t linI = vmulq_f32(vhalf, vaddq_f32(vmulq_f32(e0, e0), vmulq_f32(e1, e1)));
        vst1q_f32(out_hw + i, vsqrtq_f32(linI));
    }
    for (; i < n; ++i) {
        float linI0 = std::exp(norm_hwc[i * 2 + 0] * range + amin);
        float linI1 = std::exp(norm_hwc[i * 2 + 1] * range + amin);
        out_hw[i] = std::sqrt(0.5f * (linI0 * linI0 + linI1 * linI1));
    }
#else
    denorm_to_lina(norm_hwc, out_hw, H, W);
#endif
}

// Max relative error of the NEON log/exp kernels vs libm over the operating ranges (log: [1e-3,1e9]
// covering amp^2+EPS; exp: [-80,80]). 0.0 on non-NEON builds. A transcription typo in a Cephes
// constant shows up here as a large error — run it on the board before trusting the --neon pipeline.
[[maybe_unused]] inline double neon_math_maxrelerr()
{
#if defined(__ARM_NEON) || defined(__aarch64__)
    double maxerr = 0.0;
    for (double v = 1e-3; v <= 1e9; v *= 1.5) {
        float in[4] = {static_cast<float>(v), 0, 0, 0}, out[4];
        vst1q_f32(out, log_ps(vld1q_f32(in)));
        double ref = std::log(static_cast<double>(static_cast<float>(v)));
        if (ref != 0.0) maxerr = std::max(maxerr, std::abs((out[0] - ref) / ref));
    }
    for (double v = -80.0; v <= 80.0; v += 0.25) {
        float in[4] = {static_cast<float>(v), 0, 0, 0}, out[4];
        vst1q_f32(out, exp_ps(vld1q_f32(in)));
        double ref = std::exp(static_cast<double>(static_cast<float>(v)));
        if (ref != 0.0) maxerr = std::max(maxerr, std::abs((out[0] - ref) / ref));
    }
    return maxerr;
#else
    return 0.0;
#endif
}

// Raw complex HW2 -> linear amplitude HW.
inline void raw_to_lina(const float* in_hwc, float* out_hw, int H, int W)
{
    for (int i = 0; i < H * W; ++i) {
        float r  = in_hwc[i * 2 + 0];
        float im = in_hwc[i * 2 + 1];
        out_hw[i] = std::sqrt(r * r + im * im);
    }
}

} // namespace ddc
