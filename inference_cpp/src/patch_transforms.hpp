#pragma once
// patch_transforms.hpp — pure per-patch transforms shared by inference_runner
// and bench_pipeline.  Header-only, no deps beyond <cmath> and constants.hpp.
//
// Extracted from the anonymous namespace in inference_runner.cpp so that
// bench_pipeline.cpp can include them without duplicating the math.

#include <cmath>

#include "constants.hpp"

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
