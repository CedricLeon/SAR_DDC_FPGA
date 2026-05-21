/* metrics.hpp
 * C++ port of scripts/fpga/inference_utils.py MetricsTracker.
 *
 * All metric functions expect linear amplitude arrays (not log, not intensity).
 * Clipping to [0, AMP_LIN_99] is applied internally where specified.
 *
 * OpenCV (with opencv_quality module) is required for SSIM.
 * All other metrics are pure C++.
 *
 * EPD (Edge Preservation Degree): gradient-magnitude correlation,
 *   EPD=1.0 means perfect edge preservation.
 * ENL: Equivalent Number of Looks on linear intensity = mean(I)^2 / var(I).
 * ratio_mean / ratio_enl: statistics of the ratio image noisy_I / recon_I.
 * BPP: (num_bytes * 8) / (H * W).
 */
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

namespace ddc {

// Optional ROI for ENL computation: {row_start, row_end, col_start, col_end}
struct Roi {
    int r0, r1, c0, c1;
};

struct Metrics {
    double mse       = 0.0;
    double psnr      = 0.0;
    double ssim      = 0.0;
    double enl       = 0.0;     // on recon
    double bpp       = 0.0;
    double epd       = 0.0;
    double ratio_mean = 0.0;
    double ratio_enl  = 0.0;
};

// ---------------------------------------------------------------------------
// Individual metric functions (all work on flat float* arrays, size H*W)
// ---------------------------------------------------------------------------

// MSE: clips both to [0, AMP_LIN_99]
double compute_mse(const float* a, const float* b, int n);

// PSNR: 20*log10(AMP_LIN_99) - 10*log10(MSE)
double compute_psnr(const float* a, const float* b, int n);

// SSIM: via cv::quality::QualitySSIM (OpenCV). a and b are H×W float images.
double compute_ssim(const float* a, const float* b, int H, int W);

// ENL on linear intensity (= square of amplitude).
// roi is optional; if present, crops the image before computing.
double compute_enl(const float* a, int H, int W,
                   const Roi* roi = nullptr);

// EPD: gradient-magnitude correlation between recon and ref (linA).
double compute_epd(const float* recon, const float* ref, int H, int W);

// ratio_mean: mean(noisy_I / (recon_I + 1e-10))
double compute_ratio_mean(const float* recon, const float* noisy, int n);

// ratio_enl: ENL of ratio image noisy_I / (recon_I + 1e-10)
double compute_ratio_enl(const float* recon, const float* noisy, int n);

// BPP: (num_bytes * 8) / (H * W)
double compute_bpp(int num_bytes, int H, int W);

// ---------------------------------------------------------------------------
// MetricsAccumulator — running average over multiple patches
// ---------------------------------------------------------------------------
class MetricsAccumulator {
public:
    MetricsAccumulator()  = default;

    // Add one patch. recon and noisy are flat float arrays of size H*W (linA).
    // num_bytes: total compressed bytes for this patch (for BPP).
    // roi: optional for ENL computation.
    void update(const float* recon, const float* noisy,
                int H, int W, int num_bytes,
                const Roi* roi = nullptr);

    // Return per-metric averages.
    Metrics mean() const;

    int count() const { return count_; }
    void reset();

private:
    int    count_ = 0;
    double sum_mse_        = 0.0;
    double sum_psnr_       = 0.0;
    double sum_ssim_       = 0.0;
    double sum_enl_        = 0.0;
    double sum_bpp_        = 0.0;
    double sum_epd_        = 0.0;
    double sum_ratio_mean_ = 0.0;
    double sum_ratio_enl_  = 0.0;
};

} // namespace ddc
