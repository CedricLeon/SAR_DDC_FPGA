/* metrics.cpp
 * C++ port of scripts/fpga/inference_utils.py MetricsTracker.
 *
 * See metrics.hpp for API and conventions.
 * Requires OpenCV with opencv_quality module (available on ZCU102: 4.5.2).
 */

#include "metrics.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#ifndef WITHOUT_OPENCV
#  include <opencv2/opencv.hpp>
#  include <opencv2/quality.hpp>
#endif

#include "constants.hpp"

namespace ddc {

// ---------------------------------------------------------------------------
// compute_mse
// ---------------------------------------------------------------------------
double compute_mse(const float* a, const float* b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        double ca = std::min(static_cast<double>(a[i]),
                             static_cast<double>(AMP_LIN_99));
        double cb = std::min(static_cast<double>(b[i]),
                             static_cast<double>(AMP_LIN_99));
        ca = std::max(ca, 0.0);
        cb = std::max(cb, 0.0);
        sum += (ca - cb) * (ca - cb);
    }
    return sum / static_cast<double>(n);
}

// ---------------------------------------------------------------------------
// compute_psnr
// ---------------------------------------------------------------------------
double compute_psnr(const float* a, const float* b, int n) {
    const double mse = compute_mse(a, b, n);
    if (mse <= 0.0) return std::numeric_limits<double>::infinity();
    return 20.0 * std::log10(static_cast<double>(AMP_LIN_99))
           - 10.0 * std::log10(mse);
}

// ---------------------------------------------------------------------------
// clip_amp99 — shared basis of every reference-based distortion metric
// ---------------------------------------------------------------------------
// Mirrors src/utils/metrics.py::_clip_to_amp99: scores are taken on the 99th-percentile
// amplitude so no metric is driven by the handful of bright scatterers above it. Matters
// most on the board, where the DPU caps the recon at 2100 while float32 reaches ~1e5.
static std::vector<float> clip_amp99(const float* x, int n) {
    std::vector<float> out(n);
    for (int i = 0; i < n; ++i)
        out[i] = std::min(x[i], static_cast<float>(AMP_LIN_99));
    return out;
}

// ---------------------------------------------------------------------------
// compute_ssim — delegates to OpenCV QualitySSIM
// ---------------------------------------------------------------------------
// OpenCV hard-codes the SSIM stabilisers to C1=(0.01*255)^2, C2=(0.03*255)^2, i.e. it always
// scores at data_range=255 and offers no way to pass one. SSIM is invariant to scaling both
// images and the data_range together, so clipping to AMP_LIN_99 and rescaling by
// 255/AMP_LIN_99 makes QualitySSIM return exactly SSIM at data_range=AMP_LIN_99 — the
// convention used by src/utils/metrics.py::ssim, so board and host numbers are comparable.
double compute_ssim(const float* a, const float* b, int H, int W) {
#ifdef WITHOUT_OPENCV
    (void)a; (void)b; (void)H; (void)W;
    return 0.0; // stub — OpenCV not available in this build
#else
    const int n = H * W;
    const float scale = 255.0f / static_cast<float>(AMP_LIN_99);
    std::vector<float> a_clip = clip_amp99(a, n);
    std::vector<float> b_clip = clip_amp99(b, n);
    for (int i = 0; i < n; ++i) { a_clip[i] *= scale; b_clip[i] *= scale; }

    cv::Mat a_mat(H, W, CV_32FC1, a_clip.data());
    cv::Mat b_mat(H, W, CV_32FC1, b_clip.data());
    cv::Scalar result = cv::quality::QualitySSIM::compute(
        a_mat, b_mat, cv::noArray());
    // result[0] is the per-channel value; we have a single channel.
    return static_cast<double>(result[0]);
#endif
}

// ---------------------------------------------------------------------------
// compute_enl
// ---------------------------------------------------------------------------
double compute_enl(const float* a, int H, int W, const Roi* roi) {
    // Determine iteration range
    int r0 = 0, r1 = H, c0 = 0, c1 = W;
    if (roi) { r0 = roi->r0; r1 = roi->r1; c0 = roi->c0; c1 = roi->c1; }

    const int n = (r1 - r0) * (c1 - c0);
    if (n <= 0) return std::numeric_limits<double>::quiet_NaN();

    // Compute mean and variance of linear intensity (= amplitude^2)
    double sum = 0.0, sum2 = 0.0;
    for (int r = r0; r < r1; ++r) {
        for (int c = c0; c < c1; ++c) {
            const double I = static_cast<double>(a[r * W + c]);
            const double linI = I * I;
            sum  += linI;
            sum2 += linI * linI;
        }
    }
    const double mu  = sum / n;
    const double mu2 = sum2 / n;
    const double var = mu2 - mu * mu;
    if (var <= 0.0) return std::numeric_limits<double>::quiet_NaN();
    return mu * mu / var;
}

// ---------------------------------------------------------------------------
// compute_epd
// Gradient magnitude correlation between recon and ref.
// ---------------------------------------------------------------------------
double compute_epd(const float* recon, const float* ref, int H, int W) {
    // Clipped to AMP_LIN_99 like every other reference-based metric (see clip_amp99):
    // unclipped, the gradient sums are dominated by bright scatterers, so EPD measures
    // point-target representation rather than edge preservation.
    const std::vector<float> recon_clip = clip_amp99(recon, H * W);
    const std::vector<float> ref_clip   = clip_amp99(ref, H * W);

    // Compute central-difference gradient magnitudes
    auto grad_mag = [H, W](const float* img) {
        std::vector<float> gm(H * W, 0.0f);
        for (int r = 1; r < H - 1; ++r) {
            for (int c = 1; c < W - 1; ++c) {
                const float gx = img[r * W + (c + 1)] - img[r * W + (c - 1)];
                const float gy = img[(r + 1) * W + c] - img[(r - 1) * W + c];
                gm[r * W + c] = std::sqrt(gx * gx + gy * gy);
            }
        }
        return gm;
    };

    auto gm_recon = grad_mag(recon_clip.data());
    auto gm_ref   = grad_mag(ref_clip.data());

    double num   = 0.0;
    double denom = 0.0;
    const int n  = H * W;
    for (int i = 0; i < n; ++i) {
        num   += static_cast<double>(gm_recon[i]) * gm_ref[i];
        denom += static_cast<double>(gm_ref[i])   * gm_ref[i];
    }
    if (denom <= 0.0) return std::numeric_limits<double>::quiet_NaN();
    return num / denom;
}

// ---------------------------------------------------------------------------
// compute_ratio_mean
// ---------------------------------------------------------------------------
double compute_ratio_mean(const float* recon, const float* noisy, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        const double recon_I = static_cast<double>(recon[i]) * recon[i];
        const double noisy_I = static_cast<double>(noisy[i]) * noisy[i];
        sum += noisy_I / (recon_I + 1e-10);
    }
    return sum / static_cast<double>(n);
}

// ---------------------------------------------------------------------------
// compute_ratio_enl
// ---------------------------------------------------------------------------
double compute_ratio_enl(const float* recon, const float* noisy, int n) {
    double sum = 0.0, sum2 = 0.0;
    for (int i = 0; i < n; ++i) {
        const double recon_I = static_cast<double>(recon[i]) * recon[i];
        const double noisy_I = static_cast<double>(noisy[i]) * noisy[i];
        const double r = noisy_I / (recon_I + 1e-10);
        sum  += r;
        sum2 += r * r;
    }
    const double mu  = sum / n;
    const double mu2 = sum2 / n;
    const double var = mu2 - mu * mu;
    if (var <= 0.0) return std::numeric_limits<double>::quiet_NaN();
    return mu * mu / var;
}

// ---------------------------------------------------------------------------
// compute_bpp
// ---------------------------------------------------------------------------
double compute_bpp(int num_bytes, int H, int W) {
    return static_cast<double>(num_bytes * 8) / static_cast<double>(H * W);
}

// ---------------------------------------------------------------------------
// MetricsAccumulator::update
// ---------------------------------------------------------------------------
void MetricsAccumulator::update(const float* recon, const float* noisy,
                                 int H, int W, int num_bytes, const Roi* roi)
{
    const int n = H * W;
    sum_mse_        += compute_mse(recon, noisy, n);
    sum_psnr_       += compute_psnr(recon, noisy, n);
    sum_ssim_       += compute_ssim(recon, noisy, H, W);
    sum_enl_        += compute_enl(recon, H, W, roi);
    sum_bpp_        += compute_bpp(num_bytes, H, W);
    sum_epd_        += compute_epd(recon, noisy, H, W);
    sum_ratio_mean_ += compute_ratio_mean(recon, noisy, n);
    sum_ratio_enl_  += compute_ratio_enl(recon, noisy, n);
    ++count_;
}

// ---------------------------------------------------------------------------
// MetricsAccumulator::mean
// ---------------------------------------------------------------------------
Metrics MetricsAccumulator::mean() const {
    if (count_ == 0) return {};
    const double c = static_cast<double>(count_);
    Metrics m;
    m.mse        = sum_mse_        / c;
    m.psnr       = sum_psnr_       / c;
    m.ssim       = sum_ssim_       / c;
    m.enl        = sum_enl_        / c;
    m.bpp        = sum_bpp_        / c;
    m.epd        = sum_epd_        / c;
    m.ratio_mean = sum_ratio_mean_ / c;
    m.ratio_enl  = sum_ratio_enl_  / c;
    return m;
}

// ---------------------------------------------------------------------------
// MetricsAccumulator::reset
// ---------------------------------------------------------------------------
void MetricsAccumulator::reset() {
    count_ = 0;
    sum_mse_ = sum_psnr_ = sum_ssim_ = sum_enl_ = 0.0;
    sum_bpp_ = sum_epd_ = sum_ratio_mean_ = sum_ratio_enl_ = 0.0;
}

} // namespace ddc
