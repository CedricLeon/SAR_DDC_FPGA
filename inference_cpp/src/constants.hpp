#pragma once
// Constants shared across the C++ inference pipeline.
// Values must stay in sync with scripts/fpga/inference_utils.py.

namespace ddc {

// Normalisation constants for log-amplitude input
// log(amp^2 + EPS) is normalised to [0, 1] using:
//   norm = (logI - 2*AMP_MIN) / (2*AMP_MAX - 2*AMP_MIN)
constexpr float AMP_MIN     = 4.605170249938965f;
constexpr float AMP_MAX     = 10.742239952087402f;
constexpr float EPS         = 1e-2f;

// 99th percentile of linear amplitude — used as the PSNR peak (clipping target)
constexpr float AMP_LIN_99  = 545.2018433569272f;

// Fixed spatial sizes (must match the compiled DPU model)
constexpr int IMAGE_SIZE  = 256;   // patch width/height
constexpr int S_MAIN      = 16;    // main autoencoder downsampling factor
constexpr int S_HYPER     = 8;     // hyper autoencoder downsampling factor

constexpr int H_LATENT    = IMAGE_SIZE / S_MAIN;          // 16
constexpr int W_LATENT    = IMAGE_SIZE / S_MAIN;          // 16
constexpr int H_HYPER     = H_LATENT / S_HYPER;           // 2
constexpr int W_HYPER     = W_LATENT / S_HYPER;           // 2

constexpr int C_MAIN      = 128;   // main encoder output channels (g_a, g_s)
constexpr int C_HYPER     = C_MAIN * 2;  // hyper encoder channels (h_a, h_s)

// ENL region-of-interest for the Hamburg water body (row0, row1, col0, col1)
constexpr int HAMBURG_ENL_ROI[4] = {400, 600, 800, 1000};

} // namespace ddc
