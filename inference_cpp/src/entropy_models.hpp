/* entropy_models.hpp
 * C++ port of scripts/fpga/entropy_models_inference.py
 *
 * EntropyBottleneck and GaussianConditional use the pybind11-free rANS codec
 * (rans_interface_cxx.hpp) and load parameters from individual .npy files.
 *
 * Parameter files (loaded by load_params()):
 *   EntropyBottleneck:
 *     entropy_params/eb_quantized_cdf.npy   int32  (C, cdf_len_max)
 *     entropy_params/eb_cdf_length.npy      int32  (C,)
 *     entropy_params/eb_offset.npy          int32  (C,)
 *     entropy_params/eb_medians.npy         float32 (C,)
 *
 *   GaussianConditional (optional — only for ResidualScaleHyperprior models):
 *     entropy_params/gc_scale_table.npy     float32 (n_scales,)
 *     entropy_params/gc_quantized_cdf.npy   int32  (n_scales, cdf_len_max)
 *     entropy_params/gc_cdf_length.npy      int32  (n_scales,)
 *     entropy_params/gc_offset.npy          int32  (n_scales,)
 *
 * Data layout: C++ inference uses NHWC (the DPU output layout).
 * Batch size N is always 1 in on-board inference; the API accepts it for symmetry.
 */
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "rans/rans_interface_cxx.hpp"

namespace ddc {

// ---------------------------------------------------------------------------
// EntropyBottleneck
// ---------------------------------------------------------------------------
class EntropyBottleneck {
public:
    EntropyBottleneck() = default;

    // Load parameter tables from a directory containing individual .npy files.
    // Throws std::runtime_error on failure.
    void load_params(const std::filesystem::path& params_dir);

    // Compress one batch item of latent z.
    // z: flat float32 buffer (H*W*C), layout HWC.
    // H, W: spatial dimensions.
    // Returns one bitstring per batch item.
    std::vector<uint8_t> compress(const float* z, int H, int W) const;

    // Decompress one bitstring back to latent z (H, W, C).
    // Output: flat float32 vector, length H*W*C, layout HWC.
    std::vector<float> decompress(const std::vector<uint8_t>& bitstring,
                                  int H, int W) const;

    int channels() const { return channels_; }
    bool is_loaded() const { return loaded_; }

private:
    int channels_ = 0;
    bool loaded_  = false;

    std::vector<std::vector<int32_t>> quantized_cdf_; // [C][cdf_len]
    std::vector<int32_t>              cdf_lengths_;   // [C]
    std::vector<int32_t>              offsets_;       // [C]
    std::vector<float>                medians_;       // [C]
};

// ---------------------------------------------------------------------------
// GaussianConditional
// ---------------------------------------------------------------------------
class GaussianConditional {
public:
    GaussianConditional() = default;

    void load_params(const std::filesystem::path& params_dir);

    // Compress y (H*W*C float, HWC) given scale estimates (H*W*C float, HWC).
    // means may be nullptr (no mean shift).
    std::vector<uint8_t> compress(const float* y, const float* scales,
                                  const float* means,   // may be nullptr
                                  int H, int W, int C) const;

    // Decompress one bitstring.
    // scales, means: (H*W*C) — same as compress.
    std::vector<float> decompress(const std::vector<uint8_t>& bitstring,
                                  const float* scales,
                                  const float* means,   // may be nullptr
                                  int H, int W, int C) const;

    bool is_loaded() const { return loaded_; }

private:
    // Map a scale value to its nearest index in scale_table_.
    int32_t scale_index(float scale) const;

    bool loaded_ = false;

    std::vector<float>                scale_table_;    // [n_scales]
    std::vector<std::vector<int32_t>> quantized_cdf_;  // [n_scales][cdf_len]
    std::vector<int32_t>              cdf_lengths_;    // [n_scales]
    std::vector<int32_t>              offsets_;        // [n_scales]
};

} // namespace ddc
