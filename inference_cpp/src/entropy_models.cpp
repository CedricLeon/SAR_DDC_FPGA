/* entropy_models.cpp
 * C++ port of scripts/fpga/entropy_models_inference.py
 *
 * See entropy_models.hpp for the API contract and parameter file layout.
 */

#include "entropy_models.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "npy_io.hpp"
#include "rans/rans_interface_cxx.hpp"

namespace ddc
{

    // ---------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------
    namespace
    {

        // Load a 2-D int32 NPY into a vector-of-vectors (row-major).
        std::vector<std::vector<int32_t>> load_2d_int32(const std::filesystem::path &path)
        {
            NpyArray arr = npy_load(path.string());
            if (arr.shape.size() != 2)
                throw std::runtime_error("Expected 2-D array in " + path.string());
            const size_t rows = arr.shape[0];
            const size_t cols = arr.shape[1];
            auto flat = arr.as_int32();
            std::vector<std::vector<int32_t>> result(rows, std::vector<int32_t>(cols));
            for (size_t r = 0; r < rows; ++r)
                for (size_t c = 0; c < cols; ++c)
                    result[r][c] = flat[r * cols + c];
            return result;
        }

        // Portable banker's rounding (round-half-to-even), independent of FPU mode.
        // This matches numpy.round() and torch.round() exactly, including for values
        // that land on exactly 0.5.  std::rint also uses round-to-nearest-even but
        // reads the FPU rounding mode register, which VART/XIR may have changed.
        inline int32_t round_half_to_even(float v)
        {
            float flr = std::floor(v);
            float diff = v - flr;
            if (diff < 0.5f)
                return static_cast<int32_t>(flr);
            if (diff > 0.5f)
                return static_cast<int32_t>(flr) + 1;
            // Exactly 0.5 — round to nearest even
            int32_t i = static_cast<int32_t>(flr);
            return (i % 2 == 0) ? i : i + 1;
        }

    } // anonymous namespace

    // ---------------------------------------------------------------------------
    // EntropyBottleneck::load_params
    // ---------------------------------------------------------------------------
    void EntropyBottleneck::load_params(const std::filesystem::path &params_dir)
    {
        quantized_cdf_ = load_2d_int32(params_dir / "eb_quantized_cdf.npy");
        cdf_lengths_ = npy_load_int32((params_dir / "eb_cdf_length.npy").string());
        offsets_ = npy_load_int32((params_dir / "eb_offset.npy").string());
        medians_ = npy_load_float32((params_dir / "eb_medians.npy").string());

        channels_ = static_cast<int>(medians_.size());
        if (channels_ == 0)
            throw std::runtime_error("EntropyBottleneck: empty medians — check eb_medians.npy");
        if (static_cast<int>(quantized_cdf_.size()) != channels_)
            throw std::runtime_error("EntropyBottleneck: CDF row count != channels");
        if (static_cast<int>(cdf_lengths_.size()) != channels_)
            throw std::runtime_error("EntropyBottleneck: cdf_length size != channels");
        if (static_cast<int>(offsets_.size()) != channels_)
            throw std::runtime_error("EntropyBottleneck: offset size != channels");

        loaded_ = true;
    }

    // ---------------------------------------------------------------------------
    // EntropyBottleneck::compress
    // Input z: float*, layout HWC (H*W*C elements).
    // Returns bitstring for one patch.
    // ---------------------------------------------------------------------------
    std::vector<uint8_t> EntropyBottleneck::compress(const float *z, int H, int W) const
    {
        if (!loaded_)
            throw std::runtime_error("EntropyBottleneck: params not loaded");

        const int C = channels_;
        const int n = H * W * C;

        // symbols = round(z - medians)  per channel
        std::vector<int32_t> symbols(n);
        std::vector<int32_t> indexes(n);
        for (int h = 0; h < H; ++h)
        {
            for (int w = 0; w < W; ++w)
            {
                for (int c = 0; c < C; ++c)
                {
                    const int idx = (h * W + w) * C + c;
                    // round_half_to_even: portable banker's rounding, FPU-mode independent.
                    // Matches numpy.round() / torch.round() regardless of VART rounding mode.
                    symbols[idx] = round_half_to_even(z[idx] - medians_[c]);
                    indexes[idx] = c;
                }
            }
        }

        // Build cdf_lengths as a plain vector (rANS API expects vector<int32_t>)
        RansEncoderCxx enc;
        return enc.encode_with_indexes(symbols, indexes,
                                       quantized_cdf_, cdf_lengths_, offsets_);
    }

    // ---------------------------------------------------------------------------
    // EntropyBottleneck::decompress
    // ---------------------------------------------------------------------------
    std::vector<float> EntropyBottleneck::decompress(
        const std::vector<uint8_t> &bitstring, int H, int W) const
    {
        if (!loaded_)
            throw std::runtime_error("EntropyBottleneck: params not loaded");

        const int C = channels_;
        const int n = H * W * C;

        std::vector<int32_t> indexes(n);
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w)
                for (int c = 0; c < C; ++c)
                    indexes[(h * W + w) * C + c] = c;

        RansDecoderCxx dec;
        std::vector<int32_t> decoded = dec.decode_with_indexes(
            bitstring, indexes, quantized_cdf_, cdf_lengths_, offsets_);

        // Reconstruct: z = symbols + medians
        std::vector<float> output(n);
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w)
                for (int c = 0; c < C; ++c)
                {
                    const int i = (h * W + w) * C + c;
                    output[i] = static_cast<float>(decoded[i]) + medians_[c];
                }

        return output;
    }

    // ---------------------------------------------------------------------------
    // GaussianConditional::load_params
    // ---------------------------------------------------------------------------
    void GaussianConditional::load_params(const std::filesystem::path &params_dir)
    {
        scale_table_ = npy_load_float32((params_dir / "gc_scale_table.npy").string());
        quantized_cdf_ = load_2d_int32(params_dir / "gc_quantized_cdf.npy");
        cdf_lengths_ = npy_load_int32((params_dir / "gc_cdf_length.npy").string());
        offsets_ = npy_load_int32((params_dir / "gc_offset.npy").string());

        if (scale_table_.empty())
            throw std::runtime_error("GaussianConditional: empty scale_table");
        if (quantized_cdf_.size() != scale_table_.size())
            throw std::runtime_error("GaussianConditional: CDF row count != scale_table size");

        loaded_ = true;
    }

    // ---------------------------------------------------------------------------
    // GaussianConditional::scale_index
    // Mirrors Python: searchsorted(scale_table, scale, side='right') - 1, clamped.
    // ---------------------------------------------------------------------------
    int32_t GaussianConditional::scale_index(float scale) const
    {
        // lower_bound gives first element >= scale; subtract 1 for "right" semantics
        auto it = std::upper_bound(scale_table_.cbegin(), scale_table_.cend(), scale);
        int32_t idx = static_cast<int32_t>(std::distance(scale_table_.cbegin(), it)) - 1;
        if (idx < 0)
            idx = 0;
        if (idx >= static_cast<int32_t>(scale_table_.size()))
            idx = static_cast<int32_t>(scale_table_.size()) - 1;
        return idx;
    }

    // ---------------------------------------------------------------------------
    // GaussianConditional::compress
    // ---------------------------------------------------------------------------
    std::vector<uint8_t> GaussianConditional::compress(
        const float *y, const float *scales, const float *means,
        int H, int W, int C) const
    {
        if (!loaded_)
            throw std::runtime_error("GaussianConditional: params not loaded");

        const int n = H * W * C;
        std::vector<int32_t> symbols(n);
        std::vector<int32_t> indexes(n);

        for (int i = 0; i < n; ++i)
        {
            float val = y[i];
            if (means)
                val -= means[i];
            // round_half_to_even: portable banker's rounding, FPU-mode independent.
            // Matches numpy.round() and torch.round() regardless of VART rounding mode.
            symbols[i] = round_half_to_even(val);
            indexes[i] = scale_index(scales[i]);
        }

        RansEncoderCxx enc;
        return enc.encode_with_indexes(symbols, indexes,
                                       quantized_cdf_, cdf_lengths_, offsets_);
    }

    // ---------------------------------------------------------------------------
    // GaussianConditional::decompress
    // ---------------------------------------------------------------------------
    std::vector<float> GaussianConditional::decompress(
        const std::vector<uint8_t> &bitstring,
        const float *scales, const float *means,
        int H, int W, int C) const
    {
        if (!loaded_)
            throw std::runtime_error("GaussianConditional: params not loaded");

        const int n = H * W * C;
        std::vector<int32_t> indexes(n);
        for (int i = 0; i < n; ++i)
            indexes[i] = scale_index(scales[i]);

        RansDecoderCxx dec;
        std::vector<int32_t> decoded = dec.decode_with_indexes(
            bitstring, indexes, quantized_cdf_, cdf_lengths_, offsets_);

        std::vector<float> output(n);
        for (int i = 0; i < n; ++i)
        {
            output[i] = static_cast<float>(decoded[i]);
            if (means)
                output[i] += means[i];
        }

        return output;
    }

} // namespace ddc
