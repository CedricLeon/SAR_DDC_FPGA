/* rans_interface_cxx.cpp
 * Pybind11-free fork of CompressAI/compressai/cpp_exts/rans/rans_interface.cpp
 *
 * Logic is identical to the original. Only the buffer types have changed:
 *   py::bytes  ->  std::vector<uint8_t>   (encode / flush)
 *   std::string -> std::vector<uint8_t>   (decode input)
 *
 * The rANS64 core functions (rans64.h) are untouched.
 */

#include "rans_interface_cxx.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iterator>
#include <stdexcept>
#include <vector>

#include "rans64.h"
#include "rans_profile.hpp"

namespace ddc {

// Probability precision (bits) — must match the Python encoder.
constexpr int precision = 16;

constexpr uint16_t bypass_precision = 4;
constexpr uint16_t max_bypass_val   = (1u << bypass_precision) - 1u;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
namespace {

inline void Rans64EncPutBits(Rans64State* r, uint32_t** pptr,
                             uint32_t val, uint32_t nbits) {
    assert(nbits <= 16);
    assert(val < (1u << nbits));
    uint64_t x     = *r;
    uint32_t freq  = 1u << (16 - nbits);
    uint64_t x_max = ((RANS64_L >> 16) << 32) * freq;
    if (x >= x_max) {
        *pptr -= 1;
        **pptr = static_cast<uint32_t>(x);
        x >>= 32;
    }
    *r = (x << nbits) | val;
}

inline uint32_t Rans64DecGetBits(Rans64State* r, uint32_t** pptr, uint32_t n_bits) {
    uint64_t x   = *r;
    uint32_t val = static_cast<uint32_t>(x & ((1u << n_bits) - 1u));
    x >>= n_bits;
    if (x < RANS64_L) {
        x = (x << 32) | **pptr;
        *pptr += 1;
    }
    *r = x;
    return val;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// BufferedRansEncoderCxx::encode_with_indexes
// Mirrors original exactly — no type changes needed here.
// ---------------------------------------------------------------------------
void BufferedRansEncoderCxx::encode_with_indexes(
    const std::vector<int32_t>& symbols,
    const std::vector<int32_t>& indexes,
    const std::vector<std::vector<int32_t>>& cdfs,
    const std::vector<int32_t>& cdfs_sizes,
    const std::vector<int32_t>& offsets)
{
    assert(cdfs.size() == cdfs_sizes.size());

    DDC_RPROF_CUR(::ddc::rprof::ST_LOOKUP);  // forward CDF-lookup pass (builds _syms)

    for (size_t i = 0; i < symbols.size(); ++i) {
        const int32_t cdf_idx = indexes[i];
        assert(cdf_idx >= 0 && cdf_idx < static_cast<int32_t>(cdfs.size()));

        const auto& cdf = cdfs[cdf_idx];
        const int32_t max_value = cdfs_sizes[cdf_idx] - 2;
        assert(max_value >= 0);
        assert((max_value + 1) < static_cast<int32_t>(cdf.size()));

        int32_t value = symbols[i] - offsets[cdf_idx];

        uint32_t raw_val = 0;
        if (value < 0) {
            raw_val = static_cast<uint32_t>(-2 * value - 1);
            value   = max_value;
        } else if (value >= max_value) {
            raw_val = static_cast<uint32_t>(2 * (value - max_value));
            value   = max_value;
        }

        assert(value >= 0 && value < cdfs_sizes[cdf_idx] - 1);
        _syms.push_back({static_cast<uint16_t>(cdf[value]),
                         static_cast<uint16_t>(cdf[value + 1] - cdf[value]),
                         false});

        if (value == max_value) {
            int32_t n_bypass = 0;
            while ((raw_val >> (n_bypass * bypass_precision)) != 0) ++n_bypass;

            int32_t val = n_bypass;
            while (val >= static_cast<int32_t>(max_bypass_val)) {
                _syms.push_back({max_bypass_val, max_bypass_val + 1u, true});
                val -= max_bypass_val;
            }
            _syms.push_back({static_cast<uint16_t>(val),
                             static_cast<uint16_t>(val + 1), true});

            for (int32_t j = 0; j < n_bypass; ++j) {
                const int32_t bv = (raw_val >> (j * bypass_precision)) & max_bypass_val;
                _syms.push_back({static_cast<uint16_t>(bv),
                                 static_cast<uint16_t>(bv + 1), true});
            }
        }
    }
}

// ---------------------------------------------------------------------------
// BufferedRansEncoderCxx::flush  — changed: returns std::vector<uint8_t>
// ---------------------------------------------------------------------------
std::vector<uint8_t> BufferedRansEncoderCxx::flush() {
    DDC_RPROF_CUR(::ddc::rprof::ST_FLUSH);  // reverse rANS renorm loop + byte copy

    Rans64State rans;
    Rans64EncInit(&rans);

    std::vector<uint32_t> output(_syms.size(), 0xCCCCCCCCu);
    uint32_t* ptr = output.data() + output.size();

    while (!_syms.empty()) {
        const RansSymbol sym = _syms.back();
        if (!sym.bypass) {
            Rans64EncPut(&rans, &ptr, sym.start, sym.range, precision);
        } else {
            Rans64EncPutBits(&rans, &ptr, sym.start, bypass_precision);
        }
        _syms.pop_back();
    }

    Rans64EncFlush(&rans, &ptr);

    const size_t n32 = static_cast<size_t>(
        std::distance(ptr, output.data() + output.size()));
    // Convert the uint32_t words to a byte buffer
    const uint8_t* byte_ptr = reinterpret_cast<const uint8_t*>(ptr);
    return std::vector<uint8_t>(byte_ptr, byte_ptr + n32 * sizeof(uint32_t));
}

// ---------------------------------------------------------------------------
// RansEncoderCxx::encode_with_indexes
// ---------------------------------------------------------------------------
std::vector<uint8_t> RansEncoderCxx::encode_with_indexes(
    const std::vector<int32_t>& symbols,
    const std::vector<int32_t>& indexes,
    const std::vector<std::vector<int32_t>>& cdfs,
    const std::vector<int32_t>& cdfs_sizes,
    const std::vector<int32_t>& offsets)
{
    BufferedRansEncoderCxx enc;
    enc.encode_with_indexes(symbols, indexes, cdfs, cdfs_sizes, offsets);
    return enc.flush();
}

// ---------------------------------------------------------------------------
// RansDecoderCxx::decode_with_indexes — input changed to vector<uint8_t>
// ---------------------------------------------------------------------------
std::vector<int32_t> RansDecoderCxx::decode_with_indexes(
    const std::vector<uint8_t>& encoded,
    const std::vector<int32_t>& indexes,
    const std::vector<std::vector<int32_t>>& cdfs,
    const std::vector<int32_t>& cdfs_sizes,
    const std::vector<int32_t>& offsets)
{
    assert(cdfs.size() == cdfs_sizes.size());

    std::vector<int32_t> output(indexes.size());

    Rans64State rans;
    // The encoder writes uint32_t words; cast byte pointer to uint32_t*
    uint32_t* ptr = reinterpret_cast<uint32_t*>(
        const_cast<uint8_t*>(encoded.data()));
    Rans64DecInit(&rans, &ptr);

    for (int i = 0; i < static_cast<int>(indexes.size()); ++i) {
        const int32_t cdf_idx = indexes[i];
        assert(cdf_idx >= 0 && cdf_idx < static_cast<int32_t>(cdfs.size()));

        const auto& cdf       = cdfs[cdf_idx];
        const int32_t max_val = cdfs_sizes[cdf_idx] - 2;
        const int32_t offset  = offsets[cdf_idx];

        const uint32_t cum_freq = Rans64DecGet(&rans, precision);

        auto cdf_end = cdf.cbegin() + cdfs_sizes[cdf_idx];
        auto it = std::find_if(cdf.cbegin(), cdf_end,
                               [cum_freq](uint32_t v) { return v > cum_freq; });
        assert(it != cdf_end);
        const uint32_t s = static_cast<uint32_t>(std::distance(cdf.cbegin(), it)) - 1u;

        Rans64DecAdvance(&rans, &ptr, cdf[s], cdf[s + 1] - cdf[s], precision);

        int32_t value = static_cast<int32_t>(s);

        if (value == max_val) {
            // Bypass decode
            int32_t val      = Rans64DecGetBits(&rans, &ptr, bypass_precision);
            int32_t n_bypass = val;
            while (val == static_cast<int32_t>(max_bypass_val)) {
                val       = Rans64DecGetBits(&rans, &ptr, bypass_precision);
                n_bypass += val;
            }
            int32_t raw_val = 0;
            for (int j = 0; j < n_bypass; ++j) {
                val      = Rans64DecGetBits(&rans, &ptr, bypass_precision);
                raw_val |= val << (j * bypass_precision);
            }
            value = raw_val >> 1;
            if (raw_val & 1) value = -value - 1;
            else             value += max_val;
        }

        output[i] = value + offset;
    }

    return output;
}

// ---------------------------------------------------------------------------
// RansDecoderCxx::set_stream — input changed to vector<uint8_t>
// ---------------------------------------------------------------------------
void RansDecoderCxx::set_stream(const std::vector<uint8_t>& stream) {
    _stream = stream;
    _ptr    = reinterpret_cast<uint32_t*>(_stream.data());
    Rans64DecInit(&_rans, &_ptr);
}

// ---------------------------------------------------------------------------
// RansDecoderCxx::decode_stream — unchanged logic
// ---------------------------------------------------------------------------
std::vector<int32_t> RansDecoderCxx::decode_stream(
    const std::vector<int32_t>& indexes,
    const std::vector<std::vector<int32_t>>& cdfs,
    const std::vector<int32_t>& cdfs_sizes,
    const std::vector<int32_t>& offsets)
{
    assert(cdfs.size() == cdfs_sizes.size());

    std::vector<int32_t> output(indexes.size());

    for (int i = 0; i < static_cast<int>(indexes.size()); ++i) {
        const int32_t cdf_idx = indexes[i];
        const auto& cdf       = cdfs[cdf_idx];
        const int32_t max_val = cdfs_sizes[cdf_idx] - 2;
        const int32_t offset  = offsets[cdf_idx];

        const uint32_t cum_freq = Rans64DecGet(&_rans, precision);

        auto cdf_end = cdf.cbegin() + cdfs_sizes[cdf_idx];
        auto it = std::find_if(cdf.cbegin(), cdf_end,
                               [cum_freq](uint32_t v) { return v > cum_freq; });
        const uint32_t s = static_cast<uint32_t>(std::distance(cdf.cbegin(), it)) - 1u;

        Rans64DecAdvance(&_rans, &_ptr, cdf[s], cdf[s + 1] - cdf[s], precision);

        int32_t value = static_cast<int32_t>(s);

        if (value == max_val) {
            int32_t val      = Rans64DecGetBits(&_rans, &_ptr, bypass_precision);
            int32_t n_bypass = val;
            while (val == static_cast<int32_t>(max_bypass_val)) {
                val       = Rans64DecGetBits(&_rans, &_ptr, bypass_precision);
                n_bypass += val;
            }
            int32_t raw_val = 0;
            for (int j = 0; j < n_bypass; ++j) {
                val      = Rans64DecGetBits(&_rans, &_ptr, bypass_precision);
                raw_val |= val << (j * bypass_precision);
            }
            value = raw_val >> 1;
            if (raw_val & 1) value = -value - 1;
            else             value += max_val;
        }

        output[i] = value + offset;
    }

    return output;
}

} // namespace ddc
