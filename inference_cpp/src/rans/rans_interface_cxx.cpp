/* rans_interface_cxx.cpp
 * Pybind11-free fork of CompressAI/compressai/cpp_exts/rans/rans_interface.cpp
 *
 * Buffer types vs the original: py::bytes / std::string -> std::vector<uint8_t>.
 *
 * Encode fast path (byte-identical to the original bitstream): the flush loop uses
 * the divide-free Rans64EncPutSymbol with reciprocal symbols precomputed once from
 * the static CDF (build_enc_symbols), instead of the divide-based Rans64EncPut; and
 * _syms is reserved + stores a symbol index rather than pointer-chasing a
 * vector-of-vectors CDF. Decode is unchanged. The rANS64 core (rans64.h) is untouched.
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
// build_enc_symbols — precompute reciprocal encoder symbols from a static CDF.
// Row r contributes cdfs_sizes[r]-1 symbols (values 0 .. cdfs_sizes[r]-2), laid
// out contiguously; row_offsets_out[r] is row r's start. Emitted-byte behaviour
// of Rans64EncPutSymbol is identical to the divide-based Rans64EncPut.
// ---------------------------------------------------------------------------
void build_enc_symbols(const std::vector<std::vector<int32_t>>& cdfs,
                       const std::vector<int32_t>& cdfs_sizes,
                       std::vector<Rans64EncSymbol>& enc_syms_out,
                       std::vector<int32_t>& row_offsets_out)
{
    assert(cdfs.size() == cdfs_sizes.size());

    const size_t rows = cdfs.size();
    row_offsets_out.resize(rows);

    int32_t total = 0;
    for (size_t r = 0; r < rows; ++r) {
        row_offsets_out[r] = total;
        total += cdfs_sizes[r] - 1;  // valid symbol values: 0 .. cdfs_sizes[r]-2
    }

    enc_syms_out.resize(static_cast<size_t>(total));
    for (size_t r = 0; r < rows; ++r) {
        const auto& cdf = cdfs[r];
        const int32_t n_val = cdfs_sizes[r] - 1;
        const int32_t base  = row_offsets_out[r];
        for (int32_t v = 0; v < n_val; ++v) {
            const uint32_t start = static_cast<uint32_t>(cdf[v]);
            const uint32_t freq  = static_cast<uint32_t>(cdf[v + 1] - cdf[v]);
            Rans64EncSymbolInit(&enc_syms_out[base + v], start, freq, precision);
        }
    }
}

// ---------------------------------------------------------------------------
// BufferedRansEncoderCxx::encode_with_indexes — fast path.
// Builds _syms storing, for normal symbols, an index into the precomputed
// reciprocal-symbol table (enc_syms), and for bypass symbols the raw value. No
// per-symbol CDF pointer-chase and no per-symbol divide (that moves to load-time
// build_enc_symbols + Rans64EncPutSymbol in flush).
// ---------------------------------------------------------------------------
void BufferedRansEncoderCxx::encode_with_indexes(
    const std::vector<int32_t>& symbols,
    const std::vector<int32_t>& indexes,
    const Rans64EncSymbol* enc_syms,
    const std::vector<int32_t>& enc_row_offsets,
    const std::vector<int32_t>& cdfs_sizes,
    const std::vector<int32_t>& offsets)
{
    DDC_RPROF_CUR(::ddc::rprof::ST_LOOKUP);  // forward pass (builds _syms)

    _enc_syms = enc_syms;
    _syms.clear();
    _syms.reserve(symbols.size());  // exact for the common (no-bypass) case

    for (size_t i = 0; i < symbols.size(); ++i) {
        const int32_t cdf_idx = indexes[i];
        const int32_t max_value = cdfs_sizes[cdf_idx] - 2;
        assert(max_value >= 0);

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
        _syms.push_back({static_cast<uint32_t>(enc_row_offsets[cdf_idx] + value),
                         false});

        if (value == max_value) {
            int32_t n_bypass = 0;
            while ((raw_val >> (n_bypass * bypass_precision)) != 0) ++n_bypass;

            int32_t val = n_bypass;
            while (val >= static_cast<int32_t>(max_bypass_val)) {
                _syms.push_back({max_bypass_val, true});
                val -= max_bypass_val;
            }
            _syms.push_back({static_cast<uint32_t>(val), true});

            for (int32_t j = 0; j < n_bypass; ++j) {
                const uint32_t bv = (raw_val >> (j * bypass_precision)) & max_bypass_val;
                _syms.push_back({bv, true});
            }
        }
    }
}

// ---------------------------------------------------------------------------
// BufferedRansEncoderCxx::encode_with_indexes — convenience path (vector-of-
// vectors CDF). Builds the reciprocal-symbol table on the fly (kept alive in
// members through flush()), then delegates to the fast path above.
// ---------------------------------------------------------------------------
void BufferedRansEncoderCxx::encode_with_indexes(
    const std::vector<int32_t>& symbols,
    const std::vector<int32_t>& indexes,
    const std::vector<std::vector<int32_t>>& cdfs,
    const std::vector<int32_t>& cdfs_sizes,
    const std::vector<int32_t>& offsets)
{
    assert(cdfs.size() == cdfs_sizes.size());
    build_enc_symbols(cdfs, cdfs_sizes, _enc_syms_owned, _row_offsets_owned);
    encode_with_indexes(symbols, indexes, _enc_syms_owned.data(), _row_offsets_owned,
                        cdfs_sizes, offsets);
}

// ---------------------------------------------------------------------------
// BufferedRansEncoderCxx::flush  — reverse rANS renorm loop.
// Uses the divide-free Rans64EncPutSymbol; output is byte-identical to the
// divide-based Rans64EncPut path.
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
            Rans64EncPutSymbol(&rans, &ptr, &_enc_syms[sym.code], precision);
        } else {
            Rans64EncPutBits(&rans, &ptr, sym.code, bypass_precision);
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
// BufferedRansEncoderCxx::encode_with_indexes_legacy — pre-4ddbcc8 baseline,
// restored verbatim from commit 324744f (4ddbcc8^): per-symbol CDF pointer-chase,
// no precomputed reciprocal table. _syms_legacy stores the raw (start, range)
// directly, consumed by flush_legacy()'s divide-based Rans64EncPut.
// ---------------------------------------------------------------------------
void BufferedRansEncoderCxx::encode_with_indexes_legacy(
    const std::vector<int32_t>& symbols,
    const std::vector<int32_t>& indexes,
    const std::vector<std::vector<int32_t>>& cdfs,
    const std::vector<int32_t>& cdfs_sizes,
    const std::vector<int32_t>& offsets)
{
    assert(cdfs.size() == cdfs_sizes.size());

    DDC_RPROF_CUR(::ddc::rprof::ST_LOOKUP);  // forward CDF-lookup pass (builds _syms_legacy)

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
        _syms_legacy.push_back({static_cast<uint16_t>(cdf[value]),
                                static_cast<uint16_t>(cdf[value + 1] - cdf[value]),
                                false});

        if (value == max_value) {
            int32_t n_bypass = 0;
            while ((raw_val >> (n_bypass * bypass_precision)) != 0) ++n_bypass;

            int32_t val = n_bypass;
            while (val >= static_cast<int32_t>(max_bypass_val)) {
                _syms_legacy.push_back({max_bypass_val, max_bypass_val + 1u, true});
                val -= max_bypass_val;
            }
            _syms_legacy.push_back({static_cast<uint16_t>(val),
                                    static_cast<uint16_t>(val + 1), true});

            for (int32_t j = 0; j < n_bypass; ++j) {
                const int32_t bv = (raw_val >> (j * bypass_precision)) & max_bypass_val;
                _syms_legacy.push_back({static_cast<uint16_t>(bv),
                                        static_cast<uint16_t>(bv + 1), true});
            }
        }
    }
}

// ---------------------------------------------------------------------------
// BufferedRansEncoderCxx::flush_legacy — pre-4ddbcc8 baseline: divide-based
// Rans64EncPut per symbol (restored verbatim from 324744f). Output is
// byte-identical to flush()'s divide-free Rans64EncPutSymbol path.
// ---------------------------------------------------------------------------
std::vector<uint8_t> BufferedRansEncoderCxx::flush_legacy() {
    DDC_RPROF_CUR(::ddc::rprof::ST_FLUSH);  // reverse rANS renorm loop + byte copy

    Rans64State rans;
    Rans64EncInit(&rans);

    std::vector<uint32_t> output(_syms_legacy.size(), 0xCCCCCCCCu);
    uint32_t* ptr = output.data() + output.size();

    while (!_syms_legacy.empty()) {
        const RansSymbolLegacy sym = _syms_legacy.back();
        if (!sym.bypass) {
            Rans64EncPut(&rans, &ptr, sym.start, sym.range, precision);
        } else {
            Rans64EncPutBits(&rans, &ptr, sym.start, bypass_precision);
        }
        _syms_legacy.pop_back();
    }

    Rans64EncFlush(&rans, &ptr);

    const size_t n32 = static_cast<size_t>(
        std::distance(ptr, output.data() + output.size()));
    const uint8_t* byte_ptr = reinterpret_cast<const uint8_t*>(ptr);
    return std::vector<uint8_t>(byte_ptr, byte_ptr + n32 * sizeof(uint32_t));
}

// ---------------------------------------------------------------------------
// RansEncoderCxx::encode_with_indexes — fast path
// ---------------------------------------------------------------------------
std::vector<uint8_t> RansEncoderCxx::encode_with_indexes(
    const std::vector<int32_t>& symbols,
    const std::vector<int32_t>& indexes,
    const Rans64EncSymbol* enc_syms,
    const std::vector<int32_t>& enc_row_offsets,
    const std::vector<int32_t>& cdfs_sizes,
    const std::vector<int32_t>& offsets)
{
    BufferedRansEncoderCxx enc;
    enc.encode_with_indexes(symbols, indexes, enc_syms, enc_row_offsets,
                            cdfs_sizes, offsets);
    return enc.flush();
}

// ---------------------------------------------------------------------------
// RansEncoderCxx::encode_with_indexes — convenience path (vector-of-vectors)
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
// RansEncoderCxx::encode_with_indexes_legacy — pre-4ddbcc8 baseline (stateless)
// ---------------------------------------------------------------------------
std::vector<uint8_t> RansEncoderCxx::encode_with_indexes_legacy(
    const std::vector<int32_t>& symbols,
    const std::vector<int32_t>& indexes,
    const std::vector<std::vector<int32_t>>& cdfs,
    const std::vector<int32_t>& cdfs_sizes,
    const std::vector<int32_t>& offsets)
{
    BufferedRansEncoderCxx enc;
    enc.encode_with_indexes_legacy(symbols, indexes, cdfs, cdfs_sizes, offsets);
    return enc.flush_legacy();
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
