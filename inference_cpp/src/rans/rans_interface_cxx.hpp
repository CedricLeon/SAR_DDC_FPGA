/* rans_interface_cxx.hpp
 * Pybind11-free fork of CompressAI/compressai/cpp_exts/rans/rans_interface.hpp
 *
 * Changes vs original:
 *   - Removed all #include <pybind11/...> headers
 *   - RansEncoderCxx::encode_with_indexes() returns std::vector<uint8_t>
 *     instead of py::bytes
 *   - BufferedRansEncoderCxx::flush() returns std::vector<uint8_t>
 *   - RansDecoderCxx::decode_with_indexes() and set_stream() accept
 *     const std::vector<uint8_t>& instead of const std::string&
 *   - _stream member changed from std::string to std::vector<uint8_t>
 *
 * The underlying rANS64 mathematics (rans64.h) is unchanged.
 */
#pragma once

#include <cstdint>
#include <vector>

// rans64.h uses __int128 which triggers -Wpedantic; suppress for this vendor header only.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include "rans64.h"
#pragma GCC diagnostic pop

namespace ddc {

// One buffered symbol. For a normal (non-bypass) symbol, `code` is an index into
// a precomputed Rans64EncSymbol table (reciprocal-based encoder — see
// build_enc_symbols); for a bypass symbol, `code` is the raw bypass value.
struct RansSymbol {
    uint32_t code;
    bool     bypass;
};

// Precompute the reciprocal encoder symbols for a *static* CDF table (done once
// at model-load, not per patch). enc_syms_out[row_offsets_out[r] + value] is the
// Rans64EncSymbol for symbol `value` (0 .. cdfs_sizes[r]-2) of row r, i.e.
// start = cdf[value], freq = cdf[value+1]-cdf[value]. This lets the flush loop use
// the divide-free Rans64EncPutSymbol; the emitted bytes are identical to the
// divide-based Rans64EncPut path.
void build_enc_symbols(const std::vector<std::vector<int32_t>>& cdfs,
                       const std::vector<int32_t>& cdfs_sizes,
                       std::vector<Rans64EncSymbol>& enc_syms_out,
                       std::vector<int32_t>& row_offsets_out);

// ---------------------------------------------------------------------------
// BufferedRansEncoderCxx
// Accumulates symbols, then flushes to a byte buffer.
// ---------------------------------------------------------------------------
class BufferedRansEncoderCxx {
public:
    BufferedRansEncoderCxx() = default;
    BufferedRansEncoderCxx(const BufferedRansEncoderCxx&)            = delete;
    BufferedRansEncoderCxx(BufferedRansEncoderCxx&&)                 = delete;
    BufferedRansEncoderCxx& operator=(const BufferedRansEncoderCxx&) = delete;
    BufferedRansEncoderCxx& operator=(BufferedRansEncoderCxx&&)      = delete;

    // Fast path: caller supplies a precomputed reciprocal-symbol table (flat,
    // indexed by row_offsets[cdf_idx] + value) that outlives this call + flush().
    void encode_with_indexes(const std::vector<int32_t>& symbols,
                             const std::vector<int32_t>& indexes,
                             const Rans64EncSymbol* enc_syms,
                             const std::vector<int32_t>& enc_row_offsets,
                             const std::vector<int32_t>& cdfs_sizes,
                             const std::vector<int32_t>& offsets);

    // Convenience path (tests / one-off callers): builds the reciprocal-symbol
    // table from a vector-of-vectors CDF, then delegates to the fast path.
    void encode_with_indexes(const std::vector<int32_t>& symbols,
                             const std::vector<int32_t>& indexes,
                             const std::vector<std::vector<int32_t>>& cdfs,
                             const std::vector<int32_t>& cdfs_sizes,
                             const std::vector<int32_t>& offsets);

    std::vector<uint8_t> flush();

private:
    std::vector<RansSymbol>      _syms;
    const Rans64EncSymbol*       _enc_syms = nullptr;  // borrowed; valid through flush()
    std::vector<Rans64EncSymbol> _enc_syms_owned;      // storage for the convenience path
    std::vector<int32_t>         _row_offsets_owned;   // storage for the convenience path
};

// ---------------------------------------------------------------------------
// RansEncoderCxx
// Stateless: one call encodes and flushes.
// ---------------------------------------------------------------------------
class RansEncoderCxx {
public:
    RansEncoderCxx() = default;
    RansEncoderCxx(const RansEncoderCxx&)            = delete;
    RansEncoderCxx(RansEncoderCxx&&)                 = delete;
    RansEncoderCxx& operator=(const RansEncoderCxx&) = delete;
    RansEncoderCxx& operator=(RansEncoderCxx&&)      = delete;

    // Fast path — precomputed reciprocal-symbol table (see build_enc_symbols).
    std::vector<uint8_t> encode_with_indexes(
        const std::vector<int32_t>& symbols,
        const std::vector<int32_t>& indexes,
        const Rans64EncSymbol* enc_syms,
        const std::vector<int32_t>& enc_row_offsets,
        const std::vector<int32_t>& cdfs_sizes,
        const std::vector<int32_t>& offsets);

    // Convenience path — vector-of-vectors CDF (tests / one-off callers).
    std::vector<uint8_t> encode_with_indexes(
        const std::vector<int32_t>& symbols,
        const std::vector<int32_t>& indexes,
        const std::vector<std::vector<int32_t>>& cdfs,
        const std::vector<int32_t>& cdfs_sizes,
        const std::vector<int32_t>& offsets);
};

// ---------------------------------------------------------------------------
// RansDecoderCxx
// Supports both one-shot decode_with_indexes and streaming decode_stream.
// ---------------------------------------------------------------------------
class RansDecoderCxx {
public:
    RansDecoderCxx() = default;
    RansDecoderCxx(const RansDecoderCxx&)            = delete;
    RansDecoderCxx(RansDecoderCxx&&)                 = delete;
    RansDecoderCxx& operator=(const RansDecoderCxx&) = delete;
    RansDecoderCxx& operator=(RansDecoderCxx&&)      = delete;

    std::vector<int32_t> decode_with_indexes(
        const std::vector<uint8_t>& encoded,
        const std::vector<int32_t>& indexes,
        const std::vector<std::vector<int32_t>>& cdfs,
        const std::vector<int32_t>& cdfs_sizes,
        const std::vector<int32_t>& offsets);

    void set_stream(const std::vector<uint8_t>& stream);

    std::vector<int32_t> decode_stream(
        const std::vector<int32_t>& indexes,
        const std::vector<std::vector<int32_t>>& cdfs,
        const std::vector<int32_t>& cdfs_sizes,
        const std::vector<int32_t>& offsets);

private:
    Rans64State           _rans{};
    std::vector<uint8_t>  _stream;
    uint32_t*             _ptr = nullptr;
};

} // namespace ddc
