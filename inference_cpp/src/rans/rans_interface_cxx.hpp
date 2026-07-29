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

struct RansSymbol {
    uint16_t start;
    uint16_t range;
    bool bypass;
};

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

    void encode_with_indexes(const std::vector<int32_t>& symbols,
                             const std::vector<int32_t>& indexes,
                             const std::vector<std::vector<int32_t>>& cdfs,
                             const std::vector<int32_t>& cdfs_sizes,
                             const std::vector<int32_t>& offsets);

    std::vector<uint8_t> flush();

private:
    std::vector<RansSymbol> _syms;
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
