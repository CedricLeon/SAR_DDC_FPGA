#pragma once
// ddc_io.hpp — C++ writer/reader for the .ddc downlink container (v1).
//
// Byte-for-byte compatible with src/utils/ddc_format.py (the Python oracle). Little-endian,
// positional. All fields are written EXPLICITLY (never a raw struct dump) so the on-disk layout
// never depends on struct padding or host endianness. See docs/onboard_pipeline.md §7.
//
// HEADER: magic "DDC1" | flags u8 | arch_id u8 | N u16 | M u16 | patch u16 | stride u16 |
//         scene_H u32 | scene_W u32 | grid_r u16 | grid_a u16 | amp_min f32 | amp_max f32 |
//         eps f32 | params_sha[8] | tile_id_len u16 | tile_id | model_id_len u16 | model_id
// BODY x (grid_r*grid_a), row-major: len_z u32 | z | len_y u32 | y   (FP: len_z = 0)
// TRAILER (if flags&1): (grid_r*grid_a) x u64 record offsets; located at filesize - n*8.

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace ddc {

struct DdcHeader {
    uint8_t  flags   = 1;   // bit0 = offset index present
    uint8_t  arch_id = 0;   // 0 FP, 1 ResFP, 2 SHyp, 3 ResSHyp
    uint16_t N = 128, M = 256;
    uint16_t patch = 256, stride = 256;
    uint32_t scene_H = 0, scene_W = 0;
    uint16_t grid_r = 0, grid_a = 0;
    float    amp_min = 0.f, amp_max = 0.f, eps = 0.f;
    uint8_t  params_sha[8] = {0};
    std::string tile_id;
    std::string model_id;

    size_t n_patches() const { return static_cast<size_t>(grid_r) * grid_a; }
    bool   has_index() const { return (flags & 1u) != 0; }
};

struct DdcRecord {
    std::vector<uint8_t> z;  // hyper stream (empty for FactorizedPrior)
    std::vector<uint8_t> y;  // main stream
};

struct DdcFile {
    DdcHeader header;
    std::vector<DdcRecord> records;
};

namespace ddc_detail {

inline void put_u8(std::vector<uint8_t>& b, uint8_t v) { b.push_back(v); }
inline void put_u16(std::vector<uint8_t>& b, uint16_t v) {
    b.push_back(uint8_t(v & 0xFF));
    b.push_back(uint8_t((v >> 8) & 0xFF));
}
inline void put_u32(std::vector<uint8_t>& b, uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back(uint8_t((v >> (8 * i)) & 0xFF));
}
inline void put_u64(std::vector<uint8_t>& b, uint64_t v) {
    for (int i = 0; i < 8; ++i) b.push_back(uint8_t((v >> (8 * i)) & 0xFF));
}
inline void put_f32(std::vector<uint8_t>& b, float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    put_u32(b, u);
}
inline void put_bytes(std::vector<uint8_t>& b, const void* p, size_t n) {
    const uint8_t* q = static_cast<const uint8_t*>(p);
    b.insert(b.end(), q, q + n);
}

inline uint8_t get_u8(const std::vector<uint8_t>& b, size_t& c) { return b.at(c++); }
inline uint16_t get_u16(const std::vector<uint8_t>& b, size_t& c) {
    uint16_t v = uint16_t(b.at(c)) | (uint16_t(b.at(c + 1)) << 8);
    c += 2;
    return v;
}
inline uint32_t get_u32(const std::vector<uint8_t>& b, size_t& c) {
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= uint32_t(b.at(c + i)) << (8 * i);
    c += 4;
    return v;
}
inline uint64_t get_u64(const std::vector<uint8_t>& b, size_t& c) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= uint64_t(b.at(c + i)) << (8 * i);
    c += 8;
    return v;
}
inline float get_f32(const std::vector<uint8_t>& b, size_t& c) {
    uint32_t u = get_u32(b, c);
    float v;
    std::memcpy(&v, &u, 4);
    return v;
}

// Bounds guard for a variable-length read of `n` bytes at offset `c` into a buffer of `size` bytes.
// Throws instead of letting a malformed/truncated .ddc drive an out-of-bounds copy. Written without
// `c + n` so it cannot itself overflow (c <= size always holds here, as the length fields that
// precede each copy are read via bounds-checked get_u* / .at()).
inline void need_bytes(size_t c, size_t n, size_t size, const char* what) {
    if (c > size || n > size - c)
        throw std::runtime_error(std::string("read_ddc: truncated, need ") + std::to_string(n) +
                                 " bytes for " + what);
}

}  // namespace ddc_detail

// Serialize a .ddc to `path`. Throws std::runtime_error on record/grid mismatch or I/O failure.
inline void write_ddc(const std::string& path, const DdcHeader& h,
                      const std::vector<DdcRecord>& records) {
    using namespace ddc_detail;
    if (records.size() != h.n_patches())
        throw std::runtime_error("write_ddc: record count != grid_r*grid_a");

    std::vector<uint8_t> buf;
    const char magic[4] = {'D', 'D', 'C', '1'};
    put_bytes(buf, magic, 4);
    put_u8(buf, h.flags);
    put_u8(buf, h.arch_id);
    put_u16(buf, h.N);
    put_u16(buf, h.M);
    put_u16(buf, h.patch);
    put_u16(buf, h.stride);
    put_u32(buf, h.scene_H);
    put_u32(buf, h.scene_W);
    put_u16(buf, h.grid_r);
    put_u16(buf, h.grid_a);
    put_f32(buf, h.amp_min);
    put_f32(buf, h.amp_max);
    put_f32(buf, h.eps);
    put_bytes(buf, h.params_sha, 8);
    put_u16(buf, static_cast<uint16_t>(h.tile_id.size()));
    put_bytes(buf, h.tile_id.data(), h.tile_id.size());
    put_u16(buf, static_cast<uint16_t>(h.model_id.size()));
    put_bytes(buf, h.model_id.data(), h.model_id.size());

    std::vector<uint64_t> offsets;
    offsets.reserve(records.size());
    for (const auto& r : records) {
        offsets.push_back(buf.size());  // record start = current byte position
        put_u32(buf, static_cast<uint32_t>(r.z.size()));
        put_bytes(buf, r.z.data(), r.z.size());
        put_u32(buf, static_cast<uint32_t>(r.y.size()));
        put_bytes(buf, r.y.data(), r.y.size());
    }
    if (h.has_index())
        for (uint64_t off : offsets) put_u64(buf, off);

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("write_ddc: cannot open " + path);
    f.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
    if (!f) throw std::runtime_error("write_ddc: write failed for " + path);
}

// Read a whole .ddc (header + all records). Throws on malformed input.
inline DdcFile read_ddc(const std::string& path) {
    using namespace ddc_detail;
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("read_ddc: cannot open " + path);
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());
    if (buf.size() < 46 || std::memcmp(buf.data(), "DDC1", 4) != 0)
        throw std::runtime_error("read_ddc: not a DDC file: " + path);

    size_t c = 4;
    DdcFile out;
    DdcHeader& h = out.header;
    h.flags = get_u8(buf, c);
    h.arch_id = get_u8(buf, c);
    h.N = get_u16(buf, c);
    h.M = get_u16(buf, c);
    h.patch = get_u16(buf, c);
    h.stride = get_u16(buf, c);
    h.scene_H = get_u32(buf, c);
    h.scene_W = get_u32(buf, c);
    h.grid_r = get_u16(buf, c);
    h.grid_a = get_u16(buf, c);
    h.amp_min = get_f32(buf, c);
    h.amp_max = get_f32(buf, c);
    h.eps = get_f32(buf, c);
    std::memcpy(h.params_sha, buf.data() + c, 8);
    c += 8;
    uint16_t tlen = get_u16(buf, c);
    need_bytes(c, tlen, buf.size(), "tile_id");
    h.tile_id.assign(reinterpret_cast<const char*>(buf.data() + c), tlen);
    c += tlen;
    uint16_t mlen = get_u16(buf, c);
    need_bytes(c, mlen, buf.size(), "model_id");
    h.model_id.assign(reinterpret_cast<const char*>(buf.data() + c), mlen);
    c += mlen;

    // Cap the reservation: each record is >= 8 bytes (len_z + len_y), so a header claiming more
    // patches than the remaining bytes could hold is corrupt — reject before a huge reserve().
    if (h.n_patches() > (buf.size() - c) / 8)
        throw std::runtime_error("read_ddc: header claims more patches than the file can hold");
    out.records.reserve(h.n_patches());
    for (size_t i = 0; i < h.n_patches(); ++i) {
        uint32_t lz = get_u32(buf, c);
        need_bytes(c, lz, buf.size(), "z stream");
        DdcRecord r;
        r.z.assign(buf.begin() + c, buf.begin() + c + lz);
        c += lz;
        uint32_t ly = get_u32(buf, c);
        need_bytes(c, ly, buf.size(), "y stream");
        r.y.assign(buf.begin() + c, buf.begin() + c + ly);
        c += ly;
        out.records.push_back(std::move(r));
    }
    return out;
}

}  // namespace ddc
