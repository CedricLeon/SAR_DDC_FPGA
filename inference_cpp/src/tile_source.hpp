#pragma once
// tile_source.hpp — read a large [H,W,C] float tile from the SD card.
//
// Two access modes:
//   load_tile_whole(path)  — reuse npy_load; whole tile in RAM. OK for crops that fit DDR.
//   TileWindowReader       — parse the NPY header once, then seek+read one azimuth row-block at a
//                            time, so DRAM never holds the whole tile. Needed because the full f32
//                            scene (~3.87 GB) does NOT fit the board's ~3 GB free DDR.
//
// Rows are azimuth (slow-time), columns are range (fast-time): the natural streaming axis is
// azimuth, and one row-block spanning the full range = one patch-row. See docs/onboard_pipeline.md §3.
// (Uses std::ifstream::seekg rather than POSIX pread — same effect, matches npy_io.hpp's style.)

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "npy_io.hpp"

namespace ddc {

struct Tile {
    size_t H = 0, W = 0, C = 0;  // azimuth, range, channels (2 = re/im)
    std::vector<float> data;     // [H*W*C] float32, row-major
};

// Whole-tile load (reuses npy_load; casts f32/f64 -> f32). For tiles that fit in DDR.
inline Tile load_tile_whole(const std::string& path) {
    NpyArray a = npy_load(path);
    if (a.shape.size() != 3)
        throw std::runtime_error("load_tile_whole: expected [H,W,C], got ndim=" +
                                 std::to_string(a.shape.size()));
    Tile t;
    t.H = a.shape[0];
    t.W = a.shape[1];
    t.C = a.shape[2];
    t.data = a.to_float32_vec();
    return t;
}

// Windowed reader: header parsed once in the ctor; row-blocks read on demand.
class TileWindowReader {
public:
    explicit TileWindowReader(const std::string& path) : f_(path, std::ios::binary) {
        if (!f_) throw std::runtime_error("TileWindowReader: cannot open " + path);
        char magic[6];
        f_.read(magic, 6);
        if (std::memcmp(magic, "\x93NUMPY", 6) != 0)
            throw std::runtime_error("TileWindowReader: not an .npy: " + path);
        uint8_t major = 0, minor = 0;
        f_.read(reinterpret_cast<char*>(&major), 1);
        f_.read(reinterpret_cast<char*>(&minor), 1);
        size_t header_len = 0, len_field = 0;
        if (major == 1) {
            uint16_t hl = 0;
            f_.read(reinterpret_cast<char*>(&hl), 2);
            header_len = hl;
            len_field = 2;
        } else if (major == 2 || major == 3) {
            uint32_t hl = 0;
            f_.read(reinterpret_cast<char*>(&hl), 4);
            header_len = hl;
            len_field = 4;
        } else {
            throw std::runtime_error("TileWindowReader: unsupported npy version");
        }
        std::string hdr = npy_detail::read_header_str(f_, header_len);
        std::vector<size_t> shape;
        bool fortran = false;
        npy_detail::parse_header(hdr, dtype_, shape, fortran);
        if (fortran) throw std::runtime_error("TileWindowReader: fortran_order unsupported");
        if (shape.size() != 3)
            throw std::runtime_error("TileWindowReader: expected [H,W,C], got ndim=" +
                                     std::to_string(shape.size()));
        H_ = shape[0];
        W_ = shape[1];
        C_ = shape[2];
        if (dtype_ == "<f4" || dtype_ == "float32") elem_ = 4;
        else if (dtype_ == "<i2" || dtype_ == "int16") elem_ = 2;  // raw complex SLC (.cos payload)
        else if (dtype_ == "<f8" || dtype_ == "float64") elem_ = 8;
        else throw std::runtime_error("TileWindowReader: unsupported dtype " + dtype_);
        // Data starts after: 6 (magic) + 2 (major/minor) + len_field + header_len.
        data_offset_ = 6 + 2 + len_field + header_len;
    }

    size_t H() const { return H_; }
    size_t W() const { return W_; }
    size_t C() const { return C_; }

    // Read `nrows` azimuth rows starting at row `row0` -> [nrows*W*C] float32 (casts f64 if needed).
    std::vector<float> read_row_block(size_t row0, size_t nrows) {
        if (row0 + nrows > H_)
            throw std::runtime_error("read_row_block: out of range");
        const size_t n = nrows * W_ * C_;
        const std::streamoff off =
            static_cast<std::streamoff>(data_offset_ + row0 * W_ * C_ * elem_);
        f_.clear();  // clear any prior EOF/fail state before seeking
        f_.seekg(off, std::ios::beg);
        std::vector<float> out(n);
        if (elem_ == 4) {  // f32
            f_.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(n * 4));
        } else if (elem_ == 2) {  // int16 -> f32
            std::vector<int16_t> tmp(n);
            f_.read(reinterpret_cast<char*>(tmp.data()), static_cast<std::streamsize>(n * 2));
            for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(tmp[i]);
        } else {  // f64 -> f32
            std::vector<double> tmp(n);
            f_.read(reinterpret_cast<char*>(tmp.data()), static_cast<std::streamsize>(n * 8));
            for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(tmp[i]);
        }
        if (!f_) throw std::runtime_error("read_row_block: short read");
        return out;
    }

private:
    std::ifstream f_;
    std::string dtype_;
    size_t H_ = 0, W_ = 0, C_ = 0, elem_ = 0, data_offset_ = 0;
};

}  // namespace ddc
