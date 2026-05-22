#pragma once
// Minimal NumPy NPY file loader (format v1.0 and v2.0).
// No external dependencies. Supports float32 and int32 arrays, any shape.
//
// NPY format reference: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
//  - 6 bytes magic:   '\x93NUMPY'
//  - 1 byte major version
//  - 1 byte minor version
//  - 2 bytes (v1) or 4 bytes (v2) header_len (little-endian)
//  - header_len bytes: Python dict literal with 'descr', 'fortran_order', 'shape'
//  - raw data bytes (row-major)

#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ddc {

struct NpyArray {
    std::vector<size_t> shape;
    std::string         dtype;   // e.g. "<f4", "<i4"
    std::vector<uint8_t> data;   // raw bytes in row-major order

    size_t numel() const {
        if (shape.empty()) return 0;
        return std::accumulate(shape.begin(), shape.end(), size_t(1),
                               std::multiplies<size_t>());
    }

    const float*   as_float32() const { return reinterpret_cast<const float*>(data.data()); }
    const int32_t* as_int32()   const { return reinterpret_cast<const int32_t*>(data.data()); }

    // Convert to float32 vector regardless of on-disk dtype (float32 or float64).
    // Use this when the source dtype may be float64 (e.g. sym_Noisy.npy).
    std::vector<float> to_float32_vec() const {
        size_t n = numel();
        if (dtype == "<f4" || dtype == "float32") {
            const float* p = reinterpret_cast<const float*>(data.data());
            return std::vector<float>(p, p + n);
        } else if (dtype == "<f8" || dtype == "float64") {
            const double* p = reinterpret_cast<const double*>(data.data());
            std::vector<float> out(n);
            for (size_t i = 0; i < n; ++i)
                out[i] = static_cast<float>(p[i]);
            return out;
        }
        throw std::runtime_error("NpyArray::to_float32_vec: unsupported dtype: " + dtype);
    }

    // Helpers for common 1-D cases
    std::vector<float> to_float_vec() const {
        assert(dtype == "<f4" || dtype == "float32");
        const float* p = as_float32();
        return std::vector<float>(p, p + numel());
    }
    std::vector<int32_t> to_int32_vec() const {
        assert(dtype == "<i4" || dtype == "int32");
        const int32_t* p = as_int32();
        return std::vector<int32_t>(p, p + numel());
    }
};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
namespace npy_detail {

inline std::string read_header_str(std::istream& in, size_t len) {
    std::string s(len, '\0');
    in.read(&s[0], static_cast<std::streamsize>(len));
    return s;
}

// Very small Python dict parser. Only handles the subset written by numpy:
//   {'descr': '<f4', 'fortran_order': False, 'shape': (H,), }
// or shape: (H, W) etc.
inline void parse_header(const std::string& hdr, std::string& dtype,
                         std::vector<size_t>& shape, bool& fortran_order)
{
    // --- descr ---
    auto pos = hdr.find("'descr'");
    if (pos == std::string::npos) pos = hdr.find("\"descr\"");
    if (pos == std::string::npos) throw std::runtime_error("npy: 'descr' key not found");
    pos = hdr.find('\'', pos + 7);
    if (pos == std::string::npos) pos = hdr.find('"', pos + 7);
    char quote = hdr[pos];
    size_t start = pos + 1;
    size_t end   = hdr.find(quote, start);
    dtype = hdr.substr(start, end - start);

    // --- fortran_order ---
    fortran_order = (hdr.find("True") != std::string::npos &&
                     hdr.find("fortran_order") < hdr.find("True"));

    // --- shape ---
    pos = hdr.find("'shape'");
    if (pos == std::string::npos) pos = hdr.find("\"shape\"");
    if (pos == std::string::npos) throw std::runtime_error("npy: 'shape' key not found");
    pos = hdr.find('(', pos);
    if (pos == std::string::npos) throw std::runtime_error("npy: '(' not found in shape");
    size_t rpos = hdr.find(')', pos);
    std::string shape_str = hdr.substr(pos + 1, rpos - pos - 1);
    shape.clear();
    std::istringstream ss(shape_str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        // trim whitespace
        size_t b = token.find_first_not_of(" \t");
        if (b == std::string::npos) continue;
        size_t e = token.find_last_not_of(" \t");
        std::string tok = token.substr(b, e - b + 1);
        if (!tok.empty()) shape.push_back(static_cast<size_t>(std::stoul(tok)));
    }
}

} // namespace npy_detail

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
inline NpyArray npy_load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("npy_load: cannot open " + path);

    // Magic
    char magic[6];
    f.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0)
        throw std::runtime_error("npy_load: not a valid .npy file: " + path);

    // Version
    uint8_t major = 0, minor = 0;
    f.read(reinterpret_cast<char*>(&major), 1);
    f.read(reinterpret_cast<char*>(&minor), 1);

    // Header length
    size_t header_len = 0;
    if (major == 1) {
        uint16_t hl = 0;
        f.read(reinterpret_cast<char*>(&hl), 2);
        header_len = hl;
    } else if (major == 2 || major == 3) {
        uint32_t hl = 0;
        f.read(reinterpret_cast<char*>(&hl), 4);
        header_len = hl;
    } else {
        throw std::runtime_error("npy_load: unsupported version " +
                                 std::to_string(major) + "." + std::to_string(minor));
    }

    // Parse header
    std::string hdr = npy_detail::read_header_str(f, header_len);
    NpyArray arr;
    bool fortran_order = false;
    npy_detail::parse_header(hdr, arr.dtype, arr.shape, fortran_order);
    if (fortran_order)
        throw std::runtime_error("npy_load: fortran_order=True arrays not supported");

    // Determine element size
    size_t elem_size = 0;
    if (arr.dtype == "<f4" || arr.dtype == "float32") elem_size = 4;
    else if (arr.dtype == "<i4" || arr.dtype == "int32") elem_size = 4;
    else if (arr.dtype == "<i8" || arr.dtype == "int64") elem_size = 8;
    else if (arr.dtype == "<f8" || arr.dtype == "float64") elem_size = 8;
    else throw std::runtime_error("npy_load: unsupported dtype: " + arr.dtype);

    // Read data
    size_t nbytes = arr.numel() * elem_size;
    arr.data.resize(nbytes);
    f.read(reinterpret_cast<char*>(arr.data.data()), static_cast<std::streamsize>(nbytes));
    if (!f) throw std::runtime_error("npy_load: truncated data in " + path);

    return arr;
}

// Convenience: load and return as vector<float>
inline std::vector<float> npy_load_float32(const std::string& path) {
    return npy_load(path).to_float_vec();
}

// Convenience: load and return as vector<int32_t>
inline std::vector<int32_t> npy_load_int32(const std::string& path) {
    return npy_load(path).to_int32_vec();
}

// Save a float32 array (2-D or 1-D) as NPY v1.0
inline void npy_save_float32(const std::string& path,
                              const float* data,
                              const std::vector<size_t>& shape) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("npy_save: cannot open " + path);

    // Build header string
    std::ostringstream hdr;
    hdr << "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i) {
        hdr << shape[i];
        if (i + 1 < shape.size() || shape.size() == 1) hdr << ',';
    }
    hdr << "), }";
    std::string hdr_str = hdr.str();
    // Pad to 64-byte boundary (total header = 10 + header_len)
    size_t pad = 64 - ((10 + hdr_str.size()) % 64);
    hdr_str.append(pad - 1, ' ');
    hdr_str += '\n';

    // Write magic + version + header_len + header
    f.write("\x93NUMPY\x01\x00", 8);
    uint16_t hl = static_cast<uint16_t>(hdr_str.size());
    f.write(reinterpret_cast<const char*>(&hl), 2);
    f.write(hdr_str.data(), static_cast<std::streamsize>(hdr_str.size()));

    // Write data
    size_t n = std::accumulate(shape.begin(), shape.end(), size_t(1),
                               std::multiplies<size_t>());
    f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n * 4));
}

} // namespace ddc
