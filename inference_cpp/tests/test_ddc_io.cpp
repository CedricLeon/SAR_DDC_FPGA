// test_ddc_io.cpp — host unit test for the .ddc container codec (ddc_io.hpp).
//
// Builds a deterministic synthetic .ddc, round-trips it in C++, and asserts every field/record.
// If given an output path (argv[1]), also leaves the file there so the Python oracle
// (tests/ddc_cross_check.py) can decode it and confirm C++ and Python agree byte-for-byte.
//
// Host build:  g++ -std=c++17 -I inference_cpp/src inference_cpp/tests/test_ddc_io.cpp -o test_ddc_io

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ddc_io.hpp"

using namespace ddc;

// Synthetic fixture — MUST stay in sync with tests/ddc_cross_check.py.
static DdcHeader make_header() {
    DdcHeader h;
    h.flags = 1;
    h.arch_id = 3;  // ResSHyp
    h.N = 128;
    h.M = 256;
    h.patch = 256;
    h.stride = 256;
    h.scene_H = 512;
    h.scene_W = 768;
    h.grid_r = 3;
    h.grid_a = 2;  // n = 6
    h.amp_min = 4.605170249938965f;
    h.amp_max = 10.742239952087402f;
    h.eps = 0.01f;
    for (int i = 0; i < 8; ++i) h.params_sha[i] = static_cast<uint8_t>(i);
    h.tile_id = "TEST_TILE";
    h.model_id = "ResSHyp-relu_s0_L1000_pt";
    return h;
}

static std::vector<DdcRecord> make_records() {
    std::vector<DdcRecord> recs;
    for (int i = 0; i < 6; ++i) {
        DdcRecord r;
        r.z.assign(static_cast<size_t>(i), static_cast<uint8_t>(0xA0 + i));       // rec 0 has empty z
        r.y.assign(static_cast<size_t>(i + 1), static_cast<uint8_t>(0xB0 + i));
        recs.push_back(std::move(r));
    }
    return recs;
}

int main(int argc, char** argv) {
    DdcHeader h = make_header();
    std::vector<DdcRecord> recs = make_records();
    std::string path = (argc > 1) ? argv[1] : "/tmp/test_ddc_io.ddc";

    write_ddc(path, h, recs);
    DdcFile f = read_ddc(path);

    bool ok = true;
#define CHECK(cond)                                    \
    do {                                               \
        if (!(cond)) {                                 \
            std::printf("  FAIL: %s\n", #cond);        \
            ok = false;                                \
        }                                              \
    } while (0)

    CHECK(f.header.arch_id == h.arch_id);
    CHECK(f.header.N == h.N && f.header.M == h.M);
    CHECK(f.header.patch == h.patch && f.header.stride == h.stride);
    CHECK(f.header.scene_H == h.scene_H && f.header.scene_W == h.scene_W);
    CHECK(f.header.grid_r == h.grid_r && f.header.grid_a == h.grid_a);
    CHECK(f.header.amp_min == h.amp_min);
    CHECK(f.header.amp_max == h.amp_max);
    CHECK(f.header.eps == h.eps);
    CHECK(std::memcmp(f.header.params_sha, h.params_sha, 8) == 0);
    CHECK(f.header.tile_id == h.tile_id);
    CHECK(f.header.model_id == h.model_id);
    CHECK(f.records.size() == recs.size());
    for (size_t i = 0; i < recs.size() && i < f.records.size(); ++i) {
        CHECK(f.records[i].z == recs[i].z);
        CHECK(f.records[i].y == recs[i].y);
    }

    if (ok)
        std::printf("test_ddc_io: PASS (wrote %s)\n", path.c_str());
    else
        std::printf("test_ddc_io: FAIL\n");
    return ok ? 0 : 1;
}
