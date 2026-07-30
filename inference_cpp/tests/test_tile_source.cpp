// test_tile_source.cpp — host test for tile_source.hpp.
//
// Asserts the windowed reader agrees with the trusted whole-tile loader (npy_load), i.e. the NPY
// data offset and per-row seeks are correct. With --formula, also checks the numpy-written values
// match r*1000 + c*10 + ch (validates parsing a genuine numpy header end-to-end).
//
// Host build:  g++ -std=c++17 -I inference_cpp/src inference_cpp/tests/test_tile_source.cpp -o t

#include <cstdio>
#include <string>
#include <vector>

#include "tile_source.hpp"

using namespace ddc;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: test_tile_source <tile.npy> [--formula]\n");
        return 2;
    }
    const std::string path = argv[1];
    const bool check_formula = (argc > 2 && std::string(argv[2]) == "--formula");

    bool ok = true;
#define CHECK(cond)                             \
    do {                                        \
        if (!(cond)) {                          \
            std::printf("  FAIL: %s\n", #cond); \
            ok = false;                         \
        }                                       \
    } while (0)

    Tile whole = load_tile_whole(path);
    TileWindowReader tr(path);
    CHECK(tr.H() == whole.H && tr.W() == whole.W && tr.C() == whole.C);
    const size_t H = whole.H, W = whole.W, C = whole.C;

    // (1) windowed whole-read == npy_load whole-read, element-for-element.
    std::vector<float> wblk = tr.read_row_block(0, H);
    CHECK(wblk.size() == whole.data.size());
    CHECK(wblk == whole.data);

    // (2) a sub-block equals the corresponding slice of the whole tile.
    if (H >= 8) {
        const size_t r0 = 3, nb = 4;
        std::vector<float> sub = tr.read_row_block(r0, nb);
        bool subok = (sub.size() == nb * W * C);
        for (size_t i = 0; i < sub.size() && subok; ++i)
            subok = (sub[i] == whole.data[r0 * W * C + i]);
        CHECK(subok);
    }

    // (3) optional: numpy-written values match the known formula.
    if (check_formula) {
        bool vals = true;
        for (size_t r = 0; r < H && vals; ++r)
            for (size_t c = 0; c < W && vals; ++c)
                for (size_t ch = 0; ch < C && vals; ++ch) {
                    const float exp = static_cast<float>(r * 1000 + c * 10 + ch);
                    if (whole.data[(r * W + c) * C + ch] != exp) vals = false;
                }
        CHECK(vals);
    }

    std::printf(ok ? "test_tile_source: PASS (H=%zu W=%zu C=%zu)\n" : "test_tile_source: FAIL\n",
                H, W, C);
    return ok ? 0 : 1;
}
