// dump_latents.cpp — one-off debug tool: run BenchPipeline's compress stages (normalize -> g_a ->
// [h_a]) on a handful of named patches and dump the pre-entropy-coding y/z float latents to .npy,
// for a host-side comparison against the FP32 (Jetson ddc-edge) pipeline's latents on the same
// patches. Not part of the production inference/streaming path — a standalone tool reusing
// BenchPipeline, same pattern as stream_pipeline/benchmark_hardware.
//
// Usage:
//   dump_latents --xmodel <m.xmodel> --params <entropy_params> --tile <tile.npy> \
//       --patches "r0:c0,r1:c1,..." --out-dir <dir>
//
// Writes, per patch (r,c): <out-dir>/y_r<r>_c<c>.npy (always) and z_r<r>_c<c>.npy (hyperprior archs
// only). Each is a flat float32 array (channel/spatial order not preserved — a histogram comparison
// doesn't need it; see docs/tmp_jetson_orin_overnight.md for why this was built).

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "benchmark/bench_pipeline.hpp"
#include "npy_io.hpp"
#include "tile_source.hpp"

namespace {

struct RC {
    int r, c;
};

std::vector<RC> parse_patches(const std::string& spec) {
    std::vector<RC> out;
    std::stringstream ss(spec);
    std::string pair;
    while (std::getline(ss, pair, ',')) {
        auto colon = pair.find(':');
        if (colon == std::string::npos) throw std::runtime_error("bad --patches entry: " + pair);
        out.push_back({std::stoi(pair.substr(0, colon)), std::stoi(pair.substr(colon + 1))});
    }
    return out;
}

// Copy the [P,P,2] window at (row0, col0) out of a [.,W,2] tile into dst (matches
// stream_pipeline.cpp's fill_patch — kept local since that one isn't exported).
void fill_patch(std::vector<float>& dst, const ddc::Tile& tile, int row0, int col0, int P) {
    for (int i = 0; i < P; ++i) {
        const float* row = tile.data.data() + ((static_cast<size_t>(row0) + i) * tile.W + col0) * 2;
        std::memcpy(dst.data() + static_cast<size_t>(i) * P * 2, row,
                    static_cast<size_t>(P) * 2 * sizeof(float));
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string xmodel, params, tile_path, patches_spec, out_dir = ".";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return std::string(argv[++i]);
        };
        if (a == "--xmodel") xmodel = next();
        else if (a == "--params") params = next();
        else if (a == "--tile") tile_path = next();
        else if (a == "--patches") patches_spec = next();
        else if (a == "--out-dir") out_dir = next();
        else if (a == "-h" || a == "--help") {
            std::printf(
                "usage: dump_latents --xmodel <m.xmodel> --params <entropy_params> --tile <tile.npy> "
                "--patches \"r0:c0,r1:c1,...\" --out-dir <dir>\n");
            return 0;
        } else throw std::runtime_error("unknown argument: " + a);
    }
    if (xmodel.empty() || params.empty() || tile_path.empty() || patches_spec.empty())
        throw std::runtime_error("--xmodel, --params, --tile, --patches are all required");

    std::filesystem::create_directories(out_dir);
    const auto patches = parse_patches(patches_spec);
    const int P = 256;

    std::printf("loading model...\n");
    ddc::BenchPipeline pipe(xmodel, params);
    const bool hyper = pipe.uses_hyper();
    ddc::PatchState s = pipe.make_patch_state(P, P);

    std::printf("loading tile %s...\n", tile_path.c_str());
    const ddc::Tile tile = ddc::load_tile_whole(tile_path);
    std::printf("tile [%zu x %zu x %zu], hyper=%d, %zu patches to dump\n", tile.H, tile.W, tile.C,
                hyper, patches.size());

    for (const auto& rc : patches) {
        if (static_cast<size_t>(rc.r) + P > tile.H || static_cast<size_t>(rc.c) + P > tile.W)
            throw std::runtime_error("patch (" + std::to_string(rc.r) + "," + std::to_string(rc.c) +
                                      ") out of bounds for tile " + std::to_string(tile.H) + "x" +
                                      std::to_string(tile.W));
        fill_patch(s.noisy_hwc, tile, rc.r, rc.c, P);
        pipe.stage_normalize(s);
        pipe.stage_ga(s);

        const std::string tag = "r" + std::to_string(rc.r) + "_c" + std::to_string(rc.c);
        const std::string y_path = out_dir + "/y_" + tag + ".npy";
        ddc::npy_save_float32(y_path, s.y.data(), {s.y.size()});
        std::printf("  patch (%d,%d): y -> %s (%zu floats)\n", rc.r, rc.c, y_path.c_str(),
                    s.y.size());

        if (hyper) {
            pipe.stage_ha(s);
            const std::string z_path = out_dir + "/z_" + tag + ".npy";
            ddc::npy_save_float32(z_path, s.z.data(), {s.z.size()});
            std::printf("  patch (%d,%d): z -> %s (%zu floats)\n", rc.r, rc.c, z_path.c_str(),
                        s.z.size());
        }
    }
    std::printf("done.\n");
    return 0;
}
