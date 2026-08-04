#pragma once
// stream_pipeline.hpp — sequential streaming compressor (stream_seq).
//
// Reads a [H,W,2] tile, walks the 256x256 patch grid (row-major: azimuth-outer, range-inner;
// stride = 256 - overlap, last patch snapped flush to the edge so the full tile is covered), runs
// the existing BenchPipeline compress stages per patch, and writes a .ddc (ddc_io.hpp). This is the
// baseline before the P0/P2 pipelined executors. See docs/onboard_pipeline.md.

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>

namespace ddc {

struct StreamOptions {
    std::filesystem::path xmodel;    // active_model/*.xmodel
    std::filesystem::path params;    // entropy_params/ directory
    std::filesystem::path manifest;  // manifest.json (model_name -> arch)
    std::filesystem::path tile;      // input tile [H,W,2] float32 (.npy)
    std::filesystem::path out_ddc;   // output: .ddc (compress) or recon .npy (decode)
    std::filesystem::path decode_ddc;  // if set: decode this .ddc instead of compressing a tile
    std::string tile_id;             // empty -> derived from tile filename stem
    int max_rows = -1;               // -1 = all azimuth patch-rows (else cap, for quick tests)
    int overlap = 0;                 // patch overlap px; stride = 256 - overlap. 0 = snap-covered non-overlap
    bool windowed = false;           // stream row-blocks (one patch-row in DDR) vs load whole tile
    bool s1 = false;                 // channel-parallel g_a(real)‖g_a(imag) on two DPU cores
    bool p0 = false;                 // pipeline overlap: K workers, DPU serialized, CPU overlapped
    int threads = 3;                 // worker count for --p0
    bool prefetch = false;           // double-buffer: read row-block N+1 while compressing N (windowed)
    bool neon = false;               // NEON-vectorised normalize/denorm (else scalar libm)
    bool power = false;              // sample INA226/PMBus board power across the compress phase
    bool verbose = false;
};

struct StreamResult {
    int grid_r = 0, grid_a = 0, n_patches = 0;   // range x azimuth patch counts
    uint64_t payload_bytes = 0, file_bytes = 0;
    double bpp = 0.0;
    // per-bucket wall-clock totals (ms)
    double t_read_ms = 0, t_patchify_ms = 0, t_normalize_ms = 0, t_dpu_ms = 0,
           t_entropy_ms = 0, t_write_ms = 0, t_total_ms = 0;
    // power (only when --power and INA226 sensors present)
    bool power_ok = false;
    double avg_power_w = 0.0;                     // MPSoC group (PS+PL) mean over the compress phase
    double energy_j = 0.0;                        // avg_power_w * window duration
    std::map<std::string, double> power_groups;   // mean W per rail-group (DPU_fabric, PS, PL, …)
};

// Compress a whole tile into a .ddc. Throws std::runtime_error on any missing/invalid input.
StreamResult stream_compress_tile(const StreamOptions& opt);

// Approach-A pipeline overlap (stream_p0): K worker threads process patches concurrently; the DPU is
// a serialized resource (mutex) while the A53s overlap normalize/entropy. Records are placed by patch
// index, so the .ddc is byte-identical to stream_compress_tile. Composes with s1 (g_a channel-parallel).
StreamResult stream_compress_tile_p0(const StreamOptions& opt);

// Decode a .ddc back to per-patch linear-amplitude reconstructions, written as [n,256,256] float32
// (record order). Reuses the existing decompress stages (SHyp: EB->h_s->GC->g_s; FP: EB->g_s).
StreamResult stream_decode_ddc(const StreamOptions& opt);

}  // namespace ddc
