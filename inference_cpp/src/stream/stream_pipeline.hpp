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
#include <vector>

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
    bool fanout = false;             // --p0 modifier: K independent DPU lanes (own pipeline, no mutex)
    bool lane_major = false;         // --fanout: naive pipeline-major runner creation (baseline; else pinned)
    int threads = 3;                 // worker count for --p0 (= DPU lanes when --fanout)
    bool prefetch = false;           // double-buffer: read row-block N+1 while compressing N (windowed)
    bool neon = false;               // NEON-vectorised normalize/denorm (else scalar libm)
    bool power = false;              // sample INA226/PMBus board power across the compress phase
    std::filesystem::path trace_out; // if set (--trace, --fanout only): dump per-lane stage timeline CSV
    bool verbose = false;
};

// Per-lane timing for --fanout diagnosis (empty unless fanout). Each lane = one worker owning its
// own BenchPipeline on its own DPU core. Placement proxy: compare each lane's g_a ms/call to the
// 1-lane solo — ~1.0x = clean (own core), >~1.25x = that lane's g_a is queued behind another on a
// shared core. A *uniform* inflation across lanes is NOT a DDR/weight-load roof: three g_a colliding
// on one core (--lane-major) gives the same flat signature; the absolute ratio to solo separates them
// (g_a is compute-bound, so a clean <=3-lane split shows ~no inflation). Totals (ms); the reporter
// divides by patches / (2·patches for g_a's two calls) for per-call means.
struct LanePerf {
    int lane = 0;
    long patches = 0;                 // patches this lane processed (dynamic, load-balanced)
    double ga_ms = 0;                 // total in stage_ga (2 g_a DPU calls/patch + interleave)
    double ha_ms = 0, hs_ms = 0;      // SHyp hyperprior DPU stages
    double normalize_ms = 0, entropy_ms = 0;  // CPU stages (contention context)
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
    std::vector<LanePerf> lane_perf;              // per-lane timing (only populated by --fanout)
};

// Compress a whole tile into a .ddc. Throws std::runtime_error on any missing/invalid input.
StreamResult stream_compress_tile(const StreamOptions& opt);

// Approach-A pipeline overlap (stream_p0): K worker threads process patches concurrently. Records are
// placed by patch index, so the .ddc is byte-identical to stream_compress_tile regardless of schedule.
//   default (--p0)        : ONE shared BenchPipeline; the DPU is serialized by a mutex while the A53s
//                           overlap normalize/entropy. Composes with --s1 (g_a channel-parallel).
//   --fanout (--p0 only)  : K INDEPENDENT BenchPipelines, one per worker, DPU mutex dropped — each
//                           lane runs its own g_a on its own DPU core (VART round-robin), so the DPU
//                           runs on up to 3 cores concurrently. N1 data-parallel fan-out; excludes
//                           --s1 (both would contend for the same 3 cores). Fills res.lane_perf.
StreamResult stream_compress_tile_p0(const StreamOptions& opt);

// Decode a .ddc back to per-patch linear-amplitude reconstructions, written as [n,256,256] float32
// (record order). Reuses the existing decompress stages (SHyp: EB->h_s->GC->g_s; FP: EB->g_s).
StreamResult stream_decode_ddc(const StreamOptions& opt);

}  // namespace ddc
