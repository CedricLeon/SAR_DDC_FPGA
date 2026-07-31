// main_stream.cpp — CLI for the sequential streaming compressor (stream_pipeline binary).
//
// Example:
//   stream_pipeline --xmodel active_model/*.xmodel --params active_model/entropy_params
//                   --tile <tile>.npy --out out.ddc [--manifest manifest.json] [--max-rows N]

#include <cstdio>
#include <stdexcept>
#include <string>

#include "stream/stream_pipeline.hpp"

static const char* USAGE =
    "usage: stream_pipeline --xmodel <m.xmodel> --params <entropy_params> --tile <tile.npy> "
    "--out <out.ddc>\n"
    "                       [--manifest <manifest.json>] [--tile-id <name>] [--max-rows N]\n"
    "                       [--windowed] [--s1] [--p0] [--threads N] [--prefetch] [--verbose]";

int main(int argc, char** argv) {
    ddc::StreamOptions o;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
                return std::string(argv[++i]);
            };
            if (a == "--xmodel") o.xmodel = next();
            else if (a == "--params") o.params = next();
            else if (a == "--manifest") o.manifest = next();
            else if (a == "--tile") o.tile = next();
            else if (a == "--out") o.out_ddc = next();
            else if (a == "--decode") o.decode_ddc = next();
            else if (a == "--tile-id") o.tile_id = next();
            else if (a == "--max-rows") o.max_rows = std::stoi(next());
            else if (a == "--windowed") o.windowed = true;
            else if (a == "--s1") o.s1 = true;
            else if (a == "--p0") o.p0 = true;
            else if (a == "--threads") o.threads = std::stoi(next());
            else if (a == "--prefetch") o.prefetch = true;
            else if (a == "--verbose") o.verbose = true;
            else if (a == "-h" || a == "--help") { std::printf("%s\n", USAGE); return 0; }
            else throw std::runtime_error("unknown argument: " + a);
        }

        const bool decoding = !o.decode_ddc.empty();
        if (o.manifest.empty() && !o.xmodel.empty())
            o.manifest = o.xmodel.parent_path() / "manifest.json";
        if (o.tile_id.empty() && !o.tile.empty()) o.tile_id = o.tile.stem().string();

        const bool missing = o.xmodel.empty() || o.params.empty() || o.out_ddc.empty() ||
                             (decoding ? o.decode_ddc.empty() : o.tile.empty());
        if (missing) {
            std::fprintf(stderr, "%s\n", USAGE);
            return 2;
        }
        if (o.prefetch && (decoding || !o.windowed))
            throw std::runtime_error("--prefetch requires --windowed compression");

        const ddc::StreamResult r =
            decoding ? ddc::stream_decode_ddc(o)
                     : (o.p0 ? ddc::stream_compress_tile_p0(o) : ddc::stream_compress_tile(o));
        if (decoding) {
            std::printf("stream_pipeline decode: %d patches -> %s (%llu B)\n", r.n_patches,
                        o.out_ddc.string().c_str(), static_cast<unsigned long long>(r.file_bytes));
        } else {
            const double thr = r.t_total_ms > 0 ? r.n_patches * 1000.0 / r.t_total_ms : 0.0;
            const char* mode = o.p0 ? "p0" : (o.s1 ? "s1" : "seq");
            std::printf(
                "stream_pipeline [%s%s]: %d patches (%d x %d) | bpp=%.4f | %.2f patch/s | total=%.1f s\n",
                mode, (o.p0 && o.s1) ? "+s1" : "", r.n_patches, r.grid_a, r.grid_r, r.bpp, thr,
                r.t_total_ms / 1000.0);
            if (o.p0)
                std::printf("  [p0] threads=%d ; read=%.1f write=%.1f ms (compute stages overlapped)\n",
                            o.threads, r.t_read_ms, r.t_write_ms);
            else
                std::printf("  timing(ms): read=%.1f patchify=%.1f normalize=%.1f dpu=%.1f "
                            "entropy=%.1f write=%.1f\n",
                            r.t_read_ms, r.t_patchify_ms, r.t_normalize_ms, r.t_dpu_ms,
                            r.t_entropy_ms, r.t_write_ms);
            if (o.prefetch)
                std::printf("  [prefetch] row-block N+1 read overlapped with compress of N\n");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "stream_pipeline error: %s\n", e.what());
        return 1;
    }
    return 0;
}
