// stream_pipeline.cpp — see stream_pipeline.hpp.

#include "stream/stream_pipeline.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "benchmark/bench_pipeline.hpp"
#include "benchmark/power_sampler.hpp"
#include "constants.hpp"
#include "ddc_io.hpp"
#include "tile_source.hpp"

namespace ddc {
namespace {

using clk = std::chrono::steady_clock;

double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

template <class F>
double time_stage(F&& fn) {
    const auto a = clk::now();
    fn();
    return ms(a, clk::now());
}

// FNV-1a 64-bit over the sorted entropy_params/*.npy bytes — the .ddc decodability guard. Canonical
// spec = src/utils/ddc_format.py::params_guard (the ground-side verifier); the constants below are
// the STANDARD FNV-1a-64 offset basis / prime, written in hex to stay eyeball-checkable — a decimal
// typo here (a dropped digit → 0xcbf.../10) once made this non-standard and silently diverge from
// the Python oracle. Cross-checked byte-for-byte against params_guard on real entropy_params.
void fnv1a_params(const std::filesystem::path& dir, uint8_t out[8]) {
    uint64_t h = 0xcbf29ce484222325ULL;  // FNV-1a-64 offset basis (= 14695981039346656037)
    std::vector<std::filesystem::path> files;
    for (const auto& e : std::filesystem::directory_iterator(dir))
        if (e.path().extension() == ".npy") files.push_back(e.path());
    std::sort(files.begin(), files.end());
    for (const auto& f : files) {
        std::ifstream in(f, std::ios::binary);
        char buf[65536];
        while (in.read(buf, sizeof(buf)) || in.gcount() > 0) {
            const std::streamsize n = in.gcount();
            for (std::streamsize i = 0; i < n; ++i) {
                h ^= static_cast<uint8_t>(buf[i]);
                h *= 0x100000001b3ULL;  // FNV-1a-64 prime (= 1099511628211)
            }
        }
    }
    for (int i = 0; i < 8; ++i) out[i] = static_cast<uint8_t>((h >> (8 * i)) & 0xFF);
}

uint8_t arch_id_from_name(const std::string& name) {
    const auto dash = name.find('-');
    const std::string arch = (dash == std::string::npos) ? name : name.substr(0, dash);
    if (arch == "FP") return 0;
    if (arch == "ResFP") return 1;
    if (arch == "SHyp") return 2;
    if (arch == "ResSHyp") return 3;
    throw std::runtime_error("stream: unknown arch in model_name '" + name + "'");
}

// Bounded FIFO of (row-index, row-block) for the double-buffer prefetch: a producer thread reads
// row-blocks in order and pushes; the consumer pops in order and compresses. Capacity caps in-flight
// blocks (2 = double buffer). A blocking pop *is* the efficient wait — no busy-spin pinning an A53.
class RowBlockQueue {
public:
    explicit RowBlockQueue(size_t cap) : cap_(cap) {}
    void push(int pa, std::vector<float>&& blk) {
        std::unique_lock<std::mutex> lk(m_);
        not_full_.wait(lk, [&] { return q_.size() < cap_; });
        q_.emplace_back(pa, std::move(blk));
        not_empty_.notify_one();
    }
    bool pop(int& pa, std::vector<float>& blk) {  // false once closed and drained
        std::unique_lock<std::mutex> lk(m_);
        not_empty_.wait(lk, [&] { return !q_.empty() || closed_; });
        if (q_.empty()) return false;
        pa = q_.front().first;
        blk = std::move(q_.front().second);
        q_.pop_front();
        not_full_.notify_one();
        return true;
    }
    void close() {
        {
            std::lock_guard<std::mutex> lk(m_);
            closed_ = true;
        }
        not_empty_.notify_all();
    }

private:
    size_t cap_;
    std::deque<std::pair<int, std::vector<float>>> q_;
    std::mutex m_;
    std::condition_variable not_full_, not_empty_;
    bool closed_ = false;
};

// ---- shared plumbing (identical across the seq / p0 writers; the per-patch compute loop, which
// ---- differs by schedule, stays inline in each) --------------------------------------------------

// Read the manifest, set model_name, return the arch_id.
uint8_t resolve_arch(const StreamOptions& opt, std::string& model_name) {
    std::ifstream mf(opt.manifest);
    if (!mf) throw std::runtime_error("stream: cannot open manifest " + opt.manifest.string());
    nlohmann::json manifest;
    mf >> manifest;
    model_name = manifest.at("model_name").get<std::string>();
    return arch_id_from_name(model_name);
}

// Open the tile source (whole-tile load, or a windowed row-block reader), set H/W, validate C==2.
// Returns the elapsed open time in ms (header-only for windowed; row reads are timed in the loop).
double open_tile(const StreamOptions& opt, bool windowed, Tile& tile,
                 std::unique_ptr<TileWindowReader>& reader, size_t& H, size_t& W) {
    const auto tread = clk::now();
    if (windowed) {
        reader = std::make_unique<TileWindowReader>(opt.tile.string());
        H = reader->H();
        W = reader->W();
        if (reader->C() != 2) throw std::runtime_error("stream: expected tile [H,W,2]");
    } else {
        tile = load_tile_whole(opt.tile.string());
        H = tile.H;
        W = tile.W;
        if (tile.C != 2) throw std::runtime_error("stream: expected tile [H,W,2]");
    }
    return ms(tread, clk::now());
}

// Patch grid from tile dims; clamp azimuth rows to max_rows when set (>=0). Throws if < one patch.
void compute_grid(size_t H, size_t W, int P, int max_rows, int& grid_a, int& grid_r) {
    grid_a = static_cast<int>(H / P);        // azimuth patch-rows
    grid_r = static_cast<int>(W / P);        // range patch-cols
    if (grid_a == 0 || grid_r == 0) throw std::runtime_error("stream: tile smaller than a patch");
    if (max_rows >= 0 && max_rows < grid_a) grid_a = max_rows;
}

// Copy the [P,P,2] window at (base_row, c0) out of a [.,W,2] source into s.noisy_hwc.
void fill_patch(PatchState& s, const float* src, size_t base_row, size_t c0, int P, size_t W) {
    for (int i = 0; i < P; ++i) {
        const float* row = src + ((base_row + i) * W + c0) * 2;
        std::memcpy(s.noisy_hwc.data() + static_cast<size_t>(i) * P * 2, row,
                    static_cast<size_t>(P) * 2 * sizeof(float));
    }
}

// The .ddc header both writers emit — single source, so seq and p0 can never desync the bytes.
DdcHeader build_ddc_header(const StreamOptions& opt, const std::string& model_name, uint8_t arch_id,
                           size_t H, size_t W, int grid_r, int grid_a, int P) {
    DdcHeader h;
    h.flags = 1;
    h.arch_id = arch_id;
    h.N = static_cast<uint16_t>(C_MAIN);
    h.M = static_cast<uint16_t>(C_HYPER);
    h.patch = static_cast<uint16_t>(P);
    h.stride = static_cast<uint16_t>(P);
    h.scene_H = static_cast<uint32_t>(H);
    h.scene_W = static_cast<uint32_t>(W);
    h.grid_r = static_cast<uint16_t>(grid_r);
    h.grid_a = static_cast<uint16_t>(grid_a);
    h.amp_min = AMP_MIN;
    h.amp_max = AMP_MAX;
    h.eps = EPS;
    fnv1a_params(opt.params, h.params_sha);
    h.tile_id = opt.tile_id;
    h.model_id = model_name;
    return h;
}

// Fill the result's grid/counts/file-size/bpp/total-time (identical tail of both writers).
void finalize_result(StreamResult& res, int grid_r, int grid_a, int n, int P,
                     const StreamOptions& opt, clk::time_point t0) {
    res.grid_r = grid_r;
    res.grid_a = grid_a;
    res.n_patches = n;
    res.file_bytes = std::filesystem::file_size(opt.out_ddc);
    res.bpp = n > 0 ? res.payload_bytes * 8.0 / (static_cast<double>(n) * P * P) : 0.0;
    res.t_total_ms = ms(t0, clk::now());
}

// Stop the power sampler and record the compress-phase power (MPSoC = PS+PL) + per-group means.
void fill_power(StreamResult& res, PowerSampler& ps) {
    ps.stop();
    const PowerResult pr = ps.results();
    res.power_ok = pr.valid;
    res.power_groups = pr.groups;
    res.avg_power_w = pr.groups.count("MPSoC") ? pr.groups.at("MPSoC") : 0.0;
    res.energy_j = res.avg_power_w * pr.duration_s;
}

}  // namespace

StreamResult stream_compress_tile(const StreamOptions& opt) {
    const auto t0 = clk::now();

    std::string model_name;
    const uint8_t arch_id = resolve_arch(opt, model_name);

    BenchPipeline pipe(opt.xmodel, opt.params);  // loads xmodel + entropy tables
    const bool hyper = pipe.uses_hyper();
    pipe.set_neon(opt.neon);
    if (opt.s1) pipe.init_s1();  // channel-parallel g_a(real)‖g_a(imag)

    StreamResult res;
    const int P = 256;
    PowerSampler ps;
    const bool psok = opt.power && ps.init();  // false off-board -> --power cleanly no-ops
    if (psok) ps.start();

    const bool windowed = opt.windowed;  // whole-tile load, or row-block streaming (one row in DDR)
    Tile tile;
    std::unique_ptr<TileWindowReader> reader;
    size_t H = 0, W = 0;
    res.t_read_ms += open_tile(opt, windowed, tile, reader, H, W);

    int grid_a = 0, grid_r = 0;
    compute_grid(H, W, P, opt.max_rows, grid_a, grid_r);

    PatchState s = pipe.make_patch_state(P, P);
    std::vector<DdcRecord> records;
    records.reserve(static_cast<size_t>(grid_a) * grid_r);

    // Compress one block/tile-row of patches, single-threaded, with per-stage timing. Blocks are
    // consumed strictly in order, so appending records row-major stays correct with or without
    // prefetch. (pa unused here — records are appended, not index-placed as in p0.)
    auto process_block = [&](int pa, const float* src, size_t base_row) {
        (void)pa;
        for (int pr = 0; pr < grid_r; ++pr) {
            const size_t c0 = static_cast<size_t>(pr) * P;
            const auto tp = clk::now();
            fill_patch(s, src, base_row, c0, P, W);
            res.t_patchify_ms += ms(tp, clk::now());

            res.t_normalize_ms += time_stage([&] { pipe.stage_normalize(s); });
            res.t_dpu_ms += time_stage([&] { opt.s1 ? pipe.stage_ga_s1(s) : pipe.stage_ga(s); });
            if (hyper) res.t_dpu_ms += time_stage([&] { pipe.stage_ha(s); });

            res.t_entropy_ms += time_stage([&] { pipe.stage_eb_compress(s); });
            DdcRecord rec;
            if (hyper) {
                res.t_entropy_ms += time_stage([&] { pipe.stage_eb_decompress(s); });
                res.t_dpu_ms += time_stage([&] { pipe.stage_hs(s); });
                res.t_entropy_ms += time_stage([&] { pipe.stage_gc_compress(s); });
                rec.z = s.z_bits;
                rec.y = s.y_bits;
            } else {
                rec.y = s.y_bits;  // z stays empty for FP
            }
            res.payload_bytes += rec.z.size() + rec.y.size();
            records.push_back(std::move(rec));
        }
    };

    if (windowed && opt.prefetch) {
        // Double buffer: a producer thread reads row-block N+1 while this thread compresses block N.
        RowBlockQueue q(2);
        double read_ms = 0.0;
        std::thread producer([&] {
            for (int pa = 0; pa < grid_a; ++pa) {
                const auto tr = clk::now();
                auto blk = reader->read_row_block(static_cast<size_t>(pa) * P, P);
                read_ms += ms(tr, clk::now());  // only the producer touches read_ms
                q.push(pa, std::move(blk));
            }
            q.close();
        });
        int pa;
        std::vector<float> blk;
        while (q.pop(pa, blk)) process_block(pa, blk.data(), 0);
        producer.join();
        res.t_read_ms += read_ms;  // raw read cost, now overlapped with compute (hidden in t_total)
    } else {
        std::vector<float> rowblock;  // [P*W*2] when windowed
        for (int pa = 0; pa < grid_a; ++pa) {
            if (windowed) {
                const auto tread = clk::now();
                rowblock = reader->read_row_block(static_cast<size_t>(pa) * P, P);
                res.t_read_ms += ms(tread, clk::now());
                process_block(pa, rowblock.data(), 0);  // block is [P, W, 2]; patch rows are 0..P-1
            } else {
                process_block(pa, tile.data.data(), static_cast<size_t>(pa) * P);
            }
        }
    }

    DdcHeader h = build_ddc_header(opt, model_name, arch_id, H, W, grid_r, grid_a, P);
    const auto tw = clk::now();
    write_ddc(opt.out_ddc.string(), h, records);
    res.t_write_ms = ms(tw, clk::now());

    if (psok) fill_power(res, ps);
    finalize_result(res, grid_r, grid_a, static_cast<int>(records.size()), P, opt, t0);
    return res;
}

StreamResult stream_compress_tile_p0(const StreamOptions& opt) {
    const auto t0 = clk::now();

    std::string model_name;
    const uint8_t arch_id = resolve_arch(opt, model_name);

    BenchPipeline pipe(opt.xmodel, opt.params);
    const bool hyper = pipe.uses_hyper();
    pipe.set_neon(opt.neon);
    if (opt.s1) pipe.init_s1();

    StreamResult res;
    const int P = 256;
    PowerSampler ps;
    const bool psok = opt.power && ps.init();  // false off-board -> --power cleanly no-ops
    if (psok) ps.start();

    const bool windowed = opt.windowed;  // full scene must stream (whole f32 > DDR)
    Tile tile;
    std::unique_ptr<TileWindowReader> reader;
    size_t H = 0, W = 0;
    res.t_read_ms += open_tile(opt, windowed, tile, reader, H, W);

    int grid_a = 0, grid_r = 0;
    compute_grid(H, W, P, opt.max_rows, grid_a, grid_r);
    const int n = grid_a * grid_r;

    std::vector<DdcRecord> records(static_cast<size_t>(n));
    const int K = std::max(1, opt.threads);
    std::vector<PatchState> states;
    states.reserve(static_cast<size_t>(K));
    for (int k = 0; k < K; ++k) states.push_back(pipe.make_patch_state(P, P));
    std::mutex dpu_mtx;

    // Per-patch compress (s.noisy_hwc already filled) -> records[idx]. CPU stages run concurrently
    // across workers; the DPU bursts are serialized by dpu_mtx.
    auto process_patch = [&](PatchState& s, int idx) {
        pipe.stage_normalize(s);
        {
            std::lock_guard<std::mutex> lk(dpu_mtx);
            if (opt.s1) pipe.stage_ga_s1(s); else pipe.stage_ga(s);
            if (hyper) pipe.stage_ha(s);
        }
        pipe.stage_eb_compress(s);
        DdcRecord rec;
        if (hyper) {
            pipe.stage_eb_decompress(s);
            { std::lock_guard<std::mutex> lk(dpu_mtx); pipe.stage_hs(s); }
            pipe.stage_gc_compress(s);
            rec.z = s.z_bits;
            rec.y = s.y_bits;
        } else {
            rec.y = s.y_bits;
        }
        records[static_cast<size_t>(idx)] = std::move(rec);  // distinct slot: no lock
    };

    // Compress one row-block: K workers pull patches (pr) off an atomic counter; the DPU bursts are
    // serialized inside process_patch and records placed by index. Used by both windowed paths.
    auto process_block = [&](int pa, const float* block) {
        std::atomic<int> pr_next{0};
        auto blockworker = [&](int wid) {
            PatchState& s = states[static_cast<size_t>(wid)];
            int pr;
            while ((pr = pr_next.fetch_add(1)) < grid_r) {
                fill_patch(s, block, 0, static_cast<size_t>(pr) * P, P, W);
                process_patch(s, pa * grid_r + pr);
            }
        };
        std::vector<std::thread> pool;
        for (int k = 0; k < K; ++k) pool.emplace_back(blockworker, k);
        for (auto& t : pool) t.join();
    };

    if (windowed && opt.prefetch) {
        // Double buffer: a producer thread reads row-block N+1 while the K workers compress block N.
        RowBlockQueue q(2);
        double read_ms = 0.0;
        std::thread producer([&] {
            for (int pa = 0; pa < grid_a; ++pa) {
                const auto tr = clk::now();
                auto blk = reader->read_row_block(static_cast<size_t>(pa) * P, P);
                read_ms += ms(tr, clk::now());  // only the producer touches read_ms
                q.push(pa, std::move(blk));
            }
            q.close();
        });
        int pa;
        std::vector<float> blk;
        while (q.pop(pa, blk)) process_block(pa, blk.data());
        producer.join();
        res.t_read_ms += read_ms;  // raw read cost, now overlapped with compute (hidden in t_total)
    } else if (windowed) {
        // Outer loop over row-blocks (one in DDR at a time); K workers parallelize each block.
        std::vector<float> rowblock;
        for (int pa = 0; pa < grid_a; ++pa) {
            const auto tread = clk::now();
            rowblock = reader->read_row_block(static_cast<size_t>(pa) * P, P);
            res.t_read_ms += ms(tread, clk::now());
            process_block(pa, rowblock.data());
        }
    } else {
        std::atomic<int> next_idx{0};
        auto worker = [&](int wid) {
            PatchState& s = states[static_cast<size_t>(wid)];
            int idx;
            while ((idx = next_idx.fetch_add(1)) < n) {
                const int pa = idx / grid_r, pr = idx % grid_r;
                fill_patch(s, tile.data.data(), static_cast<size_t>(pa) * P,
                           static_cast<size_t>(pr) * P, P, W);
                process_patch(s, idx);
            }
        };
        std::vector<std::thread> pool;
        for (int k = 0; k < K; ++k) pool.emplace_back(worker, k);
        for (auto& t : pool) t.join();
    }

    for (const auto& r : records) res.payload_bytes += r.z.size() + r.y.size();

    DdcHeader h = build_ddc_header(opt, model_name, arch_id, H, W, grid_r, grid_a, P);
    const auto tw = clk::now();
    write_ddc(opt.out_ddc.string(), h, records);
    res.t_write_ms = ms(tw, clk::now());

    if (psok) fill_power(res, ps);
    finalize_result(res, grid_r, grid_a, n, P, opt, t0);  // per-stage buckets overlap in p0 -> 0
    return res;
}

StreamResult stream_decode_ddc(const StreamOptions& opt) {
    const auto t0 = clk::now();

    const auto tr = clk::now();
    DdcFile f = read_ddc(opt.decode_ddc.string());
    StreamResult res;
    res.t_read_ms = ms(tr, clk::now());

    BenchPipeline pipe(opt.xmodel, opt.params);
    const bool hyper = pipe.uses_hyper();
    pipe.set_neon(opt.neon);

    // Decodability guards — refuse a model/params mismatch loudly instead of decoding to garbage
    // (errors over silent fallbacks). arch_id {2,3} = SHyp/ResSHyp carry a hyperprior; the exact
    // params_sha (FNV-1a-64 over entropy_params) also separates same-class models, e.g. SHyp vs
    // ResSHyp. Both pass trivially when decoding with the model/params that wrote the file.
    const bool hdr_hyper = (f.header.arch_id == 2 || f.header.arch_id == 3);
    if (hdr_hyper != hyper)
        throw std::runtime_error("stream_decode: arch mismatch — .ddc arch_id=" +
                                 std::to_string(f.header.arch_id) + " but --xmodel is a " +
                                 (hyper ? "hyperprior" : "factorized") + " model");
    uint8_t sha[8];
    fnv1a_params(opt.params, sha);
    if (std::memcmp(sha, f.header.params_sha, 8) != 0)
        throw std::runtime_error(
            "stream_decode: params_sha mismatch — --params do not match the CDF tables the .ddc "
            "was written with");

    const int P = 256;
    PatchState s = pipe.make_patch_state(P, P);

    const int n = static_cast<int>(f.records.size());
    std::vector<float> out(static_cast<size_t>(n) * P * P);  // [n, P, P] linA, record order
    for (int i = 0; i < n; ++i) {
        const DdcRecord& rec = f.records[i];
        res.payload_bytes += rec.z.size() + rec.y.size();
        if (hyper) {
            s.z_bits = rec.z;
            res.t_entropy_ms += time_stage([&] { pipe.stage_eb_decompress(s); });
            res.t_dpu_ms += time_stage([&] { pipe.stage_hs(s); });
            s.y_bits = rec.y;
            res.t_entropy_ms += time_stage([&] { pipe.stage_gc_decompress(s); });
        } else {
            s.y_bits = rec.y;
            res.t_entropy_ms += time_stage([&] { pipe.stage_eb_decompress(s); });
        }
        res.t_dpu_ms += time_stage([&] { pipe.stage_gs(s); });
        res.t_normalize_ms += time_stage([&] { pipe.stage_denorm(s); });  // (denorm bucketed here)
        std::memcpy(out.data() + static_cast<size_t>(i) * P * P, s.recon_lina.data(),
                    static_cast<size_t>(P) * P * sizeof(float));
    }

    const auto tw = clk::now();
    npy_save_float32(opt.out_ddc.string(), out.data(),
                     {static_cast<size_t>(n), static_cast<size_t>(P), static_cast<size_t>(P)});
    res.t_write_ms = ms(tw, clk::now());

    res.grid_r = f.header.grid_r;
    res.grid_a = f.header.grid_a;
    res.n_patches = n;
    res.file_bytes = std::filesystem::file_size(opt.out_ddc);
    res.bpp = n > 0 ? res.payload_bytes * 8.0 / (static_cast<double>(n) * P * P) : 0.0;
    res.t_total_ms = ms(t0, clk::now());
    return res;
}

}  // namespace ddc
