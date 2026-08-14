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

// One row on the --fanout timeline (CSV): who ran what, from when to when (ms, relative to the
// start of the compress phase). lane -1 = the prefetch reader thread; patch = global patch index
// (row-block index for reads). Consumed by scripts/fpga/benchmark/fanout_gantt.py.
struct TraceEvent {
    int lane;
    int patch;
    const char* stage;
    double t0_ms, t1_ms;
};

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

// Patch top-left offsets along one axis: stride-spaced from 0, with the LAST patch snapped to
// (dim - P) so the FULL extent is covered — no dropped edge sliver. Returns {} if dim < P (caller
// throws). This snap rule is the single source of truth; the host stitcher mirrors it exactly
// (docs/onboard_pipeline.md §10 "grid rule").
std::vector<int> make_offsets(size_t dim, int P, int stride) {
    const int d = static_cast<int>(dim);
    if (d < P) return {};
    std::vector<int> offs;
    for (int o = 0; o + P <= d; o += stride) offs.push_back(o);
    if (offs.back() != d - P) offs.push_back(d - P);  // snap last patch flush to the edge
    return offs;
}

// Azimuth (row) + range (col) patch-offset grid for patch size P and overlap px (stride = P -
// overlap). max_rows (>=0) caps azimuth patch-rows for quick tests. Throws on a bad overlap or a
// sub-patch tile (errors over silent fallbacks).
void make_grid(size_t H, size_t W, int P, int overlap, int max_rows,
               std::vector<int>& row_offs, std::vector<int>& col_offs) {
    if (overlap < 0 || overlap >= P)
        throw std::runtime_error("stream: --overlap must be in [0, " + std::to_string(P) + ")");
    const int stride = P - overlap;
    row_offs = make_offsets(H, P, stride);
    col_offs = make_offsets(W, P, stride);
    if (row_offs.empty() || col_offs.empty())
        throw std::runtime_error("stream: tile smaller than a patch");
    if (max_rows >= 0 && max_rows < static_cast<int>(row_offs.size()))
        row_offs.resize(static_cast<size_t>(max_rows));
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
    h.stride = static_cast<uint16_t>(P - opt.overlap);
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

    std::vector<int> row_offs, col_offs;
    make_grid(H, W, P, opt.overlap, opt.max_rows, row_offs, col_offs);
    const int grid_a = static_cast<int>(row_offs.size());  // azimuth patch-rows
    const int grid_r = static_cast<int>(col_offs.size());  // range patch-cols

    PatchState s = pipe.make_patch_state(P, P);
    std::vector<DdcRecord> records;
    records.reserve(static_cast<size_t>(grid_a) * grid_r);

    // Compress one block/tile-row of patches, single-threaded, with per-stage timing. Blocks are
    // consumed strictly in order, so appending records row-major stays correct with or without
    // prefetch. (pa unused here — records are appended, not index-placed as in p0.)
    auto process_block = [&](int pa, const float* src, size_t base_row) {
        (void)pa;
        for (int pr = 0; pr < grid_r; ++pr) {
            const size_t c0 = static_cast<size_t>(col_offs[pr]);
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
                auto blk = reader->read_row_block(static_cast<size_t>(row_offs[pa]), P);
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
                rowblock = reader->read_row_block(static_cast<size_t>(row_offs[pa]), P);
                res.t_read_ms += ms(tread, clk::now());
                process_block(pa, rowblock.data(), 0);  // block is [P, W, 2]; patch rows are 0..P-1
            } else {
                process_block(pa, tile.data.data(), static_cast<size_t>(row_offs[pa]));
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

    // --fanout: K lanes, each its own DPU runner set + PatchState, DPU mutex dropped. The runner→core
    // mapping is set by the *order* runners are created (VART round-robin, no core-select), so we
    // deserialize the graph ONCE and create every lane's runners in a controlled global order:
    //   default (subgraph-major): all lanes' g_a, then all h_a, then all h_s ⇒ lane k pinned to core k;
    //   --lane-major (naive baseline): all of lane 0's runners, then lane 1's, … ⇒ every g_a collides
    //   on one core. `g_s` is never created (decode-only). This replaces the old per-lane graph copies,
    //   whose random pointer order made placement a run-to-run lottery (docs/onboard_pipeline.md §11-N1).
    // Non-fanout p0: a single shared pipeline whose DPU is serialized by dpu_mtx (composes with --s1).
    const int K = std::max(1, opt.threads);
#ifdef HAVE_DPU
    XModelLoader fanout_graph;  // declared first so it OUTLIVES pipes (their runners reference it)
#endif
    std::vector<std::unique_ptr<BenchPipeline>> pipes;
    if (opt.fanout) {
#ifdef HAVE_DPU
        const auto meta_json = opt.xmodel.parent_path() / "meta.json";
        fanout_graph.load(opt.xmodel.string(), meta_json.string(), /*create_runners=*/false);
        pipes.reserve(static_cast<size_t>(K));
        for (int k = 0; k < K; ++k)
            pipes.push_back(std::make_unique<BenchPipeline>(opt.params, BenchPipeline::LaneMode{}));
        std::vector<std::string> roles = {"g_a"};  // compress DPU roles, in pipeline order; NO g_s
        if (pipes[0]->uses_hyper()) { roles.emplace_back("h_a"); roles.emplace_back("h_s"); }
        auto make = [&](int k, const std::string& role) {
            pipes[static_cast<size_t>(k)]->adopt_runner(
                role, fanout_graph.create_duplicate_runner(
                          role, "L" + std::to_string(k) + "_" + role));
        };
        if (opt.lane_major)
            for (int k = 0; k < K; ++k) for (const auto& r : roles) make(k, r);
        else
            for (const auto& r : roles) for (int k = 0; k < K; ++k) make(k, r);
#else
        throw std::runtime_error("stream: --fanout requires a HAVE_DPU build");
#endif
    } else {
        pipes.push_back(std::make_unique<BenchPipeline>(opt.xmodel, opt.params));
        if (opt.s1) pipes[0]->init_s1();
    }
    for (auto& p : pipes) p->set_neon(opt.neon);
    const bool hyper = pipes[0]->uses_hyper();

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

    std::vector<int> row_offs, col_offs;
    make_grid(H, W, P, opt.overlap, opt.max_rows, row_offs, col_offs);
    const int grid_a = static_cast<int>(row_offs.size());
    const int grid_r = static_cast<int>(col_offs.size());
    const int n = grid_a * grid_r;

    std::vector<DdcRecord> records(static_cast<size_t>(n));
    std::vector<PatchState> states;
    states.reserve(static_cast<size_t>(K));
    for (int k = 0; k < K; ++k)
        states.push_back(pipes[opt.fanout ? static_cast<size_t>(k) : 0]->make_patch_state(P, P));
    std::vector<LanePerf> lane_perf(static_cast<size_t>(K));
    for (int k = 0; k < K; ++k) lane_perf[static_cast<size_t>(k)].lane = k;
    std::mutex dpu_mtx;  // used only by the shared-pipeline (non-fanout) path

    // --trace (fanout only): record each stage's [start,end] on a shared clock so a Gantt can show
    // when lanes wait/collide. Each lane writes its own vector (single writer); read_trace is written
    // only by the producer thread. Off by default -> zero overhead.
    const bool tracing = !opt.trace_out.empty();
    const auto trace_t0 = clk::now();
    std::vector<std::vector<TraceEvent>> lane_traces(static_cast<size_t>(K));
    std::vector<TraceEvent> read_trace;

    // Per-patch compress (s.noisy_hwc already filled) -> records[idx]; CPU stages overlap across
    // workers. Fan-out: worker wid owns pipes[wid] on its own DPU core (no lock) and its per-stage
    // times accumulate into lane_perf[wid] for the placement diagnosis. Non-fanout: all workers share
    // pipes[0] with the DPU serialized by dpu_mtx (the original p0, unchanged, still composes with s1).
    auto process_patch = [&](int wid, PatchState& s, int idx) {
        DdcRecord rec;
        if (opt.fanout) {
            BenchPipeline& pipe = *pipes[static_cast<size_t>(wid)];
            LanePerf& L = lane_perf[static_cast<size_t>(wid)];
            // Time a stage into `acc`, and (when tracing) log its [start,end] to this lane's timeline.
            auto ts = [&](const char* stage, double& acc, auto&& fn) {
                const auto a = clk::now();
                fn();
                const auto b = clk::now();
                acc += ms(a, b);
                if (tracing)
                    lane_traces[static_cast<size_t>(wid)].push_back(
                        TraceEvent{wid, idx, stage, ms(trace_t0, a), ms(trace_t0, b)});
            };
            ts("normalize", L.normalize_ms, [&] { pipe.stage_normalize(s); });
            ts("g_a", L.ga_ms, [&] { pipe.stage_ga(s); });  // own core, no mutex
            if (hyper) ts("h_a", L.ha_ms, [&] { pipe.stage_ha(s); });
            ts("eb", L.entropy_ms, [&] { pipe.stage_eb_compress(s); });
            if (hyper) {
                ts("eb", L.entropy_ms, [&] { pipe.stage_eb_decompress(s); });
                ts("h_s", L.hs_ms, [&] { pipe.stage_hs(s); });
                ts("gc", L.entropy_ms, [&] { pipe.stage_gc_compress(s); });
                rec.z = s.z_bits;
                rec.y = s.y_bits;
            } else {
                rec.y = s.y_bits;
            }
            L.patches++;
        } else {
            BenchPipeline& pipe = *pipes[0];
            pipe.stage_normalize(s);
            {
                std::lock_guard<std::mutex> lk(dpu_mtx);
                if (opt.s1) pipe.stage_ga_s1(s); else pipe.stage_ga(s);
                if (hyper) pipe.stage_ha(s);
            }
            pipe.stage_eb_compress(s);
            if (hyper) {
                pipe.stage_eb_decompress(s);
                { std::lock_guard<std::mutex> lk(dpu_mtx); pipe.stage_hs(s); }
                pipe.stage_gc_compress(s);
                rec.z = s.z_bits;
                rec.y = s.y_bits;
            } else {
                rec.y = s.y_bits;
            }
        }
        records[static_cast<size_t>(idx)] = std::move(rec);  // distinct slot: no lock
    };

    // Compress one row-block: K workers pull patches (pr) off an atomic counter; the DPU bursts are
    // serialized inside process_patch and records placed by index. Used by the non-prefetch windowed
    // path only (the prefetch path drains a persistent pool that spans blocks — see below).
    auto process_block = [&](int pa, const float* block) {
        std::atomic<int> pr_next{0};
        auto blockworker = [&](int wid) {
            PatchState& s = states[static_cast<size_t>(wid)];
            int pr;
            while ((pr = pr_next.fetch_add(1)) < grid_r) {
                fill_patch(s, block, 0, static_cast<size_t>(col_offs[pr]), P, W);
                process_patch(wid, s, pa * grid_r + pr);
            }
        };
        std::vector<std::thread> pool;
        for (int k = 0; k < K; ++k) pool.emplace_back(blockworker, k);
        for (auto& t : pool) t.join();
    };

    if (windowed && opt.prefetch) {
        // Persistent K-worker pool draining a patch-level queue: no per-row-block join barrier and no
        // per-block thread churn (workers cross block boundaries freely, so a fast lane never idles
        // waiting for the slowest lane at a block edge). A producer reads row-block N+1 (as a
        // shared_ptr) while the workers compress patches from blocks already queued; each patch item
        // holds a shared_ptr to its block, so the block buffer frees once its last patch is done. The
        // queue is bounded to ~2 row-blocks of items = the same double-buffer memory ceiling as before.
        // Output stays byte-identical: records are placed by absolute index, independent of schedule.
        struct Item {
            std::shared_ptr<std::vector<float>> block;
            int pa = 0, pr = 0;
        };
        std::deque<Item> q;
        std::mutex qm;
        std::condition_variable q_not_full, q_not_empty;
        bool q_closed = false;
        const size_t q_cap = static_cast<size_t>(grid_r) * 2;  // ~2 row-blocks in flight (double buffer)
        double read_ms = 0.0;
        std::thread producer([&] {
            for (int pa = 0; pa < grid_a; ++pa) {
                const auto tr = clk::now();
                auto blk = std::make_shared<std::vector<float>>(
                    reader->read_row_block(static_cast<size_t>(row_offs[pa]), P));
                const auto tr_end = clk::now();
                read_ms += ms(tr, tr_end);  // only the producer touches read_ms
                if (tracing)
                    read_trace.push_back(TraceEvent{-1, pa, "read", ms(trace_t0, tr), ms(trace_t0, tr_end)});
                for (int pr = 0; pr < grid_r; ++pr) {
                    std::unique_lock<std::mutex> lk(qm);
                    q_not_full.wait(lk, [&] { return q.size() < q_cap; });
                    q.push_back(Item{blk, pa, pr});
                    q_not_empty.notify_one();
                }
            }
            {
                std::lock_guard<std::mutex> lk(qm);
                q_closed = true;
            }
            q_not_empty.notify_all();
        });
        auto worker = [&](int wid) {
            PatchState& s = states[static_cast<size_t>(wid)];
            for (;;) {
                Item it;
                {
                    std::unique_lock<std::mutex> lk(qm);
                    q_not_empty.wait(lk, [&] { return !q.empty() || q_closed; });
                    if (q.empty()) return;  // closed and drained
                    it = std::move(q.front());
                    q.pop_front();
                    q_not_full.notify_one();
                }
                fill_patch(s, it.block->data(), 0, static_cast<size_t>(col_offs[it.pr]), P, W);
                process_patch(wid, s, it.pa * grid_r + it.pr);
            }
        };
        std::vector<std::thread> pool;
        for (int k = 0; k < K; ++k) pool.emplace_back(worker, k);
        for (auto& t : pool) t.join();
        producer.join();
        res.t_read_ms += read_ms;  // raw read cost, now overlapped with compute (hidden in t_total)
    } else if (windowed) {
        // No prefetch: read a row-block, then compress it (deliberately un-overlapped, to measure the
        // bare SD read). One-block-in-DDR quick test; keeps the per-block pool (process_block).
        std::vector<float> rowblock;
        for (int pa = 0; pa < grid_a; ++pa) {
            const auto tread = clk::now();
            rowblock = reader->read_row_block(static_cast<size_t>(row_offs[pa]), P);
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
                fill_patch(s, tile.data.data(), static_cast<size_t>(row_offs[pa]),
                           static_cast<size_t>(col_offs[pr]), P, W);
                process_patch(wid, s, idx);
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
    if (opt.fanout) res.lane_perf = std::move(lane_perf);  // per-lane placement diagnosis

    if (tracing) {  // write the per-lane stage timeline (fanout_gantt.py reads this CSV)
        std::ofstream tf(opt.trace_out.string());
        if (!tf) throw std::runtime_error("stream: cannot open --trace file " + opt.trace_out.string());
        tf << "lane,patch,stage,t0_ms,t1_ms\n";
        auto dump = [&](const std::vector<TraceEvent>& v) {
            for (const auto& e : v)
                tf << e.lane << ',' << e.patch << ',' << e.stage << ',' << e.t0_ms << ',' << e.t1_ms
                   << '\n';
        };
        dump(read_trace);
        for (const auto& lt : lane_traces) dump(lt);
    }
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
    // Stream each decoded patch straight to the [n, P, P] NPY (record order) rather than buffering the
    // whole ~n*P*P array — at full-scene+overlap that buffer is ~2.2 GB, close to the DDR ceiling.
    std::ofstream fout(opt.out_ddc.string(), std::ios::binary);
    if (!fout) throw std::runtime_error("stream_decode: cannot open " + opt.out_ddc.string());
    npy_write_header_float32(
        fout, {static_cast<size_t>(n), static_cast<size_t>(P), static_cast<size_t>(P)});
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
        res.t_write_ms += time_stage([&] {
            fout.write(reinterpret_cast<const char*>(s.recon_lina.data()),
                       static_cast<std::streamsize>(P) * P * sizeof(float));
        });
    }
    fout.close();
    if (!fout) throw std::runtime_error("stream_decode: short write to " + opt.out_ddc.string());

    res.grid_r = f.header.grid_r;
    res.grid_a = f.header.grid_a;
    res.n_patches = n;
    res.file_bytes = std::filesystem::file_size(opt.out_ddc);
    res.bpp = n > 0 ? res.payload_bytes * 8.0 / (static_cast<double>(n) * P * P) : 0.0;
    res.t_total_ms = ms(t0, clk::now());
    return res;
}

}  // namespace ddc
