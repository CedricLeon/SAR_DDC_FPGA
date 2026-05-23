/* main_benchmark.cpp - CLI for benchmark_hardware.
 *
 * Usage (on ZCU102):
 *   ./benchmark_hardware
 *       --xmodel  active_model/model.xmodel
 *       --params  active_model/entropy_params
 *       --data    data/test_sub500_seed42.npy
 *       [--config    s0]           default: s0
 *       [--scenario  compress]     compress | full
 *       [--warmup    5]
 *       [--iters     50]
 *       [--subset    20]           patches loaded and cycled
 *       [--dpu-cores    1]         runner replicas (S0 ignores >1)
 *       [--entropy-threads 1]      entropy workers (S0 ignores >1)
 *       [--output    result.json]
 *       [--power]                  enable power sampling (M2 — no-op in M1)
 *       [--verbose]
 *
 * Output: JSON file with per-stage stats + throughput + metadata.
 * Schema extends benchmark_fpga.py so results drop into benchmark_analysis.ipynb.
 */

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "bench_configs.hpp"
#include "bench_pipeline.hpp"
#include "stage_timer.hpp"

static void usage(const char* prog)
{
    std::cerr
        << "Usage: " << prog << "\n"
        << "  --xmodel  <path>           .xmodel file (required)\n"
        << "  --params  <path>           entropy params dir with eb_*.npy (required)\n"
        << "  --data    <path>           test .npy (N,H,W,4) (required)\n"
        << "  --config  <s0>             benchmark config [default: s0]\n"
        << "  --scenario <compress|full> pipeline scenario [default: compress]\n"
        << "  --warmup  <N>              warmup iterations [default: 5]\n"
        << "  --iters   <N>              timed iterations [default: 50]\n"
        << "  --subset  <N>              patches to load and cycle [default: 20]\n"
        << "  --dpu-cores <N>            DPU runner replicas [default: 1]\n"
        << "  --entropy-threads <N>      entropy workers [default: 1]\n"
        << "  --output  <path>           JSON output path [default: benchmark_result.json]\n"
        << "  --power                    enable power sampling (M2; no-op now)\n"
        << "  --verbose                  enable verbose logging\n";
}

int main(int argc, char** argv)
{
    namespace fs = std::filesystem;

    std::string  xmodel_str, params_str, data_str;
    std::string  config_name   = "s0";
    std::string  scenario_str  = "compress";
    std::string  output_str    = "benchmark_result.json";
    int          warmup        = 5;
    int          iters         = 50;
    int          subset        = 20;
    int          dpu_cores     = 1;
    int          entropy_thds  = 1;
    bool         power_flag    = false;
    bool         verbose       = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(EXIT_FAILURE);
            }
            return argv[++i];
        };

        if      (arg == "--xmodel")           xmodel_str   = next();
        else if (arg == "--params")           params_str   = next();
        else if (arg == "--data")             data_str     = next();
        else if (arg == "--config")           config_name  = next();
        else if (arg == "--scenario")         scenario_str = next();
        else if (arg == "--warmup")           warmup       = std::stoi(next());
        else if (arg == "--iters")            iters        = std::stoi(next());
        else if (arg == "--subset")           subset       = std::stoi(next());
        else if (arg == "--dpu-cores")        dpu_cores    = std::stoi(next());
        else if (arg == "--entropy-threads")  entropy_thds = std::stoi(next());
        else if (arg == "--output")           output_str   = next();
        else if (arg == "--power")            power_flag   = true;
        else if (arg == "--verbose")          verbose      = true;
        else if (arg == "--help" || arg == "-h") { usage(argv[0]); return 0; }
        else {
            std::cerr << "Unknown argument: " << arg << "\n";
            usage(argv[0]);
            return 1;
        }
    }

    // Validate required args
    if (xmodel_str.empty() || params_str.empty() || data_str.empty()) {
        std::cerr << "Error: --xmodel, --params, and --data are required\n";
        usage(argv[0]);
        return 1;
    }

    // Validate config
    if (config_name != "s0") {
        std::cerr << "Error: unknown --config '" << config_name
                  << "'. Available: s0\n";
        return 1;
    }

    // Parse scenario
    ddc::Scenario scenario;
    if      (scenario_str == "compress") scenario = ddc::Scenario::compress;
    else if (scenario_str == "full")     scenario = ddc::Scenario::full;
    else {
        std::cerr << "Error: unknown --scenario '" << scenario_str
                  << "'. Use: compress | full\n";
        return 1;
    }

    // Check paths
    for (const auto& [label, p] : std::initializer_list<std::pair<const char*, const char*>>{
            {"xmodel", xmodel_str.c_str()},
            {"params", params_str.c_str()},
            {"data",   data_str.c_str()}}) {
        if (!fs::exists(p)) {
            std::cerr << "Error: " << label << " not found: " << p << "\n";
            return 1;
        }
    }

    if (power_flag)
        std::cerr << "[info] --power specified; power sampling not yet implemented (M2).\n";

    if (verbose)
        std::cerr << "[info] verbose mode enabled.\n";

    // ISO-8601 timestamp
    auto now    = std::chrono::system_clock::now();
    std::time_t t_now = std::chrono::system_clock::to_time_t(now);
    std::ostringstream ts;
    ts << std::put_time(std::gmtime(&t_now), "%Y-%m-%dT%H:%M:%SZ");

    try {
        // Load models — use named temporaries to avoid the most-vexing-parse
        fs::path xmodel_path{xmodel_str};
        fs::path params_path{params_str};
        ddc::BenchPipeline pipeline{xmodel_path, params_path};

        ddc::RunConfig cfg;
        cfg.scenario        = scenario;
        cfg.warmup          = warmup;
        cfg.iters           = iters;
        cfg.data_path       = data_str;
        cfg.subset          = subset;
        cfg.dpu_cores       = dpu_cores;
        cfg.entropy_threads = entropy_thds;

        // Dispatch
        ddc::BenchResult result;
        if (config_name == "s0")
            result = ddc::run_s0(pipeline, cfg);

        // Serialize to JSON
        nlohmann::json out;
        out["config"]           = config_name;
        out["scenario"]         = scenario_str;
        out["arch"]             = result.is_shyp ? "SHyp" : "FP";
        out["xmodel"]           = xmodel_str;
        out["params"]           = params_str;
        out["dpu_cores"]        = dpu_cores;
        out["entropy_threads"]  = entropy_thds;
        out["warmup"]           = warmup;
        out["iters"]            = result.num_iters;
        out["subset_patches"]   = subset;
        out["wall_time_s"]      = result.wall_time_s;
        out["throughput_fps"]   = result.throughput_fps;
        out["total_latency_mean_ms"] = result.total_latency_mean_ms;
        out["evaluated_at"]     = ts.str();

        nlohmann::json stages_json;
        for (const auto& [label, st] : result.stage_stats) {
            nlohmann::json sj;
            sj["mean_ms"]   = st.mean_s   * 1e3;
            sj["std_ms"]    = st.std_s    * 1e3;
            sj["median_ms"] = st.median_s * 1e3;
            sj["p95_ms"]    = st.p95_s    * 1e3;
            sj["min_ms"]    = st.min_s    * 1e3;
            sj["max_ms"]    = st.max_s    * 1e3;
            sj["n"]         = static_cast<int>(st.n);
            stages_json[label] = sj;
        }
        out["stages"] = stages_json;

        // Write
        std::ofstream f(output_str);
        if (!f) throw std::runtime_error("cannot open output: " + output_str);
        f << out.dump(4) << "\n";

        // Print summary to stderr
        std::cerr << "\n=== benchmark_hardware " << config_name
                  << " (" << scenario_str << ") ===\n"
                  << "  arch:        " << out["arch"] << "\n"
                  << "  iters:       " << result.num_iters << "\n"
                  << "  wall time:   " << result.wall_time_s << " s\n"
                  << "  throughput:  " << result.throughput_fps << " fps\n"
                  << "  total lat:   " << result.total_latency_mean_ms << " ms\n"
                  << "\n  per-stage (mean | p95 ms):\n";
        for (const auto& [label, st] : result.stage_stats) {
            std::cerr << "    " << label
                      << ": " << st.mean_s * 1e3 << " | " << st.p95_s * 1e3 << " ms\n";
        }
        std::cerr << "\n  -> " << output_str << "\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
