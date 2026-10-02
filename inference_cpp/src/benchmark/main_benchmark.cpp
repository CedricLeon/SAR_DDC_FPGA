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
 *       [--power]                  enable power sampling (INA226 + PMBus)
 *       [--idle-baseline-s 10]     idle window duration in seconds
 *       [--verbose]
 *
 * Output: JSON file with per-stage stats + throughput + metadata.
 * Schema extends benchmark_fpga.py so results drop into benchmark_analysis.ipynb.
 * With --power, a "power" object is added containing idle and active windows
 * with per-rail stats and group aggregates (PL, PS, DPU_fabric, etc.).
 */

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "bench_configs.hpp"
#include "bench_pipeline.hpp"
#include "power_sampler.hpp"
#include "stage_timer.hpp"

static void usage(const char* prog)
{
    std::cerr
        << "Usage: " << prog << "\n"
        << "  --xmodel  <path>           .xmodel file (required)\n"
        << "  --params  <path>           entropy params dir with eb_*.npy (required)\n"
        << "  --data    <path>           test .npy (N,H,W,4) (required)\n"
        << "  --config  <name>           benchmark config [default: s0]\n"
        << "                             s0         sequential baseline (1 DPU, 1 thread)\n"
        << "                             s1         channel-parallel g_a/g_s (2 runners each)\n"
        << "                             nn_only    DPU data-parallel ceiling\n"
        << "                             entropy_only  CPU entropy ceiling\n"
        << "  --scenario <compress|full> pipeline scenario [default: compress]\n"
        << "  --warmup  <N>              warmup iterations [default: 5]\n"
        << "  --iters   <N>              timed iterations [default: 50]\n"
        << "  --subset  <N>              patches to load and cycle [default: 20]\n"
        << "  --dpu-cores <N>            nn_only: N concurrent pipelines [default: 1]\n"
        << "                             s0/s1: ignored\n"
        << "  --entropy-threads <N>      entropy_only: N concurrent workers [default: 1]\n"
        << "                             s0/s1: ignored\n"
        << "  --output  <path>           JSON output path [default: benchmark_result.json]\n"
        << "  --power                    enable INA226 + PMBus power sampling\n"
        << "  --idle-baseline-s <N>      idle baseline window in seconds [default: 10]\n"
        << "  --no-entropy-opt           disable the rANS flattened-CDF+reciprocal optimization\n"
        << "                             (4ddbcc8); s0/s1 only. Default: optimization ON (matches\n"
        << "                             stream_pipeline's default; pass this for a pre-optimization\n"
        << "                             sequential baseline).\n"
        << "  --verbose                  enable verbose logging\n";
}

int main(int argc, char** argv)
{
    namespace fs = std::filesystem;

    std::string  xmodel_str, params_str, data_str;
    std::string  config_name      = "s0";
    std::string  scenario_str     = "compress";
    std::string  output_str       = "benchmark_result.json";
    int          warmup           = 5;
    int          iters            = 50;
    int          subset           = 20;
    int          dpu_cores        = 1;
    int          entropy_thds     = 1;
    int          idle_baseline_s  = 10;
    bool         power_flag       = false;
    bool         verbose          = false;
    bool         entropy_off      = false;

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
        else if (arg == "--output")           output_str      = next();
        else if (arg == "--power")            power_flag      = true;
        else if (arg == "--idle-baseline-s")  idle_baseline_s = std::stoi(next());
        else if (arg == "--no-entropy-opt")   entropy_off     = true;
        else if (arg == "--verbose")          verbose         = true;
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
    const std::initializer_list<const char*> valid_configs =
        {"s0", "s1", "nn_only", "entropy_only"};
    bool config_valid = false;
    for (const char* c : valid_configs)
        if (config_name == c) { config_valid = true; break; }
    if (!config_valid) {
        std::cerr << "Error: unknown --config '" << config_name
                  << "'. Available: s0 | s1 | nn_only | entropy_only\n";
        return 1;
    }

    // Warn on inapplicable flags (soft warning — scripted sweeps may pass a fixed flag set)
    auto warn_ignored = [](const char* flag, const char* cfg, const char* hint = nullptr) {
        std::cerr << "[warn] --" << flag << " has no effect for --config " << cfg;
        if (hint) std::cerr << "; " << hint;
        std::cerr << "\n";
    };
    if (config_name == "s0" || config_name == "s1") {
        if (dpu_cores    != 1) warn_ignored("dpu-cores",       config_name.c_str(), "fixed at 2 runners/pair for s1, 1 for s0");
        if (entropy_thds != 1) warn_ignored("entropy-threads", config_name.c_str());
    } else if (config_name == "nn_only") {
        if (entropy_thds != 1) warn_ignored("entropy-threads", "nn_only", "use --dpu-cores for nn_only");
    } else if (config_name == "entropy_only") {
        if (dpu_cores    != 1) warn_ignored("dpu-cores",       "entropy_only", "use --entropy-threads for entropy_only");
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

    if (verbose)
        std::cerr << "[info] verbose mode enabled.\n";

    // ISO-8601 timestamp
    auto now    = std::chrono::system_clock::now();
    std::time_t t_now = std::chrono::system_clock::to_time_t(now);
    std::ostringstream ts;
    ts << std::put_time(std::gmtime(&t_now), "%Y-%m-%dT%H:%M:%SZ");

    try {
        fs::path xmodel_path{xmodel_str};
        fs::path params_path{params_str};

        ddc::RunConfig cfg;
        cfg.scenario        = scenario;
        cfg.warmup          = warmup;
        cfg.iters           = iters;
        cfg.data_path       = data_str;
        cfg.subset          = subset;
        cfg.dpu_cores       = dpu_cores;
        cfg.entropy_threads = entropy_thds;
        cfg.xmodel_path     = xmodel_path;   // used by nn_only / entropy_only
        cfg.params_path     = params_path;

        // Construct pipeline before the power window for s0/s1 (avoids including
        // DPU runner initialisation in the active power reading).
        // nn_only/entropy_only construct their own pipelines inside the run function.
        std::unique_ptr<ddc::BenchPipeline> pipeline_ptr;
        if (config_name == "s0" || config_name == "s1") {
            pipeline_ptr = std::make_unique<ddc::BenchPipeline>(xmodel_path, params_path);
            if (config_name == "s1")
                pipeline_ptr->init_s1();
            pipeline_ptr->set_entropy_opt(!entropy_off);
        }

        // Power sampler — idle baseline before, active window around benchmark
        ddc::PowerSampler sampler;
        ddc::PowerResult  power_idle, power_active;
        bool power_ok = false;
        if (power_flag) {
            if (!sampler.init()) {
                std::cerr << "[warn] --power: no INA226 sensors found; power data will be empty.\n";
            } else {
                std::cerr << "[info] Running idle baseline (" << idle_baseline_s << " s)...\n";
                power_idle = sampler.idle_baseline(static_cast<double>(idle_baseline_s));
                auto it = power_idle.rails.find("VCCINT");
                if (it != power_idle.rails.end())
                    std::cerr << "[info] Idle VCCINT: " << it->second.avg_power_w << " W ("
                              << it->second.n_samples << " samples)\n";
                std::cerr << "[info] Starting active power window...\n";
                sampler.start();
                power_ok = true;
            }
        }

        // Dispatch
        ddc::BenchResult result;
        if (config_name == "s0")
            result = ddc::run_s0(*pipeline_ptr, cfg);
        else if (config_name == "s1")
            result = ddc::run_s1(*pipeline_ptr, cfg);
        else if (config_name == "nn_only")
            result = ddc::run_nn_only(cfg);
        else   // entropy_only
            result = ddc::run_entropy_only(cfg);

        if (power_ok) {
            sampler.stop();
            power_active = sampler.results();
            auto it = power_active.rails.find("VCCINT");
            if (it != power_active.rails.end())
                std::cerr << "[info] Active VCCINT: " << it->second.avg_power_w << " W ("
                          << it->second.n_samples << " samples)\n";
        }

        // Serialize to JSON
        nlohmann::json out;
        out["config"]           = config_name;
        out["scenario"]         = scenario_str;
        // Derive arch from active_model/manifest.json ("model_name" field prefix).
        // The xmodel filename cannot distinguish ResSHyp from SHyp (both compile
        // to ResidualScaleHyperpriorDPUWrapper_pt.xmodel), but manifest.json
        // always carries the exact model name (e.g. "ResSHyp-relu_s0_L1000_pt").
        {
            fs::path manifest = xmodel_path.parent_path() / "manifest.json";
            if (!fs::exists(manifest))
                throw std::runtime_error("manifest.json not found at " + manifest.string()
                    + " — deploy the model with deploy.py before benchmarking");
            std::ifstream mf(manifest);
            auto mj = nlohmann::json::parse(mf);
            std::string model_name = mj.at("model_name").get<std::string>();
            auto dash = model_name.find('-');
            if (dash == std::string::npos)
                throw std::runtime_error("manifest.json model_name has unexpected format: '"
                    + model_name + "' (expected e.g. 'ResSHyp-relu_s0_L1000_pt')");
            out["arch"] = model_name.substr(0, dash);
        }
        out["xmodel"]           = xmodel_str;
        out["params"]           = params_str;
        out["dpu_cores"]        = dpu_cores;
        out["entropy_threads"]  = entropy_thds;
        out["entropy_opt"]      = !entropy_off;
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

        // Byte counts per timed iteration (run_s0 / run_s1 only — for correctness verification)
        if (!result.bytes_per_iter.empty())
            out["bytes_per_iter"] = result.bytes_per_iter;

        // Power results (only present when --power was specified and sensors found)
        if (power_ok) {
            auto power_to_json = [](const ddc::PowerResult& pr) {
                nlohmann::json pj;
                pj["valid"]      = pr.valid;
                pj["duration_s"] = pr.duration_s;
                nlohmann::json rails_j;
                for (const auto& [rail, rs] : pr.rails) {
                    rails_j[rail] = {
                        {"avg_power_w", rs.avg_power_w},
                        {"energy_j",    rs.energy_j},
                        {"n_samples",   rs.n_samples},
                        {"duration_s",  rs.duration_s},
                    };
                }
                pj["rails"]  = rails_j;
                nlohmann::json groups_j;
                for (const auto& [grp, w] : pr.groups)
                    groups_j[grp] = w;
                pj["groups"] = groups_j;
                return pj;
            };
            out["power"]["idle_baseline_s"] = idle_baseline_s;
            out["power"]["idle"]            = power_to_json(power_idle);
            out["power"]["active"]          = power_to_json(power_active);
        }

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
            std::cerr << "    " << std::left << std::setw(16) << label
                      << ": " << std::fixed << std::setprecision(3)
                      << st.mean_s * 1e3 << " | " << st.p95_s * 1e3 << " ms\n";
        }

        // Power summary
        if (power_ok) {
            // Helper: look up a group's power from a PowerResult (-1 = not found)
            auto grp_w = [](const ddc::PowerResult& pr, const std::string& g) -> double {
                auto it = pr.groups.find(g);
                return it != pr.groups.end() ? it->second : -1.0;
            };
            auto rail_w = [](const ddc::PowerResult& pr, const std::string& r) -> double {
                auto it = pr.rails.find(r);
                return it != pr.rails.end() ? it->second.avg_power_w : -1.0;
            };
            // Print one row: label  idle  active  +delta W
            auto pw_row = [&](const char* label, double idle, double active) {
                if (idle < 0.0 || active < 0.0) return;
                std::cerr << "    " << std::left << std::setw(14) << label
                          << std::right << std::fixed << std::setprecision(3)
                          << std::setw(7) << idle   << "  "
                          << std::setw(7) << active << "  "
                          << std::showpos << std::setw(7) << (active - idle)
                          << std::noshowpos << " W\n";
            };

            int n_ina = 0;
            {
                auto it = power_idle.rails.find("VCCINT");
                if (it != power_idle.rails.end()) n_ina = it->second.n_samples;
            }
            std::cerr << "\n  --- power (idle | active | delta) ---\n"
                      << "    " << std::left << std::setw(14) << ""
                      << std::right << std::setw(7) << "idle"  << "  "
                      << std::setw(7) << "active" << "  "
                      << std::setw(8) << "delta\n";
            pw_row("VCCINT",      rail_w(power_idle, "VCCINT"),     rail_w(power_active, "VCCINT"));
            pw_row("DPU_fabric",  grp_w(power_idle,  "DPU_fabric"), grp_w(power_active,  "DPU_fabric"));
            pw_row("PL",          grp_w(power_idle,  "PL"),         grp_w(power_active,  "PL"));
            pw_row("PS",          grp_w(power_idle,  "PS"),         grp_w(power_active,  "PS"));
            pw_row("MPSoC",       grp_w(power_idle,  "MPSoC"),      grp_w(power_active,  "MPSoC"));
            pw_row("peripherals", grp_w(power_idle,  "peripherals"),grp_w(power_active,  "peripherals"));
            std::cerr << "    (idle baseline: " << idle_baseline_s << " s"
                      << ", " << n_ina << " INA226 samples)\n";

            // Energy per patch
            double lat_s = result.total_latency_mean_ms / 1e3;
            double mpsoc_active = grp_w(power_active, "MPSoC");
            if (mpsoc_active > 0.0 && lat_s > 0.0)
                std::cerr << "\n  energy/patch (MPSoC × lat): "
                          << std::fixed << std::setprecision(2)
                          << mpsoc_active * lat_s * 1e3 << " mJ"
                          << "  (" << mpsoc_active << " W × "
                          << result.total_latency_mean_ms << " ms)\n";
        }

        std::cerr << "\n  -> " << output_str << "\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
