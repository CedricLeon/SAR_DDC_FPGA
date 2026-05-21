/* main.cpp
 * CLI entry point for the C++ hybrid inference binary.
 *
 * Usage (on ZCU102):
 *   ./inference_hybrid \
 *       --xmodel  /path/to/model.xmodel \
 *       --params  /path/to/entropy_params/ \
 *       --data    /path/to/test_set.npy \
 *       --output  /path/to/results/ \
 *       --subset  100 \
 *       [--verbose]
 */

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "inference_runner.hpp"
#include "logger.hpp"

static void usage(const char *prog)
{
    std::cerr
        << "Usage: " << prog << "\n"
        << "  --xmodel  <path>   .xmodel file (required)\n"
        << "  --params  <path>   entropy params directory with eb_*.npy files (required)\n"
        << "  --data    <path>   test set .npy (N,H,W,4) (required)\n"
        << "  --output  <path>   output directory [default: xmodel_dir/results]\n"
        << "  --subset  <int>    number of test-set patches to evaluate [default: 100]\n"
        << "  --debug-patch <N>  enable verbose stats only for patch index N (0-based)\n"
        << "  --verbose          enable verbose per-sample logging\n"
        << "  --log     <file>   write log to file in addition to stderr\n";
}

int main(int argc, char **argv)
{
    ddc::InferenceConfig cfg;
    std::string log_file;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        auto next = [&]() -> std::string
        {
            if (i + 1 >= argc)
            {
                std::cerr << "Error: " << arg << " requires an argument\n";
                std::exit(EXIT_FAILURE);
            }
            return argv[++i];
        };

        if (arg == "--xmodel")
            cfg.xmodel_path = next();
        else if (arg == "--params")
            cfg.params_dir = next();
        else if (arg == "--data")
            cfg.data_path = next();
        else if (arg == "--output")
            cfg.output_dir = next();
        else if (arg == "--subset")
            cfg.subset = std::stoi(next());
        else if (arg == "--debug-patch")
            cfg.debug_patch = std::stoi(next());
        else if (arg == "--log")
            log_file = next();
        else if (arg == "--verbose")
            cfg.verbose = true;
        else if (arg == "--help" || arg == "-h")
        {
            usage(argv[0]);
            return 0;
        }
        else
        {
            std::cerr << "Unknown argument: " << arg << "\n";
            usage(argv[0]);
            return 1;
        }
    }

    // Validate required arguments
    if (cfg.xmodel_path.empty())
    {
        std::cerr << "Error: --xmodel required\n";
        return 1;
    }
    if (cfg.params_dir.empty())
    {
        std::cerr << "Error: --params required\n";
        return 1;
    }
    if (cfg.data_path.empty())
    {
        std::cerr << "Error: --data required\n";
        return 1;
    }

    if (!std::filesystem::exists(cfg.xmodel_path))
    {
        std::cerr << "Error: xmodel not found: " << cfg.xmodel_path << "\n";
        return 1;
    }
    if (!std::filesystem::exists(cfg.params_dir))
    {
        std::cerr << "Error: params dir not found: " << cfg.params_dir << "\n";
        return 1;
    }
    if (!std::filesystem::exists(cfg.data_path))
    {
        std::cerr << "Error: data file not found: " << cfg.data_path << "\n";
        return 1;
    }

    // Default output dir: <xmodel_dir>/results
    if (cfg.output_dir.empty())
        cfg.output_dir = cfg.xmodel_path.parent_path() / "results";

    // Set up logger
    if (cfg.verbose)
        ddc::Logger::instance().set_verbose(true);
    if (!log_file.empty())
        ddc::Logger::instance().open_log_file(log_file);

    try
    {
        ddc::InferenceRunner runner(cfg);
        runner.run();
    }
    catch (const std::exception &e)
    {
        LOG_ERROR(std::string("Fatal: ") + e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
