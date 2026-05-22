/* inference_runner.hpp
 * Top-level orchestrator that mirrors run_hybrid_inference() from
 * scripts/fpga/inference_hybrid.py.
 *
 * Two internal phases called sequentially by run():
 *   1. Test-subset phase — processes N 256×256 patches from a .npy test set,
 *      writes metrics.json and vis arrays.
 *   2. Tile-eval phase — scans data_path.parent() for large-tile subdirs
 *      containing sym_Noisy.npy, runs overlap-blended patch inference,
 *      writes per-tile metrics JSON and recon linA npy.
 *
 * DPU subgraph selection is automatic: if gc_scale_table.npy is present in
 * params_dir the ScaleHyperprior path (g_a→h_a→EB→h_s→GC→g_s) is used;
 * otherwise the FactorizedPrior path (g_a→EB→g_s).
 */
#pragma once

#include <filesystem>
#include <string>

namespace ddc
{

    // Forward declaration — full definition is in inference_runner.cpp
    class InferencePipeline;

    struct InferenceConfig
    {
        std::filesystem::path xmodel_path; // path to .xmodel
        std::filesystem::path params_dir;  // directory with eb_*.npy / gc_*.npy
        std::filesystem::path data_path;   // test set .npy  (N, H, W, 4)
        std::filesystem::path output_dir;  // where results are written
        int subset = 100;                  // how many test-set patches to evaluate
        bool verbose = false;
        bool skip_test_set = false;        // skip test-subset phase, run tile eval only
        // If >= 0: enable verbose logging only for this patch index (0-based).
        // Useful for diagnosing Python/C++ divergence on a specific patch.
        int debug_patch = -1;
        // large-tile scan: data_path.parent_path() is scanned automatically
    };

    class InferenceRunner
    {
    public:
        explicit InferenceRunner(const InferenceConfig &cfg);

        // Run the full pipeline (test subset + tile scan).
        void run();

    private:
        void _run_test_subset_impl(InferencePipeline &pipeline);
        void _run_tile_eval_impl(InferencePipeline &pipeline);

        InferenceConfig cfg_;
    };

} // namespace ddc
