/* dpu_runners.hpp
 * C++ port of scripts/fpga/inference_utils.py DPUSubgraphRunner + identify_subgraphs.
 *
 * All DPU-dependent code is guarded by #ifdef HAVE_DPU so the project
 * compiles on a development host (HAVE_DPU=OFF) without VART/XIR headers.
 *
 * DPUSubgraphRunner wraps a single subgraph (g_a / h_a / h_s / g_s):
 *   - get_attr("fix_point") -> input_scale = 2^fix_point, output_scale = 2^-fix_point
 *   - run(): float* in -> quantize to int8 -> execute_async -> wait -> dequantize -> float* out
 *
 * XModelLoader:
 *   - Deserializes the .xmodel, reads meta.json kernel list,
 *     matches each kernel name to a role (g_a/h_a/h_s/g_s).
 *   - Returns a map<string, DPUSubgraphRunner>.
 */
#pragma once

#ifdef HAVE_DPU

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <vart/runner.hpp>
#include <xir/graph/graph.hpp>
#include <xir/graph/subgraph.hpp>

namespace ddc {

// ---------------------------------------------------------------------------
// DPUSubgraphRunner — float-in, float-out wrapper around a VART runner
// ---------------------------------------------------------------------------
class DPUSubgraphRunner {
public:
    // Construct from a DPU subgraph. Reads fix_point attrs and logs shapes.
    // name: human-readable label ("g_a" etc.) for log output.
    explicit DPUSubgraphRunner(xir::Subgraph* subgraph, std::string name);

    // No copy; move OK (unique_ptr inside).
    DPUSubgraphRunner(const DPUSubgraphRunner&)            = delete;
    DPUSubgraphRunner& operator=(const DPUSubgraphRunner&) = delete;
    DPUSubgraphRunner(DPUSubgraphRunner&&)                 = default;
    DPUSubgraphRunner& operator=(DPUSubgraphRunner&&)      = default;

    // Synchronous float-in, float-out inference.
    // input:  float array, layout NHWC, size = product(input_shape_).
    // output: caller-allocated float array, size = product(output_shape_).
    void run(const float* input, float* output) const;

    // Shape accessors (NHWC)
    const std::vector<int>& input_shape()  const { return input_shape_; }
    const std::vector<int>& output_shape() const { return output_shape_; }

    // Element counts
    int input_numel()  const;
    int output_numel() const;

    const std::string& name() const { return name_; }

private:
    std::string name_;
    std::unique_ptr<vart::Runner> runner_;

    std::vector<int> input_shape_;   // [N, H, W, C]
    std::vector<int> output_shape_;  // [N, H, W, C]
    float input_scale_  = 1.0f;      // 2^fix_point
    float output_scale_ = 1.0f;      // 2^(-fix_point)
};

// ---------------------------------------------------------------------------
// XModelLoader — load .xmodel + meta.json, return named runners
// ---------------------------------------------------------------------------
class XModelLoader {
public:
    // Load xmodel from xmodel_path.
    // meta_json_path: path to meta.json produced by Vitis-AI compiler.
    // create_runners=false deserializes the graph and identifies subgraphs but creates NO runners —
    // for callers that then create every runner themselves via create_duplicate_runner() in a
    // controlled global order (the fan-out placement fix; runner-creation order sets the VART core).
    // Throws std::runtime_error on failure.
    void load(const std::string& xmodel_path,
              const std::string& meta_json_path,
              bool create_runners = true);

    // True once load() has succeeded.
    bool is_loaded() const { return loaded_; }

    // Get a runner by role name ("g_a", "h_a", "h_s", "g_s").
    // Throws std::out_of_range if role not found.
    DPUSubgraphRunner& runner(const std::string& role);
    const DPUSubgraphRunner& runner(const std::string& role) const;

    // True if a role exists (e.g. h_a/h_s only present for ScaleHyperprior).
    bool has_role(const std::string& role) const;

    // Create an additional runner for 'role' with label 'alias'.
    // The new runner obtains the next VART round-robin core assignment.
    // Used by BenchPipeline::init_s1() to create g_a_1 / g_s_1 duplicates.
    // Throws if 'role' is not found or load() has not been called.
    DPUSubgraphRunner create_duplicate_runner(const std::string& role,
                                              const std::string& alias) const;

private:
    bool loaded_ = false;
    std::unique_ptr<xir::Graph>              graph_;
    std::map<std::string, DPUSubgraphRunner> runners_;
    std::map<std::string, xir::Subgraph*>   subgraphs_;  // for create_duplicate_runner
};

} // namespace ddc

#endif // HAVE_DPU
