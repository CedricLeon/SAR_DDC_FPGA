/* dpu_runners.cpp
 * C++ port of scripts/fpga/inference_utils.py DPUSubgraphRunner + identify_subgraphs.
 *
 * Only compiled when HAVE_DPU is defined (ZCU102 target).
 * See dpu_runners.hpp for the API contract.
 */

#ifdef HAVE_DPU

#include "dpu_runners.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// nlohmann/json — available system-wide on ZCU102 at /usr/include/nlohmann/json.hpp
#include <nlohmann/json.hpp>

// CpuFlatTensorBuffer is declared in the experimental header (still in namespace vart)
#include <vart/experimental/runner_helper.hpp>

#include "logger.hpp"

namespace ddc
{

    // ---------------------------------------------------------------------------
    // Helpers: quantize float→int8 and dequantize int8→float
    // Mirrors inference_utils.py: float_to_DPU_int / DPU_int_to_float
    // ---------------------------------------------------------------------------
    namespace
    {

        void quantize_float_to_int8(const float *src, int8_t *dst, int n, float scale)
        {
            for (int i = 0; i < n; ++i)
            {
                float v = src[i] * scale;
                // Saturating truncation toward zero (matches numpy .astype(np.int8))
                if (v > 127.0f)
                    v = 127.0f;
                if (v < -128.0f)
                    v = -128.0f;
                dst[i] = static_cast<int8_t>(v);
            }
        }

        void dequantize_int8_to_float(const int8_t *src, float *dst, int n, float scale)
        {
            for (int i = 0; i < n; ++i)
                dst[i] = static_cast<float>(src[i]) * scale;
        }

        int product(const std::vector<int> &shape)
        {
            return std::accumulate(shape.cbegin(), shape.cend(), 1, std::multiplies<int>());
        }

        [[maybe_unused]] std::vector<int> tensor_dims(vart::TensorBuffer *tb)
        {
            const auto &t = tb->get_tensor();
            std::vector<int> dims(t->get_shape().begin(), t->get_shape().end());
            return dims;
        }

    } // anonymous namespace

    // ---------------------------------------------------------------------------
    // DPUSubgraphRunner constructor
    // ---------------------------------------------------------------------------
    DPUSubgraphRunner::DPUSubgraphRunner(xir::Subgraph *subgraph, std::string name)
        : name_(std::move(name))
    {
        runner_ = vart::Runner::create_runner(subgraph, "run");
        if (!runner_)
            throw std::runtime_error("DPUSubgraphRunner[" + name_ + "]: create_runner returned null");

        auto in_tensors = runner_->get_input_tensors();
        auto out_tensors = runner_->get_output_tensors();

        if (in_tensors.empty() || out_tensors.empty())
            throw std::runtime_error("DPUSubgraphRunner[" + name_ + "]: no input/output tensors");

        // Shapes — VART returns dims in NHWC order
        for (int d : in_tensors[0]->get_shape())
            input_shape_.push_back(d);
        for (int d : out_tensors[0]->get_shape())
            output_shape_.push_back(d);

        // fix_point attributes -> quantisation scales
        int in_fixpos = in_tensors[0]->get_attr<int>("fix_point");
        int out_fixpos = out_tensors[0]->get_attr<int>("fix_point");
        input_scale_ = std::pow(2.0f, static_cast<float>(in_fixpos));
        output_scale_ = std::pow(2.0f, -static_cast<float>(out_fixpos));

        LOG_INFO("[DPUSubgraphRunner] " + name_ + " in=" + std::to_string(input_numel()) + " out=" + std::to_string(output_numel()) + " in_scale=" + std::to_string(input_scale_) + " out_scale=" + std::to_string(output_scale_));
    }

    int DPUSubgraphRunner::input_numel() const { return product(input_shape_); }
    int DPUSubgraphRunner::output_numel() const { return product(output_shape_); }

    // ---------------------------------------------------------------------------
    // DPUSubgraphRunner::run
    // ---------------------------------------------------------------------------
    void DPUSubgraphRunner::run(const float *input, float *output) const
    {
        const int in_n = input_numel();
        const int out_n = output_numel();

        // Quantize float -> int8
        std::vector<int8_t> in_buf(in_n);
        quantize_float_to_int8(input, in_buf.data(), in_n, input_scale_);

        // Allocate output int8 buffer
        std::vector<int8_t> out_buf(out_n);

        // Wrap raw buffers in CpuFlatTensorBuffer
        // VART CpuFlatTensorBuffer takes (data_ptr, tensor*)
        auto in_tensors = runner_->get_input_tensors();
        auto out_tensors = runner_->get_output_tensors();

        std::vector<vart::TensorBuffer *> in_tb_ptrs;
        std::vector<vart::TensorBuffer *> out_tb_ptrs;

        // CpuFlatTensorBuffer is in <vart/experimental/runner_helper.hpp>
        // Constructor: (void* data, const xir::Tensor*)
        vart::CpuFlatTensorBuffer in_tb(static_cast<void *>(in_buf.data()), in_tensors[0]);
        vart::CpuFlatTensorBuffer out_tb(static_cast<void *>(out_buf.data()), out_tensors[0]);

        in_tb_ptrs.push_back(&in_tb);
        out_tb_ptrs.push_back(&out_tb);

        auto [job_id, status] = runner_->execute_async(in_tb_ptrs, out_tb_ptrs);
        runner_->wait(static_cast<int>(job_id), -1);

        if (status != 0)
        {
            throw std::runtime_error("DPUSubgraphRunner[" + name_ + "]: execute_async returned status " + std::to_string(status));
        }

        // Dequantize int8 -> float
        dequantize_int8_to_float(out_buf.data(), output, out_n, output_scale_);
    }

    // ---------------------------------------------------------------------------
    // XModelLoader::load
    // ---------------------------------------------------------------------------
    void XModelLoader::load(const std::string &xmodel_path,
                            const std::string &meta_json_path)
    {
        // Deserialize the xmodel graph
        graph_ = xir::Graph::deserialize(xmodel_path);
        if (!graph_)
            throw std::runtime_error("XModelLoader: failed to deserialize " + xmodel_path);

        // Read meta.json to get kernel names
        std::ifstream meta_file(meta_json_path);
        if (!meta_file.is_open())
            throw std::runtime_error("XModelLoader: cannot open " + meta_json_path);

        nlohmann::json meta;
        meta_file >> meta;

        const auto &kernels = meta.at("kernel");

        // Map kernel name -> role (based on substring match, identical to Python identify_subgraphs)
        std::map<std::string, std::string> name_to_role;
        for (const auto &k_name : kernels)
        {
            const std::string s = k_name.get<std::string>();
            for (const char *role : {"g_a", "g_s", "h_a", "h_s"})
            {
                if (s.find(role) != std::string::npos)
                {
                    name_to_role[s] = role;
                    break;
                }
            }
        }

        LOG_INFO("XModelLoader: found " + std::to_string(name_to_role.size()) + " recognised subgraphs in meta.json");

        // Walk direct children of root — DPU kernels are always at depth 1.
        // C++ XIR exposes get_children() (std::set); toposort_child_subgraph() is
        // a Python-binding-only helper that doesn't exist in the C++ headers.
        xir::Subgraph *root = graph_->get_root_subgraph();
        for (xir::Subgraph *sg : root->get_children())
        {
            auto it = name_to_role.find(sg->get_name());
            if (it == name_to_role.end())
                continue;

            const std::string &role = it->second;
            subgraphs_[role] = sg;
            runners_.emplace(role, DPUSubgraphRunner(sg, role));
            LOG_INFO("XModelLoader: created runner for role=" + role + " (kernel=" + sg->get_name() + ")");
        }

        if (runners_.empty())
            throw std::runtime_error("XModelLoader: no subgraphs matched; check meta.json and xmodel");

        loaded_ = true;
    }

    DPUSubgraphRunner &XModelLoader::runner(const std::string &role)
    {
        return runners_.at(role);
    }

    const DPUSubgraphRunner &XModelLoader::runner(const std::string &role) const
    {
        return runners_.at(role);
    }

    bool XModelLoader::has_role(const std::string &role) const
    {
        return runners_.count(role) > 0;
    }

    DPUSubgraphRunner XModelLoader::create_duplicate_runner(
        const std::string& role, const std::string& alias) const
    {
        if (!loaded_)
            throw std::runtime_error("XModelLoader::create_duplicate_runner: not loaded");
        auto it = subgraphs_.find(role);
        if (it == subgraphs_.end())
            throw std::runtime_error(
                "XModelLoader::create_duplicate_runner: role '" + role + "' not found");
        return DPUSubgraphRunner(it->second, alias);
    }

} // namespace ddc

#endif // HAVE_DPU
