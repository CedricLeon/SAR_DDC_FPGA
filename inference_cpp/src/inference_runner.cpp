/* inference_runner.cpp
 * C++ port of scripts/fpga/inference_hybrid.py run_hybrid_inference().
 *
 * Pipeline for one 256×256 patch (ScaleHyperprior path):
 *   0. normalize raw complex -> norm_logI  (CPU)
 *   1. g_a(real), g_a(imag)               (DPU)
 *   2. concatenate y = interleaved NHWC [y_real, y_imag per spatial pos] (CPU)
 *   3. h_a(|y|)                            (DPU)
 *   4. EB compress/decompress z            (CPU rANS)
 *   5. h_s(z_hat)                          (DPU) -> scales
 *   6. GC compress/decompress y            (CPU rANS)
 *   7. g_s(y_hat_real), g_s(y_hat_imag)   (DPU)
 *   8. denormalize recon -> linA           (CPU)
 *
 * FactorizedPrior path omits steps 3/5/6 (just EB on y).
 */

#include "inference_runner.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "constants.hpp"
#include "entropy_models.hpp"
#include "logger.hpp"
#include "metrics.hpp"
#include "npy_io.hpp"
#include "patch_transforms.hpp"
#include "scoped_timer.hpp"

#ifdef HAVE_DPU
#include "dpu_runners.hpp"
#endif

namespace ddc
{

    // ---------------------------------------------------------------------------
    // helpers
    // ---------------------------------------------------------------------------
    namespace
    {

        // Sigmoid blend ramp, length n, renormalised to [0,1].
        std::vector<float> sigmoid_ramp(int n, float alpha = 6.0f)
        {
            std::vector<float> r(n);
            for (int i = 0; i < n; ++i)
            {
                float t = static_cast<float>(i) / static_cast<float>(n - 1);
                r[i] = 1.0f / (1.0f + std::exp(-alpha * (t - 0.5f)));
            }
            float r0 = r[0], r1 = r[n - 1];
            for (auto &v : r)
                v = (v - r0) / (r1 - r0 + 1e-12f);
            return r;
        }

        // Build the list of patch top-left offsets for one dimension.
        std::vector<int> make_offsets(int dim, int patch, int stride)
        {
            std::vector<int> offs;
            for (int o = 0; o + patch <= dim; o += stride)
                offs.push_back(o);
            int last = dim - patch;
            if (offs.empty() || offs.back() != last)
                offs.push_back(last);
            return offs;
        }

        // Save a flat float array as NPY (shape = {H, W}).
        [[maybe_unused]] void save_float_hw(const std::filesystem::path &p, const float *data, int H, int W)
        {
            npy_save_float32(p.string(), data, {static_cast<size_t>(H), static_cast<size_t>(W)});
        }

        // Log per-tensor statistics (min/max/mean/std/sum) for verbose step-by-step debugging.
        [[maybe_unused]] void log_stats(const std::string &label, const float *data, size_t n)
        {
            if (n == 0)
                return;
            float mn = data[0], mx = data[0], sum = 0.f;
            for (size_t i = 0; i < n; ++i)
            {
                mn = std::min(mn, data[i]);
                mx = std::max(mx, data[i]);
                sum += data[i];
            }
            float mean = sum / static_cast<float>(n);
            float sq = 0.f;
            for (size_t i = 0; i < n; ++i)
                sq += (data[i] - mean) * (data[i] - mean);
            LOG_INFO("  [" + label + "] n=" + std::to_string(n) + " min=" + std::to_string(mn) + " max=" + std::to_string(mx) + " mean=" + std::to_string(mean) + " std=" + std::to_string(std::sqrt(sq / static_cast<float>(n))) + " sum=" + std::to_string(sum));
        }

    } // anonymous namespace

    // ---------------------------------------------------------------------------
    // InferencePipeline — internal object that holds loaded models and runs patches
    // ---------------------------------------------------------------------------
    struct PatchResult
    {
        std::vector<float> recon_norm_logI; // [H*W*2] normalised log-intensity
        int num_bytes = 0;
        int z_bytes = 0; // EB bytes for z; 0 for FactorizedPrior
        int y_bytes = 0; // GC bytes for y (SHyp) or EB bytes (FP)
    };

    class InferencePipeline
    {
    public:
        explicit InferencePipeline(const InferenceConfig &cfg) : cfg_(cfg)
        {
            // Load entropy models
            LOG_INFO("Loading entropy params from " + cfg_.params_dir.string());
            eb_.load_params(cfg_.params_dir);
            LOG_INFO("EntropyBottleneck loaded, C=" + std::to_string(eb_.channels()));

            // Attempt GC load (only present for ScaleHyperprior models)
            auto gc_scale = cfg_.params_dir / "gc_scale_table.npy";
            if (std::filesystem::exists(gc_scale))
            {
                gc_.load_params(cfg_.params_dir);
                has_gc_ = gc_.is_loaded();
                LOG_INFO("GaussianConditional loaded.");
            }
            else
            {
                LOG_INFO("No gc_scale_table.npy — using FactorizedPrior path.");
            }

#ifdef HAVE_DPU
            LOG_INFO("Loading xmodel from " + cfg_.xmodel_path.string());
            auto meta_json = cfg_.xmodel_path.parent_path() / "meta.json";
            loader_.load(cfg_.xmodel_path.string(), meta_json.string());
#else
            LOG_INFO("[STUB] HAVE_DPU not set — DPU calls will throw.");
#endif
        }

        bool uses_hyper() const { return has_gc_; }

        // Run one 256x256 patch (HWC, raw complex float).
        // Returns normalised log-intensity [H*W*2] and total compressed bytes.
        PatchResult run_patch(const float *noisy_hwc, int H, int W) const
        {
#ifndef HAVE_DPU
            (void)noisy_hwc;
            (void)H;
            (void)W;
            throw std::runtime_error("run_patch: compiled without HAVE_DPU");
#else
            return has_gc_ ? _run_shyp(noisy_hwc, H, W) : _run_fp(noisy_hwc, H, W);
#endif
        }

    private:
#ifdef HAVE_DPU
        // ScaleHyperprior: g_a -> h_a -> EB -> h_s -> GC -> g_s
        PatchResult _run_shyp(const float *noisy_hwc, int H, int W) const
        {
            // 0. normalise
            std::vector<float> norm(H * W * 2);
            normalize_patch(noisy_hwc, norm.data(), H, W);

            // Split into real / imag channels: each [1, H, W, 1] = NHWC with N=1, C=1
            // DPU runner expects flat NHWC so we pass [H*W] floats for C=1.
            std::vector<float> real_ch(H * W), imag_ch(H * W);
            for (int i = 0; i < H * W; ++i)
            {
                real_ch[i] = norm[i * 2 + 0];
                imag_ch[i] = norm[i * 2 + 1];
            }

            // 1. g_a
            const auto &ga = loader_.runner("g_a");
            std::vector<float> y_real(ga.output_numel()), y_imag(ga.output_numel());
            ga.run(real_ch.data(), y_real.data());
            ga.run(imag_ch.data(), y_imag.data());
            if (ddc::Logger::instance().is_verbose())
            {
                log_stats("y_real (g_a)", y_real.data(), y_real.size());
                log_stats("y_imag (g_a)", y_imag.data(), y_imag.size());
            }

            // 2. combine y = interleaved NHWC [y_real, y_imag per spatial pos]
            // Python: np.concatenate((y_real, y_imag), axis=-1) → [N, H', W', 2*C_MAIN]
            // h_a and GC both require NHWC with C=2*C_MAIN; block layout causes wrong
            // spatial-channel combinations in h_a and mismatched y/scales indices in GC.
            const auto &y_shape = ga.output_shape(); // [N, H', W', C_MAIN]
            const int yh = y_shape[1], yw = y_shape[2], yc = y_shape[3];
            const int y_len = yh * yw * yc;
            std::vector<float> y(y_len * 2), y_abs(y_len * 2);
            for (int hw = 0; hw < yh * yw; ++hw)
            {
                for (int c = 0; c < yc; ++c)
                {
                    y[hw * 2 * yc + c] = y_real[hw * yc + c];      // channels [0, C_MAIN)
                    y[hw * 2 * yc + yc + c] = y_imag[hw * yc + c]; // channels [C_MAIN, 2*C_MAIN)
                }
            }
            for (int i = 0; i < y_len * 2; ++i)
                y_abs[i] = std::abs(y[i]);
            if (ddc::Logger::instance().is_verbose())
            {
                log_stats("y (interleaved)", y.data(), y.size());
                log_stats("y_abs", y_abs.data(), y_abs.size());
            }

            // 3. h_a
            const auto &ha = loader_.runner("h_a");
            std::vector<float> z(ha.output_numel());
            ha.run(y_abs.data(), z.data());
            if (ddc::Logger::instance().is_verbose())
                log_stats("z (h_a)", z.data(), z.size());

            // 4. EB compress/decompress z
            // z shape is [1, H_hyper, W_hyper, C_hyper] — get H/W from runner
            const auto &z_shape = ha.output_shape(); // [N, H', W', C]
            int zh = z_shape[1], zw = z_shape[2];
            std::vector<uint8_t> z_bits = eb_.compress(z.data(), zh, zw);
            std::vector<float> z_hat = eb_.decompress(z_bits, zh, zw);
            int z_bytes = static_cast<int>(z_bits.size());
            if (ddc::Logger::instance().is_verbose())
            {
                LOG_INFO("  [z_bits] bytes=" + std::to_string(z_bytes));
                log_stats("z_hat (EB)", z_hat.data(), z_hat.size());
            }

            // 5. h_s(z_hat) -> scales
            const auto &hs = loader_.runner("h_s");
            std::vector<float> scales(hs.output_numel());
            hs.run(z_hat.data(), scales.data());
            if (ddc::Logger::instance().is_verbose())
                log_stats("scales (h_s)", scales.data(), scales.size());

            // 6. GC compress/decompress y (means = 0)
            // y_shape, yh, yw, yc already set in step 2; y and scales both NHWC C=2*C_MAIN ✓
            std::vector<float> means(y_len * 2, 0.0f);
            std::vector<uint8_t> y_bits = gc_.compress(y.data(), scales.data(), means.data(),
                                                       yh, yw, C_MAIN * 2);
            std::vector<float> y_hat = gc_.decompress(y_bits, scales.data(), means.data(),
                                                      yh, yw, C_MAIN * 2);
            int y_bytes = static_cast<int>(y_bits.size());
            if (ddc::Logger::instance().is_verbose())
            {
                LOG_INFO("  [y_bits] bytes=" + std::to_string(y_bytes));
                log_stats("y_hat (GC)", y_hat.data(), y_hat.size());
            }

            // 7. g_s — de-interleave y_hat back to real/imag (reverse of step 2 interleaving)
            std::vector<float> yh_real(y_len), yh_imag(y_len);
            for (int hw = 0; hw < yh * yw; ++hw)
            {
                for (int c = 0; c < yc; ++c)
                {
                    yh_real[hw * yc + c] = y_hat[hw * 2 * yc + c];
                    yh_imag[hw * yc + c] = y_hat[hw * 2 * yc + yc + c];
                }
            }
            const auto &gs = loader_.runner("g_s");
            std::vector<float> recon_real(gs.output_numel()), recon_imag(gs.output_numel());
            gs.run(yh_real.data(), recon_real.data());
            gs.run(yh_imag.data(), recon_imag.data());
            if (ddc::Logger::instance().is_verbose())
            {
                log_stats("recon_real (g_s)", recon_real.data(), recon_real.size());
                log_stats("recon_imag (g_s)", recon_imag.data(), recon_imag.size());
            }

            // 8. pack real/imag back into HW2
            PatchResult res;
            res.num_bytes = z_bytes + y_bytes;
            res.z_bytes = z_bytes;
            res.y_bytes = y_bytes;
            res.recon_norm_logI.resize(H * W * 2);
            for (int i = 0; i < H * W; ++i)
            {
                res.recon_norm_logI[i * 2 + 0] = recon_real[i];
                res.recon_norm_logI[i * 2 + 1] = recon_imag[i];
            }
            return res;
        }

        // FactorizedPrior: g_a -> EB -> g_s
        PatchResult _run_fp(const float *noisy_hwc, int H, int W) const
        {
            std::vector<float> norm(H * W * 2);
            normalize_patch(noisy_hwc, norm.data(), H, W);

            std::vector<float> real_ch(H * W), imag_ch(H * W);
            for (int i = 0; i < H * W; ++i)
            {
                real_ch[i] = norm[i * 2 + 0];
                imag_ch[i] = norm[i * 2 + 1];
            }

            const auto &ga = loader_.runner("g_a");
            std::vector<float> y_real(ga.output_numel()), y_imag(ga.output_numel());
            ga.run(real_ch.data(), y_real.data());
            ga.run(imag_ch.data(), y_imag.data());

            // Interleaved NHWC — maps each real/imag pair to the correct EB CDF channel.
            // Block layout tested empirically: +4 bytes on 40/100 patches, same MERLIN quality.
            const auto &y_shape = ga.output_shape(); // [N, H', W', C_MAIN]
            const int yh = y_shape[1], yw = y_shape[2], yc = y_shape[3];
            const int y_len = yh * yw * yc;
            std::vector<float> y(y_len * 2);
            for (int hw = 0; hw < yh * yw; ++hw)
            {
                for (int c = 0; c < yc; ++c)
                {
                    y[hw * 2 * yc + c]      = y_real[hw * yc + c];
                    y[hw * 2 * yc + yc + c] = y_imag[hw * yc + c];
                }
            }

            std::vector<uint8_t> y_bits = eb_.compress(y.data(), yh, yw);
            std::vector<float> y_hat = eb_.decompress(y_bits, yh, yw);

            std::vector<float> yh_real(y_len), yh_imag(y_len);
            for (int hw = 0; hw < yh * yw; ++hw)
            {
                for (int c = 0; c < yc; ++c)
                {
                    yh_real[hw * yc + c] = y_hat[hw * 2 * yc + c];
                    yh_imag[hw * yc + c] = y_hat[hw * 2 * yc + yc + c];
                }
            }

            const auto &gs = loader_.runner("g_s");
            std::vector<float> recon_real(gs.output_numel()), recon_imag(gs.output_numel());
            gs.run(yh_real.data(), recon_real.data());
            gs.run(yh_imag.data(), recon_imag.data());

            PatchResult res;
            res.num_bytes = static_cast<int>(y_bits.size());
            res.z_bytes = 0;
            res.y_bytes = res.num_bytes;
            res.recon_norm_logI.resize(H * W * 2);
            for (int i = 0; i < H * W; ++i)
            {
                res.recon_norm_logI[i * 2 + 0] = recon_real[i];
                res.recon_norm_logI[i * 2 + 1] = recon_imag[i];
            }
            return res;
        }

        mutable XModelLoader loader_;
#endif

        InferenceConfig cfg_;
        EntropyBottleneck eb_;
        GaussianConditional gc_;
        bool has_gc_ = false;
    };

    // ---------------------------------------------------------------------------
    // overlap-blended tiling (port of patch_infer_fpga)
    // image_hwc: [H, W, 2] float, raw complex
    // Returns: recon_norm_logI [H, W, 2] and total_bytes
    // ---------------------------------------------------------------------------
    static std::pair<std::vector<float>, int> tile_infer(
        const float *image_hwc, int H, int W,
        const InferencePipeline &pipeline,
        int patch_size = IMAGE_SIZE, int overlap = 16)
    {
        const int stride = patch_size - overlap;
        auto row_offs = make_offsets(H, patch_size, stride);
        auto col_offs = make_offsets(W, patch_size, stride);

        LOG_INFO("tile_infer: " + std::to_string(H) + "x" + std::to_string(W) + " -> " + std::to_string(row_offs.size() * col_offs.size()) + " patches");

        // canvas: [H, W, 2] weighted sum; weight_canvas: [H, W]
        std::vector<float> canvas(H * W * 2, 0.0f);
        std::vector<float> weight_canvas(H * W, 0.0f);
        int total_bytes = 0;

        auto ramp = sigmoid_ramp(overlap);

        for (int ro : row_offs)
        {
            for (int co : col_offs)
            {
                // Extract patch [patch_size, patch_size, 2]
                std::vector<float> patch(patch_size * patch_size * 2);
                for (int r = 0; r < patch_size; ++r)
                    std::memcpy(patch.data() + r * patch_size * 2,
                                image_hwc + (ro + r) * W * 2 + co * 2,
                                patch_size * 2 * sizeof(float));

                PatchResult res = pipeline.run_patch(patch.data(), patch_size, patch_size);
                total_bytes += res.num_bytes;

                // Build 2-D weight map (sigmoid ramp at leading edges, 1 elsewhere)
                std::vector<float> wmap(patch_size * patch_size, 1.0f);

                bool touch_top = (ro == 0);
                bool touch_bottom = (ro + patch_size == H);
                bool touch_left = (co == 0);
                bool touch_right = (co + patch_size == W);

                // Apply ramps row-wise (vertical) and column-wise (horizontal)
                for (int r = 0; r < patch_size; ++r)
                {
                    float vw = 1.0f;
                    if (!touch_top && r < overlap)
                        vw = ramp[r];
                    if (!touch_bottom && r >= patch_size - overlap)
                        vw = ramp[patch_size - 1 - r];
                    for (int c = 0; c < patch_size; ++c)
                    {
                        float hw = 1.0f;
                        if (!touch_left && c < overlap)
                            hw = ramp[c];
                        if (!touch_right && c >= patch_size - overlap)
                            hw = ramp[patch_size - 1 - c];
                        wmap[r * patch_size + c] = vw * hw;
                    }
                }

                // Accumulate onto canvas
                for (int r = 0; r < patch_size; ++r)
                {
                    for (int c = 0; c < patch_size; ++c)
                    {
                        int img_i = (ro + r) * W + (co + c);
                        int patch_i = r * patch_size + c;
                        float w = wmap[patch_i];
                        canvas[img_i * 2 + 0] += res.recon_norm_logI[patch_i * 2 + 0] * w;
                        canvas[img_i * 2 + 1] += res.recon_norm_logI[patch_i * 2 + 1] * w;
                        weight_canvas[img_i] += w;
                    }
                }
            }
        }

        // Normalize by accumulated weights
        for (int i = 0; i < H * W; ++i)
        {
            float w = weight_canvas[i];
            if (w > 1e-12f)
            {
                canvas[i * 2 + 0] /= w;
                canvas[i * 2 + 1] /= w;
            }
        }

        return {canvas, total_bytes};
    }

    // ---------------------------------------------------------------------------
    // InferenceRunner implementation
    // ---------------------------------------------------------------------------
    InferenceRunner::InferenceRunner(const InferenceConfig &cfg) : cfg_(cfg) {}

    void InferenceRunner::run()
    {
        // Clean and recreate output dir
        if (std::filesystem::exists(cfg_.output_dir))
            std::filesystem::remove_all(cfg_.output_dir);
        std::filesystem::create_directories(cfg_.output_dir);

        // Write inference_meta.json
        {
            nlohmann::json meta;
            auto manifest = cfg_.xmodel_path.parent_path() / "manifest.json";
            if (std::filesystem::exists(manifest))
            {
                std::ifstream f(manifest);
                nlohmann::json build;
                f >> build;
                meta["model_run_name"] = build.value("model_name", "Unknown");
                meta["model_compiled_at"] = build.value("compiled_at", "Unknown");
            }
            // ISO-8601 timestamp
            auto now = std::chrono::system_clock::now();
            std::time_t t = std::chrono::system_clock::to_time_t(now);
            std::ostringstream ts;
            ts << std::put_time(std::gmtime(&t), "%Y-%m-%dT%H:%M:%SZ");
            meta["evaluated_at"] = ts.str();

            std::ofstream f(cfg_.output_dir / "inference_meta.json");
            f << meta.dump(4);
        }

        LOG_INFO("=== C++ Hybrid Inference ===");
        LOG_INFO("xmodel:  " + cfg_.xmodel_path.string());
        LOG_INFO("params:  " + cfg_.params_dir.string());
        LOG_INFO("data:    " + cfg_.data_path.string());
        LOG_INFO("output:  " + cfg_.output_dir.string());
        LOG_INFO("subset:  " + std::to_string(cfg_.subset));

        InferencePipeline pipeline(cfg_);

        if (!cfg_.skip_test_set)
            _run_test_subset_impl(pipeline);
        else
            LOG_INFO("Test-subset phase skipped (--skip-test-set).");
        _run_tile_eval_impl(pipeline);
    }

    // ---------------------------------------------------------------------------
    // _run_test_subset_impl
    // ---------------------------------------------------------------------------
    void InferenceRunner::_run_test_subset_impl(InferencePipeline &pipeline)
    {
        ScopedTimer t("test_subset");

        // Load test NPY: shape (N, H, W, 4)
        //   channels: 0=real, 1=imag, 2=adam_noc_linA, 3=merlin_linA
        NpyArray arr = npy_load(cfg_.data_path.string());
        if (arr.shape.size() != 4 || arr.shape[3] != 4)
            throw std::runtime_error("Expected data shape (N,H,W,4), got mismatch");

        const int N = static_cast<int>(std::min(static_cast<size_t>(cfg_.subset), arr.shape[0]));
        const int H = static_cast<int>(arr.shape[1]);
        const int W = static_cast<int>(arr.shape[2]);
        auto data = arr.as_float32(); // flat (N,H,W,4)

        LOG_INFO("Test subset: N=" + std::to_string(N) + " H=" + std::to_string(H) + " W=" + std::to_string(W));

        MetricsAccumulator acc_noisy, acc_adam, acc_merlin;

        // Visualisation: 5 evenly spaced indices
        std::vector<int> vis_idx;
        for (int k = 0; k < 5; ++k)
            vis_idx.push_back(static_cast<int>((static_cast<double>(k) / 4.0) * (N - 1)));

        std::vector<float> vis_noisy, vis_recon, vis_adam, vis_merlin; // log-I, HW each

        auto t0_all = std::chrono::steady_clock::now();

        for (int i = 0; i < N; ++i)
        {
            auto t0 = std::chrono::steady_clock::now();

            const float *sample = data + i * H * W * 4;

            // Extract raw complex [H, W, 2]
            std::vector<float> noisy_hwc(H * W * 2);
            for (int p = 0; p < H * W; ++p)
            {
                noisy_hwc[p * 2 + 0] = sample[p * 4 + 0]; // real
                noisy_hwc[p * 2 + 1] = sample[p * 4 + 1]; // imag
            }

            // Enable verbose logging only for the debug patch (if requested).
            bool debug_this_patch = (cfg_.debug_patch >= 0 && i == cfg_.debug_patch);
            if (debug_this_patch)
            {
                ddc::Logger::instance().set_verbose(true);
                LOG_INFO("=== DEBUG PATCH " + std::to_string(i) + " ===");
            }
            PatchResult res = pipeline.run_patch(noisy_hwc.data(), H, W);
            if (debug_this_patch && !cfg_.verbose)
                ddc::Logger::instance().set_verbose(false);

            // Denorm recon -> linA [H, W]
            std::vector<float> recon_lina(H * W);
            denorm_to_lina(res.recon_norm_logI.data(), recon_lina.data(), H, W);

            // Noisy -> linA
            std::vector<float> noisy_lina(H * W);
            raw_to_lina(noisy_hwc.data(), noisy_lina.data(), H, W);

            // Ground truths — already linA in channels 2 and 3
            std::vector<float> adam_lina(H * W), merlin_lina(H * W);
            for (int p = 0; p < H * W; ++p)
            {
                adam_lina[p] = sample[p * 4 + 2];
                merlin_lina[p] = sample[p * 4 + 3];
            }

            acc_noisy.update(recon_lina.data(), noisy_lina.data(), H, W, res.num_bytes);
            acc_adam.update(recon_lina.data(), adam_lina.data(), H, W, res.num_bytes);
            acc_merlin.update(recon_lina.data(), merlin_lina.data(), H, W, res.num_bytes);

            // Visualisation (log-I)
            if (std::find(vis_idx.begin(), vis_idx.end(), i) != vis_idx.end())
            {
                // noisy log-I: log(real^2 + imag^2 + EPS)
                std::vector<float> noisy_logI(H * W);
                for (int p = 0; p < H * W; ++p)
                {
                    float r = noisy_hwc[p * 2 + 0], im = noisy_hwc[p * 2 + 1];
                    noisy_logI[p] = std::log(r * r + im * im + static_cast<float>(EPS));
                }
                // recon log-I: log(linI + EPS) where linI = linA^2
                std::vector<float> recon_logI(H * W);
                for (int p = 0; p < H * W; ++p)
                {
                    float la = recon_lina[p];
                    recon_logI[p] = std::log(la * la + static_cast<float>(EPS));
                }
                // adam/merlin log-I
                std::vector<float> adam_logI(H * W), merlin_logI(H * W);
                for (int p = 0; p < H * W; ++p)
                {
                    float a = adam_lina[p], m = merlin_lina[p];
                    adam_logI[p] = std::log(a * a + static_cast<float>(EPS));
                    merlin_logI[p] = std::log(m * m + static_cast<float>(EPS));
                }
                vis_noisy.insert(vis_noisy.end(), noisy_logI.begin(), noisy_logI.end());
                vis_recon.insert(vis_recon.end(), recon_logI.begin(), recon_logI.end());
                vis_adam.insert(vis_adam.end(), adam_logI.begin(), adam_logI.end());
                vis_merlin.insert(vis_merlin.end(), merlin_logI.begin(), merlin_logI.end());
            }

            if (i % 10 == 0)
            {
                auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now() - t0)
                              .count();
                LOG_INFO("Sample " + std::to_string(i) + " processed in " + std::to_string(dt) + " ms");
            }
        }

        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - t0_all)
                            .count();
        LOG_INFO("Test subset done in " + std::to_string(total_ms) + " ms (" + std::to_string(total_ms / N) + " ms/sample)");

        // Build JSON summary (mirrors Python run_hybrid_inference output)
        auto noisy_m = acc_noisy.mean();
        auto adam_m = acc_adam.mean();
        auto merlin_m = acc_merlin.mean();

        auto to_json = [](const Metrics &m, bool include_ratio)
        {
            nlohmann::json j;
            j["bpp"] = m.bpp;
            j["mse"] = m.mse;
            j["psnr"] = m.psnr;
            j["ssim"] = m.ssim;
            if (include_ratio)
            {
                j["enl"] = m.enl;
                j["ratio_mean"] = m.ratio_mean;
                j["ratio_enl"] = m.ratio_enl;
            }
            else
            {
                j["epd"] = m.epd;
            }
            return j;
        };

        nlohmann::json summary;
        summary["Noisy"] = to_json(noisy_m, true);
        summary["ADAM"] = to_json(adam_m, false);
        summary["MERLIN"] = to_json(merlin_m, false);
        summary["recon"] = {
            {"enl", noisy_m.enl},
            {"ratio_mean", noisy_m.ratio_mean},
            {"ratio_enl", noisy_m.ratio_enl},
        };

        LOG_INFO("=== Test Set Results ===");
        LOG_INFO("  bpp:  " + std::to_string(merlin_m.bpp));
        LOG_INFO("  psnr(MERLIN): " + std::to_string(merlin_m.psnr));
        LOG_INFO("  ssim(MERLIN): " + std::to_string(merlin_m.ssim));

        {
            std::ofstream f(cfg_.output_dir / "metrics.json");
            f << summary.dump(4);
        }

        // Save vis arrays
        auto vis_dir = cfg_.output_dir / "reconstructions_test_set";
        std::filesystem::create_directories(vis_dir);
        size_t nv = vis_idx.size();
        if (!vis_noisy.empty())
        {
            npy_save_float32((vis_dir / "vis_noisy.npy").string(), vis_noisy.data(),
                             {nv, static_cast<size_t>(H), static_cast<size_t>(W)});
            npy_save_float32((vis_dir / "vis_recon.npy").string(), vis_recon.data(),
                             {nv, static_cast<size_t>(H), static_cast<size_t>(W)});
            npy_save_float32((vis_dir / "vis_adam.npy").string(), vis_adam.data(),
                             {nv, static_cast<size_t>(H), static_cast<size_t>(W)});
            npy_save_float32((vis_dir / "vis_merlin.npy").string(), vis_merlin.data(),
                             {nv, static_cast<size_t>(H), static_cast<size_t>(W)});
        }
    }

    // ---------------------------------------------------------------------------
    // _run_tile_eval_impl
    // ---------------------------------------------------------------------------
    void InferenceRunner::_run_tile_eval_impl(InferencePipeline &pipeline)
    {
        namespace fs = std::filesystem;
        auto root = cfg_.data_path.parent_path();

        LOG_INFO("Scanning " + root.string() + " for large-tile directories");

        for (const auto &entry : fs::directory_iterator(root))
        {
            if (!entry.is_directory())
                continue;

            auto noisy_path = entry.path() / "sym_Noisy.npy";
            if (!fs::exists(noisy_path))
            {
                LOG_VERBOSE("Skipping " + entry.path().filename().string() + " (no sym_Noisy.npy)");
                continue;
            }

            LOG_INFO("--- Tile: " + entry.path().filename().string() + " ---");
            ScopedTimer ttile("tile:" + entry.path().filename().string());

            NpyArray tile_arr = npy_load(noisy_path.string());
            if (tile_arr.shape.size() != 3 || tile_arr.shape[2] != 2)
                throw std::runtime_error("sym_Noisy.npy must be (H, W, 2), got unexpected shape");

            const int TH = static_cast<int>(tile_arr.shape[0]);
            const int TW = static_cast<int>(tile_arr.shape[1]);
            // to_float32_vec() handles both float32 and float64 on-disk dtypes.
            // sym_Noisy.npy is saved as float64 by numpy; as_float32() would
            // reinterpret the raw bytes and produce garbage — hence this call.
            auto tile_data_vec = tile_arr.to_float32_vec();
            const float *tile_data = tile_data_vec.data();

            // Overlap-blended inference
            auto [recon_norm, total_bytes] = tile_infer(tile_data, TH, TW, pipeline);

            // Denorm -> linA [H, W]
            std::vector<float> recon_lina(TH * TW);
            denorm_to_lina(recon_norm.data(), recon_lina.data(), TH, TW);

            // Noisy -> linA
            std::vector<float> noisy_lina(TH * TW);
            raw_to_lina(tile_data, noisy_lina.data(), TH, TW);

            double bpp = compute_bpp(total_bytes, TH, TW);

            nlohmann::json tile_metrics;
            tile_metrics["bpp"] = bpp;
            tile_metrics["mse_noisy"] = compute_mse(recon_lina.data(), noisy_lina.data(), TH * TW);
            tile_metrics["psnr_noisy"] = compute_psnr(recon_lina.data(), noisy_lina.data(), TH * TW);

            // Reference GT metrics
            static const std::vector<std::pair<std::string, std::string>> refs = {
                {"MERLIN", "linA_MERLIN.npy"},
                {"ADAM-NOC", "linA_ADAM_NOC.npy"},
                {"MERLIN_DDS", "linA_MERLIN_DDS.npy"},
            };
            for (const auto &[ref_name, ref_file] : refs)
            {
                auto ref_path = entry.path() / ref_file;
                if (!fs::exists(ref_path))
                {
                    LOG_INFO("  Missing " + ref_file + ", skipping " + ref_name + " metrics.");
                    continue;
                }
                auto ref_arr = npy_load_float32(ref_path.string()); // flat [H*W]
                tile_metrics["psnr_" + ref_name] = compute_psnr(recon_lina.data(), ref_arr.data(), TH * TW);
                tile_metrics["mse_" + ref_name] = compute_mse(recon_lina.data(), ref_arr.data(), TH * TW);
                tile_metrics["ssim_" + ref_name] = compute_ssim(recon_lina.data(), ref_arr.data(), TH, TW);
                tile_metrics["epd_" + ref_name] = compute_epd(recon_lina.data(), ref_arr.data(), TH, TW);
            }

            // Reference-free SAR metrics
            Roi hamburg_roi = {HAMBURG_ENL_ROI[0], HAMBURG_ENL_ROI[1],
                               HAMBURG_ENL_ROI[2], HAMBURG_ENL_ROI[3]};
            tile_metrics["enl_recon"] = compute_enl(recon_lina.data(), TH, TW);
            tile_metrics["enl_roi"] = compute_enl(recon_lina.data(), TH, TW, &hamburg_roi);
            tile_metrics["ratio_mean"] = compute_ratio_mean(recon_lina.data(), noisy_lina.data(), TH * TW);
            tile_metrics["ratio_enl"] = compute_ratio_enl(recon_lina.data(), noisy_lina.data(), TH * TW);

            LOG_INFO("  bpp=" + std::to_string(bpp) + " psnr_noisy=" + std::to_string(tile_metrics["psnr_noisy"].get<double>()));

            // Save recon linA
            auto save_recon = cfg_.output_dir / (entry.path().filename().string() + "_recon_linA.npy");
            npy_save_float32(save_recon.string(), recon_lina.data(),
                             {static_cast<size_t>(TH), static_cast<size_t>(TW)});

            // Save tile metrics JSON
            auto meta_path = cfg_.output_dir / (entry.path().filename().string() + "_metrics.json");
            std::ofstream f(meta_path);
            f << tile_metrics.dump(4);

            LOG_INFO("  Saved recon -> " + save_recon.string());
        }
        LOG_INFO("Tile scan complete.");
    }

} // namespace ddc
