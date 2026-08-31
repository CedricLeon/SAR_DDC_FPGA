/* rans_profile.hpp — opt-in sub-stage timing for the entropy encode path.
 *
 * Compiled in ONLY when DDC_RANS_PROFILE is defined (CMake: -DDDC_RANS_PROFILE=ON).
 * Otherwise every macro below is a no-op: zero instructions, zero footprint, the
 * production bitstream path is byte-for-byte unchanged.
 *
 * Purpose (N6, docs/onboard_pipeline.md): split the per-patch entropy cost into
 *   symbuild — round(z - median) / scale_index() symbol + index construction
 *   lookup   — the forward CDF lookup pass that builds _syms (encode_with_indexes)
 *   flush    — the reverse rANS renorm loop (Rans64EncPut) + output byte copy
 * separately for the EntropyBottleneck (EB) and GaussianConditional (GC) coders.
 */
#pragma once

#ifdef DDC_RANS_PROFILE

#include <atomic>
#include <chrono>
#include <cstdint>

namespace ddc
{
    namespace rprof
    {

        enum Ctx { CTX_EB = 0, CTX_GC = 1, N_CTX = 2 };
        enum Stage { ST_SYM = 0, ST_LOOKUP = 1, ST_FLUSH = 2, N_STAGE = 3 };

        struct Acc {
            std::atomic<uint64_t> ns{0};
            std::atomic<uint64_t> calls{0};
        };

        // One accumulator per (context, stage) — defined in rans_profile.cpp.
        Acc &acc(int ctx, int stage);

        // Thread-local "which coder is running" so the shared rANS core
        // (encode_with_indexes / flush) attributes to EB vs GC.
        int current_ctx();
        void set_ctx(int ctx);

        using clk = std::chrono::steady_clock;

        // Accumulates elapsed ns + 1 call into acc(ctx, stage) when it leaves scope.
        struct Scoped {
            Acc &a;
            clk::time_point t0;
            Scoped(int ctx, int stage) : a(acc(ctx, stage)), t0(clk::now()) {}
            ~Scoped()
            {
                a.ns.fetch_add(static_cast<uint64_t>(
                                   std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       clk::now() - t0)
                                       .count()),
                               std::memory_order_relaxed);
                a.calls.fetch_add(1, std::memory_order_relaxed);
            }
        };

    } // namespace rprof
} // namespace ddc

// Time the enclosing scope into an explicit (ctx, stage).
#define DDC_RPROF(ctx, stage) ::ddc::rprof::Scoped _rprof_guard((ctx), (stage))
// Time the enclosing scope into (current thread-local ctx, stage).
#define DDC_RPROF_CUR(stage) ::ddc::rprof::Scoped _rprof_guard(::ddc::rprof::current_ctx(), (stage))
// Set the thread-local coder context.
#define DDC_RPROF_SETCTX(ctx) ::ddc::rprof::set_ctx((ctx))

#else // !DDC_RANS_PROFILE

#define DDC_RPROF(ctx, stage) do {} while (0)
#define DDC_RPROF_CUR(stage) do {} while (0)
#define DDC_RPROF_SETCTX(ctx) do {} while (0)

#endif // DDC_RANS_PROFILE
