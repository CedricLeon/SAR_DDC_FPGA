/* rans_profile.cpp — storage + at-exit dump for the entropy encode profiler.
 *
 * Entire TU is empty unless DDC_RANS_PROFILE is defined (see rans_profile.hpp).
 */
#include "rans_profile.hpp"

#ifdef DDC_RANS_PROFILE

#include <cstdio>

namespace ddc
{
    namespace rprof
    {

        Acc &acc(int ctx, int stage)
        {
            static Acc table[N_CTX][N_STAGE];
            return table[ctx][stage];
        }

        static thread_local int g_ctx = CTX_EB;
        int current_ctx() { return g_ctx; }
        void set_ctx(int ctx) { g_ctx = ctx; }

        namespace
        {
            // Prints the accumulated split once, at process teardown, to stderr.
            struct Dumper {
                ~Dumper()
                {
                    const char *cn[N_CTX] = {"EB", "GC"};
                    const char *sn[N_STAGE] = {"symbuild", "lookup", "flush"};

                    std::fprintf(stderr,
                                 "\n===== rANS encode profile (DDC_RANS_PROFILE) =====\n");
                    std::fprintf(stderr, "%-4s %-9s %12s %10s %12s\n",
                                 "ctx", "stage", "total_ms", "patches", "us/patch");
                    for (int c = 0; c < N_CTX; ++c)
                    {
                        double ctx_ms = 0.0;
                        unsigned long long ctx_calls = 0;
                        bool any = false;
                        for (int s = 0; s < N_STAGE; ++s)
                        {
                            Acc &a = acc(c, s);
                            unsigned long long calls =
                                (unsigned long long)a.calls.load(std::memory_order_relaxed);
                            if (calls == 0)
                                continue;
                            any = true;
                            double ms = a.ns.load(std::memory_order_relaxed) / 1e6;
                            double uspc = ms * 1000.0 / (double)calls;
                            std::fprintf(stderr, "%-4s %-9s %12.4f %10llu %12.4f\n",
                                         cn[c], sn[s], ms, calls, uspc);
                            ctx_ms += ms;
                            if (s == ST_SYM)
                                ctx_calls = calls; // one Scoped per compress() call
                        }
                        if (any && ctx_calls)
                            std::fprintf(stderr, "%-4s %-9s %12.4f %10llu %12.4f\n",
                                         cn[c], "TOTAL", ctx_ms, ctx_calls,
                                         ctx_ms * 1000.0 / (double)ctx_calls);
                    }
                    std::fprintf(stderr,
                                 "==================================================\n");
                }
            };
            static Dumper g_dumper;
        } // namespace

    } // namespace rprof
} // namespace ddc

#endif // DDC_RANS_PROFILE
