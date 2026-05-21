#pragma once
// RAII scope timer. Prints elapsed milliseconds to stderr on destruction.
// Usage:
//   {
//     ddc::ScopedTimer t("g_a forward");
//     runner.run(input);
//   }  // prints "[timer] g_a forward: 12 ms"

#include <chrono>
#include <cstdio>
#include <string>

namespace ddc {

struct ScopedTimer {
    std::string name_;
    std::chrono::steady_clock::time_point t0_;

    explicit ScopedTimer(std::string name)
        : name_(std::move(name))
        , t0_(std::chrono::steady_clock::now())
    {}

    ~ScopedTimer() {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t0_).count();
        fprintf(stderr, "[timer] %s: %ld ms\n", name_.c_str(), static_cast<long>(ms));
    }

    // Non-copyable, non-movable
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
};

} // namespace ddc
