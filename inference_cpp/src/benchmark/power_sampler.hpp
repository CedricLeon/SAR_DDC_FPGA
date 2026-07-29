#pragma once
// power_sampler.hpp — INA226 (sysfs) + PMBus (I2C) power sampler for benchmark_hardware.
//
// Two background threads poll the ZCU102 power rails:
//   INA226 thread : /sys/class/hwmon  (18 rails, default 10 ms interval)
//   PMBus thread  : /dev/i2c-4 I2C_RDWR ioctl (3 rails, default 40 ms interval)
//
// Compiles on any Linux host (plain POSIX + kernel headers).
// init() returns false when no /sys/class/hwmon INA226 sensors are found
// (off-board build), so --power cleanly no-ops everywhere.
//
// Intended usage:
//   PowerSampler ps;
//   if (!ps.init()) { /* no sensors; skip power */ }
//   PowerResult idle   = ps.idle_baseline(10.0);   // blocking
//   ps.start();
//   run_benchmark();
//   ps.stop();
//   PowerResult active = ps.results();

#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ddc {

// Per-rail statistics for one measurement window.
struct RailStats {
    double avg_power_w = 0.0;
    double energy_j    = 0.0;
    double duration_s  = 0.0;
    int    n_samples   = 0;
};

// Full power result from one measurement window (idle or active).
struct PowerResult {
    std::map<std::string, RailStats> rails;   // per-rail stats
    std::map<std::string, double>    groups;  // summed avg_power_w by group
    double duration_s = 0.0;                  // INA226 window duration
    bool   valid      = false;                // false if < 2 samples collected
};

// Samples ZCU102 power rails using INA226 sysfs and PMBus I2C in background threads.
class PowerSampler {
public:
    explicit PowerSampler(double ina226_interval_s = 0.01,
                          double pmbus_interval_s  = 0.04);
    ~PowerSampler();

    // Discover INA226 hwmon entries and probe PMBus VOUT_MODE exponent per rail.
    // Must be called before start(). Returns true if at least one INA226 was found.
    bool init();
    bool initialized() const { return initialized_; }

    void start();  // launch background threads; clears previous samples
    void stop();   // signal threads to exit and join them

    // Compute per-rail statistics from accumulated samples.
    // Call only after stop() (or accept approximate in-flight values with a lock).
    PowerResult results() const;

    // Blocking convenience: start → sleep(duration_s) → stop → results.
    // Calls init() first if not already initialized.
    PowerResult idle_baseline(double duration_s = 10.0);

private:
    void ina226_loop();
    void pmbus_loop();

    // PMBus I2C helpers
    static bool   rdwr(int fd, uint8_t dev_addr, uint8_t cmd, uint8_t* buf, int n);
    static double linear11(uint16_t raw);
    static double linear16(uint16_t raw, int exp);
    double pmbus_sample_rail(int fd, const std::string& rail) const;

    double ina226_interval_s_;
    double pmbus_interval_s_;
    bool   initialized_ = false;

    struct PmbusRailInfo { uint8_t addr; int vout_exp; };

    std::map<std::string, std::string>   ina226_paths_;  // rail → /sys/.../power1_input
    std::map<std::string, PmbusRailInfo> pmbus_rails_;   // rail → {addr, vout_exp}

    std::atomic<bool> running_{false};
    std::thread       ina226_thread_;
    std::thread       pmbus_thread_;

    mutable std::mutex mtx_;
    std::map<std::string, std::vector<double>> samples_w_;  // rail → [W] per sample
    // Window timestamps (seconds, steady_clock) for duration computation
    double t0_ina_ = 0.0, t1_ina_ = 0.0;
    double t0_pmb_ = 0.0, t1_pmb_ = 0.0;
};

} // namespace ddc
