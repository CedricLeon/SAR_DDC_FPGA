/* power_sampler.cpp — INA226 sysfs + PMBus I2C power sampler.
 *
 * INA226 thread: reads /sys/class/hwmon/hwmonN/power1_input (µW as integer
 *   string) every ina226_interval_s (default 10 ms). Converts µW → W.
 *
 * PMBus thread: reads READ_VOUT (Linear16) + READ_IOUT (Linear11) per rail
 *   via /dev/i2c-4 I2C_RDWR ioctl every pmbus_interval_s (default 40 ms).
 *   Power computed as V × I. VOUT_MODE exponent read once in init().
 *
 * Thread pacing: both loops sleep the *remainder* of the poll interval after
 * each batch read.  Never busy-wait — busy-wait pins an A53 core and inflates
 * PS power readings.
 *
 * Sensor map and group formulas mirror benchmark_fpga.py (ZCU102_SENSOR_MAP,
 * POWER_GROUPS, PMBUS_RAIL_MAP) so results are directly comparable.
 */

#include "power_sampler.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <thread>

#include <fcntl.h>       // open, O_RDWR
#include <linux/i2c.h>   // struct i2c_msg, I2C_M_RD
#include <sys/ioctl.h>
#include <unistd.h>      // close

#ifndef I2C_RDWR
#define I2C_RDWR 0x0707
#endif

namespace ddc {

// ---------------------------------------------------------------------------
// Sensor / group maps (mirrors benchmark_fpga.py)
// ---------------------------------------------------------------------------

static const std::map<std::string, std::string> kDesignatorToRail = {
    // PL_PMBUS
    {"u79", "VCCINT"},    {"u81", "VCCBRAM"},    {"u80", "VCCAUX"},
    {"u84", "VCC1V2"},    {"u16", "VCC3V3"},     {"u65", "VADJ_FMC"},
    {"u74", "MGTAVCC"},   {"u75", "MGTAVTT"},
    // PS_PMBUS
    {"u76", "VCCPSINTFP"},{"u77", "VCCPSINTLP"},{"u78", "VCCPSAUX"},
    {"u87", "VCCPSPLL"},  {"u85", "MGTRAVCC"},  {"u86", "MGTRAVTT"},
    {"u93", "VCCO_PSDDR_504"}, {"u88", "VCCOPS"}, {"u15", "VCCOPS3"},
    {"u92", "VCCPSDDRPLL"},
};

// Rails that belong to each named group (mirrors POWER_GROUPS).
// MPSoC is computed as PL + PS in results(), not listed here.
static const std::map<std::string, std::vector<std::string>> kPowerGroups = {
    {"PL",          {"VCCINT","VCCBRAM","VCCAUX","VCC1V2","VCC3V3"}},
    {"PS",          {"VCCPSINTFP","VCCPSINTLP","VCCPSAUX","VCCPSPLL",
                     "VCCO_PSDDR_504","VCCOPS","VCCOPS3","VCCPSDDRPLL"}},
    {"MGT",         {"MGTAVCC","MGTAVTT","MGTRAVCC","MGTRAVTT"}},
    {"FMC",         {"VADJ_FMC"}},
    {"DPU_fabric",  {"VCCINT","VCCBRAM"}},
    {"PS_compute",  {"VCCPSINTFP","VCCPSINTLP"}},
    {"peripherals", {"DDR4_DIMM_VDDQ","UTIL_3V3","UTIL_5V0"}},
};

static const std::map<std::string, uint8_t> kPmbusAddrMap = {
    {"DDR4_DIMM_VDDQ", 0x1D},
    {"UTIL_3V3",        0x1A},
    {"UTIL_5V0",        0x1B},
};
static const char* kPmbusBus = "/dev/i2c-4";

static constexpr uint8_t CMD_VOUT_MODE = 0x20;
static constexpr uint8_t CMD_READ_VOUT = 0x8B;
static constexpr uint8_t CMD_READ_IOUT = 0x8C;

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

PowerSampler::PowerSampler(double ina226_interval_s, double pmbus_interval_s)
    : ina226_interval_s_(ina226_interval_s)
    , pmbus_interval_s_(pmbus_interval_s)
{}

PowerSampler::~PowerSampler()
{
    if (running_.load()) {
        running_.store(false);
        if (ina226_thread_.joinable()) ina226_thread_.join();
        if (pmbus_thread_.joinable())  pmbus_thread_.join();
    }
}

// ---------------------------------------------------------------------------
// init — discover sensors and probe PMBus VOUT_MODE
// ---------------------------------------------------------------------------

bool PowerSampler::init()
{
    namespace fs = std::filesystem;

    ina226_paths_.clear();
    pmbus_rails_.clear();

    // INA226 discovery: scan /sys/class/hwmon/*/name for "ina226_uXX"
    fs::path hwmon_root("/sys/class/hwmon");
    if (fs::exists(hwmon_root)) {
        std::error_code ec;
        for (const auto& entry : fs::directory_iterator(hwmon_root, ec)) {
            auto name_path  = entry.path() / "name";
            auto power_path = entry.path() / "power1_input";
            if (!fs::exists(name_path, ec) || !fs::exists(power_path, ec))
                continue;
            std::ifstream nf(name_path);
            std::string chip_name;
            if (!nf || !(nf >> chip_name))
                continue;
            if (chip_name.rfind("ina226_", 0) != 0)
                continue;
            std::string u_ref = chip_name.substr(7);
            auto it = kDesignatorToRail.find(u_ref);
            if (it != kDesignatorToRail.end())
                ina226_paths_[it->second] = power_path.string();
        }
    }

    // PMBus: open /dev/i2c-4 and read VOUT_MODE exponent for each rail.
    // Falls back to exp=-12 (measured value on ZCU102) on read failure.
    int fd = ::open(kPmbusBus, O_RDWR);
    if (fd >= 0) {
        for (const auto& [rail, addr] : kPmbusAddrMap) {
            uint8_t raw = 0;
            if (rdwr(fd, addr, CMD_VOUT_MODE, &raw, 1)) {
                int exp = raw & 0x1F;
                if (exp > 15) exp -= 32;
                pmbus_rails_[rail] = {addr, exp};
            } else {
                pmbus_rails_[rail] = {addr, -12};
            }
        }
        ::close(fd);
    }

    initialized_ = !ina226_paths_.empty();
    return initialized_;
}

// ---------------------------------------------------------------------------
// PMBus I2C helpers
// ---------------------------------------------------------------------------

// I2C_RDWR ioctl data struct (mirrors linux/i2c-dev.h, defined here to avoid
// header conflict between linux/i2c.h and linux/i2c-dev.h).
struct LocalI2cRdwr {
    struct i2c_msg* msgs;
    uint32_t        nmsgs;
};

bool PowerSampler::rdwr(int fd, uint8_t dev_addr, uint8_t cmd, uint8_t* buf, int n)
{
    uint8_t cmd_byte = cmd;
    struct i2c_msg msgs[2];
    msgs[0].addr  = dev_addr;
    msgs[0].flags = 0;
    msgs[0].len   = 1;
    msgs[0].buf   = &cmd_byte;
    msgs[1].addr  = dev_addr;
    msgs[1].flags = I2C_M_RD;
    msgs[1].len   = static_cast<__u16>(n);
    msgs[1].buf   = buf;
    LocalI2cRdwr data = {msgs, 2};
    return ::ioctl(fd, I2C_RDWR, &data) >= 0;
}

// PMBus Linear11: 5-bit signed exponent in bits [15:11], 11-bit signed mantissa in [10:0].
double PowerSampler::linear11(uint16_t raw)
{
    int exp  = (raw >> 11) & 0x1F;
    int mant = raw & 0x7FF;
    if (exp  > 15)   exp  -= 32;
    if (mant > 1023) mant -= 2048;
    return static_cast<double>(mant) * std::exp2(static_cast<double>(exp));
}

// PMBus Linear16: raw uint16 scaled by 2^exp (exp from VOUT_MODE, e.g. -12 on ZCU102).
double PowerSampler::linear16(uint16_t raw, int exp)
{
    return static_cast<double>(raw) * std::exp2(static_cast<double>(exp));
}

double PowerSampler::pmbus_sample_rail(int fd, const std::string& rail) const
{
    auto it = pmbus_rails_.find(rail);
    if (it == pmbus_rails_.end()) return 0.0;
    const PmbusRailInfo& info = it->second;

    uint8_t buf[2] = {};
    if (!rdwr(fd, info.addr, CMD_READ_VOUT, buf, 2)) return 0.0;
    double v = linear16(static_cast<uint16_t>(buf[0] | (buf[1] << 8)), info.vout_exp);

    if (!rdwr(fd, info.addr, CMD_READ_IOUT, buf, 2)) return 0.0;
    double i = linear11(static_cast<uint16_t>(buf[0] | (buf[1] << 8)));

    double p = v * i;
    return p > 0.0 ? p : 0.0;
}

// ---------------------------------------------------------------------------
// Timing helper
// ---------------------------------------------------------------------------

static double now_steady_s()
{
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// Polling loops (run in background threads)
// ---------------------------------------------------------------------------

void PowerSampler::ina226_loop()
{
    bool first = true;
    while (running_.load()) {
        double t0 = now_steady_s();

        // Read all INA226 rails (µW → W) and stage locally before locking.
        std::vector<std::pair<std::string, double>> batch;
        batch.reserve(ina226_paths_.size());
        for (const auto& [rail, path] : ina226_paths_) {
            std::ifstream f(path);
            double uw = 0.0;
            if (f) f >> uw;
            batch.emplace_back(rail, uw / 1e6);
        }

        {
            std::lock_guard<std::mutex> lk(mtx_);
            for (const auto& [rail, w] : batch)
                samples_w_[rail].push_back(w);
            if (first) { t0_ina_ = t0; first = false; }
            t1_ina_ = t0;
        }

        double elapsed = now_steady_s() - t0;
        double rem = ina226_interval_s_ - elapsed;
        if (rem > 0.0)
            std::this_thread::sleep_for(std::chrono::duration<double>(rem));
    }
}

void PowerSampler::pmbus_loop()
{
    if (pmbus_rails_.empty()) return;

    int fd = ::open(kPmbusBus, O_RDWR);
    if (fd < 0) return;

    bool first = true;
    while (running_.load()) {
        double t0 = now_steady_s();

        std::vector<std::pair<std::string, double>> batch;
        batch.reserve(pmbus_rails_.size());
        for (const auto& kv : pmbus_rails_)
            batch.emplace_back(kv.first, pmbus_sample_rail(fd, kv.first));

        {
            std::lock_guard<std::mutex> lk(mtx_);
            for (const auto& [rail, w] : batch)
                samples_w_[rail].push_back(w);
            if (first) { t0_pmb_ = t0; first = false; }
            t1_pmb_ = t0;
        }

        double elapsed = now_steady_s() - t0;
        double rem = pmbus_interval_s_ - elapsed;
        if (rem > 0.0)
            std::this_thread::sleep_for(std::chrono::duration<double>(rem));
    }
    ::close(fd);
}

// ---------------------------------------------------------------------------
// start / stop
// ---------------------------------------------------------------------------

void PowerSampler::start()
{
    {
        std::lock_guard<std::mutex> lk(mtx_);
        samples_w_.clear();
        t0_ina_ = t1_ina_ = 0.0;
        t0_pmb_ = t1_pmb_ = 0.0;
    }
    running_.store(true);
    ina226_thread_ = std::thread(&PowerSampler::ina226_loop, this);
    pmbus_thread_  = std::thread(&PowerSampler::pmbus_loop,  this);
}

void PowerSampler::stop()
{
    running_.store(false);
    if (ina226_thread_.joinable()) ina226_thread_.join();
    if (pmbus_thread_.joinable())  pmbus_thread_.join();
}

// ---------------------------------------------------------------------------
// results
// ---------------------------------------------------------------------------

PowerResult PowerSampler::results() const
{
    PowerResult out;
    std::lock_guard<std::mutex> lk(mtx_);

    out.duration_s = (t1_ina_ > t0_ina_) ? (t1_ina_ - t0_ina_) : 0.0;

    for (const auto& [rail, sv] : samples_w_) {
        if (sv.size() < 2) continue;

        double dur = 0.0;
        if (ina226_paths_.count(rail) && t1_ina_ > t0_ina_)
            dur = t1_ina_ - t0_ina_;
        else if (pmbus_rails_.count(rail) && t1_pmb_ > t0_pmb_)
            dur = t1_pmb_ - t0_pmb_;

        double sum = 0.0;
        for (double w : sv) sum += w;
        double avg = sum / static_cast<double>(sv.size());

        RailStats rs;
        rs.avg_power_w = avg;
        rs.duration_s  = dur;
        rs.energy_j    = avg * dur;
        rs.n_samples   = static_cast<int>(sv.size());
        out.rails[rail] = rs;
        out.valid = true;
    }

    // Group aggregates: sum avg_power_w for member rails
    for (const auto& [group, members] : kPowerGroups) {
        double total = 0.0;
        bool   any   = false;
        for (const auto& rail : members) {
            auto it = out.rails.find(rail);
            if (it != out.rails.end()) {
                total += it->second.avg_power_w;
                any = true;
            }
        }
        if (any) out.groups[group] = total;
    }

    // MPSoC = PL + PS (computed separately to avoid circular group references)
    {
        double mpsoc = 0.0;
        bool   any   = false;
        for (const auto& g : std::vector<std::string>{"PL", "PS"}) {
            auto it = out.groups.find(g);
            if (it != out.groups.end()) { mpsoc += it->second; any = true; }
        }
        if (any) out.groups["MPSoC"] = mpsoc;
    }

    return out;
}

// ---------------------------------------------------------------------------
// idle_baseline — blocking convenience wrapper
// ---------------------------------------------------------------------------

PowerResult PowerSampler::idle_baseline(double duration_s)
{
    if (!initialized_) init();
    start();
    std::this_thread::sleep_for(std::chrono::duration<double>(duration_s));
    stop();
    return results();
}

} // namespace ddc
