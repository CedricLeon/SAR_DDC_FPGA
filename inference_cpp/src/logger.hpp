#pragma once
// Minimal logger — writes to stderr with timestamps.
// No external dependency (no spdlog: confirmed absent on ZCU102).

#include <chrono>
#include <cstdio>
#include <ctime>
#include <string>

namespace ddc {

class Logger {
public:
    static Logger& instance() {
        static Logger inst;
        return inst;
    }

    void set_verbose(bool v) { verbose_ = v; }
    bool is_verbose() const { return verbose_; }

    void info(const std::string& msg) const {
        print_prefix("[INFO ]");
        fputs(msg.c_str(), stderr);
        fputc('\n', stderr);
        if (log_file_) {
            fputs(msg.c_str(), log_file_);
            fputc('\n', log_file_);
        }
    }

    void verbose(const std::string& msg) const {
        if (!verbose_) return;
        print_prefix("[DEBUG]");
        fputs(msg.c_str(), stderr);
        fputc('\n', stderr);
    }

    void error(const std::string& msg) const {
        print_prefix("[ERROR]");
        fputs(msg.c_str(), stderr);
        fputc('\n', stderr);
    }

    bool open_log_file(const std::string& path) {
        if (log_file_) fclose(log_file_);
        log_file_ = fopen(path.c_str(), "a");
        return log_file_ != nullptr;
    }

    ~Logger() {
        if (log_file_) fclose(log_file_);
    }

private:
    Logger() = default;
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    bool   verbose_  = false;
    FILE*  log_file_ = nullptr;

    static void print_prefix(const char* level) {
        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);
        char buf[32];
        std::strftime(buf, sizeof(buf), "%H:%M:%S", std::localtime(&t));
        fprintf(stderr, "%s %s ", buf, level);
    }
};

// Convenience macros — callers always pass a pre-composed std::string or string literal.
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LOG_INFO(msg)    ddc::Logger::instance().info(std::string(msg))
#define LOG_VERBOSE(msg) ddc::Logger::instance().verbose(std::string(msg))
#define LOG_ERROR(msg)   ddc::Logger::instance().error(std::string(msg))

} // namespace ddc
