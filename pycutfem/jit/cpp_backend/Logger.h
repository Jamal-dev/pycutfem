#pragma once
#include <iostream>
#include <cstdlib>

// A singleton to check env var once
struct Logger {
    static bool isEnabled() {
        static bool enabled = (std::getenv("ENABLE_LOGS") != nullptr);
        return enabled;
    }
};

// The Macro: It creates a "if" statement. 
// If logs are off, the code after '<<' is never executed (saving CPU).
#define LOG_INFO(msg) \
    if (Logger::isEnabled()) { \
        std::cout << "[INFO] " << msg << std::endl; \
    }
