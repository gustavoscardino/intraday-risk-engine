// Command-line helpers of the programs: values are validated and rejected with a message,
// never silently replaced by a default.
#pragma once

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <string>

namespace cli {

inline std::string lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

inline bool int_arg(const std::string& opt, const std::string& v, int& out, int lo, int hi) {
    char* end = nullptr;
    const long x = std::strtol(v.c_str(), &end, 10);
    if (v.empty() || *end != '\0' || x < lo || x > hi) {
        std::fprintf(stderr, "error: %s expects a whole number in [%d, %d], got '%s'\n", opt.c_str(), lo, hi, v.c_str());
        return false;
    }
    out = static_cast<int>(x);
    return true;
}

inline bool double_arg(const std::string& opt, const std::string& v, double& out, double lo, double hi) {
    char* end = nullptr;
    const double x = std::strtod(v.c_str(), &end);
    if (v.empty() || *end != '\0' || !std::isfinite(x) || x < lo || x > hi) {
        std::fprintf(stderr, "error: %s expects a number in [%g, %g], got '%s'\n", opt.c_str(), lo, hi, v.c_str());
        return false;
    }
    out = x;
    return true;
}

// position of v among the choices (case-insensitive), or -1 with a message
inline int choice_arg(const std::string& opt, const std::string& v, std::initializer_list<const char*> choices) {
    int k = 0;
    std::string all;
    for (const char* c : choices) {
        if (lower(v) == c) return k;
        all += (k++ ? "|" : "") + std::string(c);
    }
    std::fprintf(stderr, "error: %s expects %s, got '%s'\n", opt.c_str(), all.c_str(), v.c_str());
    return -1;
}

}  // namespace cli
