// Strict command-line value parsing shared by the apps: a value must be a whole number / one
// of the listed choices (case-insensitive), otherwise the option is rejected with a message
// instead of silently falling back to a default.
#pragma once

#include <cctype>
#include <cerrno>
#include <climits>
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

inline bool to_int(const std::string& opt, const std::string& s, int& out, int lo = INT_MIN, int hi = INT_MAX) {
    char* end = nullptr;
    errno = 0;
    const long v = std::strtol(s.c_str(), &end, 10);
    if (s.empty() || *end != '\0' || errno != 0 || v < lo || v > hi) {
        std::fprintf(stderr, "error: %s expects an integer in [%d, %d], got '%s'\n", opt.c_str(), lo, hi, s.c_str());
        return false;
    }
    out = static_cast<int>(v);
    return true;
}

inline bool to_double(const std::string& opt, const std::string& s, double& out, double lo = -HUGE_VAL,
                      double hi = HUGE_VAL) {
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (s.empty() || *end != '\0' || !std::isfinite(v) || v < lo || v > hi) {
        std::fprintf(stderr, "error: %s expects a number in [%g, %g], got '%s'\n", opt.c_str(), lo, hi, s.c_str());
        return false;
    }
    out = v;
    return true;
}

// index of `s` in `choices` (case-insensitive), or -1 with an error message
inline int choice(const std::string& opt, const std::string& s, std::initializer_list<const char*> choices) {
    const std::string v = lower(s);
    int k = 0;
    for (const char* c : choices) {
        if (v == c) return k;
        ++k;
    }
    std::string list;
    for (const char* c : choices) list += (list.empty() ? "" : "|") + std::string(c);
    std::fprintf(stderr, "error: %s expects %s, got '%s'\n", opt.c_str(), list.c_str(), s.c_str());
    return -1;
}

}  // namespace cli
