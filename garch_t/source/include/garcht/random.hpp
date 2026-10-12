// Deterministic random numbers (xoshiro256** + own transforms).
// std::normal_distribution & co. are implementation-defined, so they would produce a
// different synthetic dataset on each standard library; these generators do not
// (up to last-bit differences of std::log/std::sqrt between math libraries).
#pragma once

#include <cmath>
#include <cstdint>

namespace garcht {

class Rng {
public:
    explicit Rng(std::uint64_t seed) {
        for (auto& s : s_) s = splitmix64(seed);
    }

    std::uint64_t next_u64() {
        const std::uint64_t result = rotl(s_[1] * 5, 7) * 9;
        const std::uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 45);
        return result;
    }

    // Uniform in (0, 1): never returns exactly 0 or 1. 52 random bits plus a half-step:
    // the largest value is 1 - 2^-53, which is representable (with 53 bits plus 0.5 the
    // largest value would round to 1.0).
    double uniform() { return (static_cast<double>(next_u64() >> 12) + 0.5) * 0x1.0p-52; }

    // Standard Normal (Marsaglia polar method).
    double normal() {
        if (has_spare_) {
            has_spare_ = false;
            return spare_;
        }
        double u, v, s;
        do {
            u = 2.0 * uniform() - 1.0;
            v = 2.0 * uniform() - 1.0;
            s = u * u + v * v;
        } while (s >= 1.0 || s == 0.0);
        const double m = std::sqrt(-2.0 * std::log(s) / s);
        spare_ = v * m;
        has_spare_ = true;
        return u * m;
    }

    // Gamma(shape, 1) (Marsaglia & Tsang, 2000).
    double gamma(double shape) {
        if (shape < 1.0) return gamma(shape + 1.0) * std::pow(uniform(), 1.0 / shape);
        const double d = shape - 1.0 / 3.0, c = 1.0 / std::sqrt(9.0 * d);
        for (;;) {
            double x, v;
            do {
                x = normal();
                v = 1.0 + c * x;
            } while (v <= 0.0);
            v = v * v * v;
            const double u = uniform();
            if (u < 1.0 - 0.0331 * x * x * x * x) return d * v;
            if (std::log(u) < 0.5 * x * x + d * (1.0 - v + std::log(v))) return d * v;
        }
    }

    // Unit-variance Student-t with nu > 2 degrees of freedom (+inf => Normal).
    double std_t(double nu) {
        if (!std::isfinite(nu)) return normal();
        const double chi2 = 2.0 * gamma(0.5 * nu);
        return normal() / std::sqrt(chi2 / nu) * std::sqrt((nu - 2.0) / nu);
    }

private:
    static std::uint64_t splitmix64(std::uint64_t& x) {
        std::uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    static std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

    std::uint64_t s_[4];
    double spare_ = 0.0;
    bool has_spare_ = false;
};

}  // namespace garcht
