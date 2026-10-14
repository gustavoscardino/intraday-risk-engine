// Backtesting of VaR / ES forecasts.
//
//  * Kupiec (unconditional coverage), Christoffersen (independence, conditional coverage)
//    and a Basel-style traffic light generalized to any sample size;
//  * Wilson confidence interval for the exception rate;
//  * Berkowitz (2001) test on the probability integral transforms: uses the whole forecast
//    distribution, so it is far more powerful than counting exceptions;
//  * Acerbi & Szekely (2014) Z1 / Z2 tests for Expected Shortfall, with Monte Carlo p-values
//    simulated under the forecast distributions.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "garcht/risk.hpp"

namespace garcht {

enum class TrafficLight { Green, Yellow, Red };
const char* to_string(TrafficLight z);

struct CoverageTest {
    int n = 0;           // observations
    int exceptions = 0;  // loss > VaR
    double expected_rate = 0.0;
    double rate = 0.0;
    double rate_lo = 0.0, rate_hi = 0.0;  // 95% Wilson interval for the exception rate
    double lr_uc = 0.0, p_uc = 1.0;    // Kupiec POF, chi2(1)
    double lr_ind = 0.0, p_ind = 1.0;  // Christoffersen independence, chi2(1)
    double lr_cc = 0.0, p_cc = 1.0;    // conditional coverage, chi2(2)
    double binom_cdf = 0.0;  // P(X <= exceptions) under the model, X ~ Bin(n, p)
    TrafficLight zone = TrafficLight::Green;
};

// hits[t] = 1 if the loss at t exceeded VaR, 0 otherwise. tail_prob = 1 - confidence.
CoverageTest coverage_test(const std::vector<int>& hits, double tail_prob);

// P(X <= k) for X ~ Binomial(n, p).
double binomial_cdf(int k, int n, double p);

// Wilson score interval for a binomial proportion k / n at the given two-sided level.
void wilson_interval(int k, int n, double level, double& lo, double& hi);

// One-step-ahead forecast of a P&L (or return): X = mean + sigma * Z, Z ~ `shock`
// (unit-variance Student-t, Normal, or t body with a GPD left tail).
struct DistForecast {
    double mean = 0.0;
    double sigma = 0.0;
    ShockDist shock;
    DistForecast() = default;
    DistForecast(double m, double s, double nu) : mean(m), sigma(s), shock(ShockDist::student(nu)) {}
    DistForecast(double m, double s, const ShockDist& d) : mean(m), sigma(s), shock(d) {}
};
double forecast_cdf(const DistForecast& f, double x);  // probability integral transform

struct BerkowitzTest {
    int n = 0;
    double mu = 0.0, rho = 0.0, sigma = 1.0;  // AR(1) fit of Phi^{-1}(PIT)
    double lr = 0.0, p = 1.0;                // LR test of (mu, rho, sigma) = (0, 0, 1), chi2(3)
};
// Non-finite PIT values are skipped; PITs are clamped to [1e-12, 1 - 1e-12].
BerkowitzTest berkowitz_test(const std::vector<double>& pit);

struct EsTest {
    double confidence = 0.0;
    int n = 0, exceptions = 0;
    double z1 = 0.0, p1 = 1.0;  // conditional on exceptions (NaN if none)
    double z2 = 0.0, p2 = 1.0;  // unconditional (Basel-proposed)
};
// x = realized P&L (losses negative). Z < 0 means losses beyond VaR were worse than the
// forecast ES; p-values are one-sided P(Z_sim <= Z_obs) from `n_sims` simulations.
EsTest acerbi_szekely_test(const std::vector<double>& x, const std::vector<DistForecast>& f,
                           double confidence, int n_sims = 2000, std::uint64_t seed = 12345);

}  // namespace garcht
