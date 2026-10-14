// Backtest of VaR / ES forecasts:
//   * Kupiec (unconditional coverage) and Christoffersen (independence, conditional coverage);
//   * Wilson 95% interval of the exception rate and the Basel traffic light (cumulative
//     binomial probability of the exception count: green < 95%, red >= 99.99%);
//   * Berkowitz (2001): AR(1) likelihood-ratio test on Phi^-1 of the PITs (whole distribution);
//   * Acerbi-Szekely (2014) Z2 test of the ES, p-value simulated under the forecasts.
#pragma once

#include <cstdint>
#include <vector>

namespace ewmar {

enum class TrafficLight { Green, Yellow, Red };
const char* to_string(TrafficLight z);

struct CoverageTest {
    int n = 0, exceptions = 0;
    double expected_rate = 0.0, rate = 0.0, rate_lo = 0.0, rate_hi = 0.0;
    double lr_uc = 0.0, p_uc = 1.0;    // Kupiec, chi2(1)
    double lr_ind = 0.0, p_ind = 1.0;  // Christoffersen independence, chi2(1)
    double lr_cc = 0.0, p_cc = 1.0;    // conditional coverage, chi2(2)
    double binom_cdf = 0.0;
    TrafficLight zone = TrafficLight::Green;
};

// hits[t] = 1 when the loss exceeded the VaR; tail_prob = 1 - confidence.
CoverageTest coverage_test(const std::vector<int>& hits, double tail_prob);
double binomial_cdf(int k, int n, double p);  // P(X <= k), X ~ Bin(n, p)
void wilson_interval(int k, int n, double level, double& lo, double& hi);

// Zero-mean forecast X = sigma * Z, Z unit-variance t(nu) (nu = inf: Normal).
struct DistForecast {
    double sigma = 0.0, nu = 0.0;
};
double forecast_cdf(const DistForecast& f, double x);

struct BerkowitzTest {
    int n = 0;
    double mu = 0.0, rho = 0.0, sigma = 1.0, lr = 0.0, p = 1.0;
};
BerkowitzTest berkowitz_test(const std::vector<double>& pit);

struct EsTest {
    int n = 0, exceptions = 0;
    double z2 = 0.0, p2 = 1.0;
};
// x: realized P&L (losses negative). Z2 = sum(X_t I_t / ES_t) / (T alpha) + 1; Z2 < 0 means the
// losses beyond VaR were larger than the forecast ES. One-sided p = P(Z2_sim <= Z2_obs).
EsTest acerbi_szekely_z2(const std::vector<double>& x, const std::vector<DistForecast>& f, double confidence,
                         int n_sims = 2000, std::uint64_t seed = 12345);

}  // namespace ewmar
