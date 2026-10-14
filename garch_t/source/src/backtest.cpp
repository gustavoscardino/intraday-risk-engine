#include "garcht/backtest.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "garcht/math.hpp"
#include "garcht/random.hpp"
#include "garcht/risk.hpp"

namespace garcht {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// x * log(y) with the convention 0 * log(0) = 0.
double xlogy(double x, double y) { return x == 0.0 ? 0.0 : x * std::log(y); }

}  // namespace

const char* to_string(TrafficLight z) {
    switch (z) {
        case TrafficLight::Green: return "green";
        case TrafficLight::Yellow: return "yellow";
        case TrafficLight::Red: return "red";
    }
    return "unknown";
}

double binomial_cdf(int k, int n, double p) {
    if (k < 0) return 0.0;
    if (k >= n) return 1.0;
    // P(X <= k) = I_{1-p}(n - k, k + 1)
    return math::inc_beta(n - k, k + 1.0, 1.0 - p);
}

void wilson_interval(int k, int n, double level, double& lo, double& hi) {
    if (n <= 0) {
        lo = hi = kNaN;
        return;
    }
    const double z = math::norm_ppf(0.5 + 0.5 * level);
    const double nn = n, ph = k / nn, z2 = z * z;
    const double denom = 1.0 + z2 / nn;
    const double center = (ph + z2 / (2.0 * nn)) / denom;
    const double half = z * std::sqrt(ph * (1.0 - ph) / nn + z2 / (4.0 * nn * nn)) / denom;
    lo = std::max(0.0, center - half);
    hi = std::min(1.0, center + half);
}

CoverageTest coverage_test(const std::vector<int>& hits, double p) {
    CoverageTest r;
    r.expected_rate = p;
    r.n = static_cast<int>(hits.size());
    if (r.n == 0) return r;
    for (int h : hits) r.exceptions += (h != 0);
    const double n = r.n, x = r.exceptions;
    r.rate = x / n;
    wilson_interval(r.exceptions, r.n, 0.95, r.rate_lo, r.rate_hi);

    // Kupiec proportion-of-failures test.
    r.lr_uc = -2.0 * (xlogy(n - x, 1.0 - p) + xlogy(x, p) - xlogy(n - x, 1.0 - r.rate) -
                      xlogy(x, r.rate));
    r.lr_uc = std::fmax(r.lr_uc, 0.0);
    r.p_uc = math::chi2_sf(r.lr_uc, 1.0);

    // Christoffersen independence test (first-order Markov alternative).
    double n00 = 0, n01 = 0, n10 = 0, n11 = 0;
    for (size_t t = 1; t < hits.size(); ++t) {
        const int a = hits[t - 1] != 0, b = hits[t] != 0;
        if (!a && !b) ++n00;
        else if (!a && b) ++n01;
        else if (a && !b) ++n10;
        else ++n11;
    }
    const double pi01 = (n00 + n01) > 0 ? n01 / (n00 + n01) : 0.0;
    const double pi11 = (n10 + n11) > 0 ? n11 / (n10 + n11) : 0.0;
    const double pi = (n01 + n11) / std::fmax(1.0, n00 + n01 + n10 + n11);
    const double l0 = xlogy(n00 + n10, 1.0 - pi) + xlogy(n01 + n11, pi);
    const double l1 =
        xlogy(n00, 1.0 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1.0 - pi11) + xlogy(n11, pi11);
    r.lr_ind = std::fmax(-2.0 * (l0 - l1), 0.0);
    r.p_ind = math::chi2_sf(r.lr_ind, 1.0);
    r.lr_cc = r.lr_uc + r.lr_ind;
    r.p_cc = math::chi2_sf(r.lr_cc, 2.0);

    // Traffic light: Basel thresholds on the cumulative binomial probability.
    r.binom_cdf = binomial_cdf(r.exceptions, r.n, p);
    if (r.binom_cdf < 0.95) r.zone = TrafficLight::Green;
    else if (r.binom_cdf < 0.9999) r.zone = TrafficLight::Yellow;
    else r.zone = TrafficLight::Red;
    return r;
}

double forecast_cdf(const DistForecast& f, double x) {
    if (!std::isfinite(x)) return kNaN;
    if (!(f.sigma > 0.0)) return x >= f.mean ? 1.0 : 0.0;
    return f.shock.cdf((x - f.mean) / f.sigma);
}

BerkowitzTest berkowitz_test(const std::vector<double>& pit) {
    BerkowitzTest r;
    std::vector<double> y;
    y.reserve(pit.size());
    for (double u : pit)
        if (std::isfinite(u)) y.push_back(math::norm_ppf(std::clamp(u, 1e-12, 1.0 - 1e-12)));
    r.n = static_cast<int>(y.size());
    if (r.n < 3) {
        r.lr = r.p = r.mu = r.rho = r.sigma = kNaN;
        return r;
    }
    // Conditional ML of y_t = c + rho y_{t-1} + e_t, e ~ N(0, s^2) == OLS.
    const size_t m = y.size() - 1;
    double mx = 0.0, mz = 0.0;
    for (size_t t = 1; t < y.size(); ++t) {
        mx += y[t - 1];
        mz += y[t];
    }
    mx /= static_cast<double>(m);
    mz /= static_cast<double>(m);
    double sxx = 0.0, sxz = 0.0;
    for (size_t t = 1; t < y.size(); ++t) {
        sxx += (y[t - 1] - mx) * (y[t - 1] - mx);
        sxz += (y[t - 1] - mx) * (y[t] - mz);
    }
    const double rho = sxx > 0.0 ? sxz / sxx : 0.0;
    const double c = mz - rho * mx;
    double rss = 0.0, ss0 = 0.0;
    for (size_t t = 1; t < y.size(); ++t) {
        const double e = y[t] - c - rho * y[t - 1];
        rss += e * e;
        ss0 += y[t] * y[t];
    }
    const double s2 = std::max(rss / static_cast<double>(m), 1e-300);
    const double md = static_cast<double>(m);
    const double l1 = -0.5 * md * (math::kLn2Pi + std::log(s2) + 1.0);
    const double l0 = -0.5 * md * math::kLn2Pi - 0.5 * ss0;
    r.rho = rho;
    r.mu = std::fabs(1.0 - rho) > 1e-12 ? c / (1.0 - rho) : kNaN;
    r.sigma = std::sqrt(s2);
    r.lr = std::max(0.0, 2.0 * (l1 - l0));
    r.p = math::chi2_sf(r.lr, 3.0);
    return r;
}

EsTest acerbi_szekely_test(const std::vector<double>& x, const std::vector<DistForecast>& f,
                           double confidence, int n_sims, std::uint64_t seed) {
    EsTest r;
    r.confidence = confidence;
    const double alpha = 1.0 - confidence;
    // usable observations and their (standardized) VaR / ES factors, cached per nu
    std::vector<size_t> idx;
    std::vector<double> q, e;  // per usable obs: VaR and ES factors of the unit shock
    ShockDist last;
    TailFactors tf{};
    bool have_tf = false;
    for (size_t t = 0; t < x.size() && t < f.size(); ++t) {
        if (!std::isfinite(x[t]) || !(f[t].sigma > 0.0) || !std::isfinite(f[t].mean)) continue;
        if (!have_tf || !(f[t].shock == last)) {  // factors only change when the shock does
            tf = f[t].shock.factors(confidence);
            last = f[t].shock;
            have_tf = true;
        }
        if (!std::isfinite(tf.var) || !std::isfinite(tf.es)) continue;
        idx.push_back(t);
        q.push_back(tf.var);
        e.push_back(tf.es);
    }
    const size_t T = idx.size();
    r.n = static_cast<int>(T);
    if (T == 0) {
        r.z1 = r.z2 = r.p1 = r.p2 = kNaN;
        return r;
    }

    // statistic on standardized shocks z: X = m + s z, VaR = -m + s q, ES = -m + s e
    auto stats = [&](auto&& shock, double& z1, double& z2, int& n_exc) {
        double sum = 0.0;
        n_exc = 0;
        for (size_t k = 0; k < T; ++k) {
            const DistForecast& d = f[idx[k]];
            const double xv = shock(k, d);
            const double var = -d.mean + d.sigma * q[k];
            if (xv < -var) {
                const double es = -d.mean + d.sigma * e[k];
                sum += xv / es;
                ++n_exc;
            }
        }
        z2 = sum / (static_cast<double>(T) * alpha) + 1.0;
        z1 = n_exc > 0 ? sum / n_exc + 1.0 : kNaN;
    };
    int n_exc = 0;
    stats([&](size_t k, const DistForecast&) { return x[idx[k]]; }, r.z1, r.z2, n_exc);
    r.exceptions = n_exc;
    if (n_sims <= 0) {  // statistics only: no simulated p-values
        r.p1 = r.p2 = kNaN;
        return r;
    }

    Rng rng(seed);
    int le2 = 0, le1 = 0, def1 = 0;
    for (int s = 0; s < n_sims; ++s) {
        double z1s, z2s;
        int ns;
        stats(
            [&](size_t, const DistForecast& d) { return d.mean + d.sigma * d.shock.sample(rng); },
            z1s, z2s, ns);
        le2 += z2s <= r.z2;
        if (std::isfinite(z1s)) {
            ++def1;
            le1 += std::isfinite(r.z1) && z1s <= r.z1;
        }
    }
    r.p2 = (le2 + 1.0) / (n_sims + 1.0);
    r.p1 = std::isfinite(r.z1) && def1 > 0 ? (le1 + 1.0) / (def1 + 1.0) : kNaN;
    return r;
}

}  // namespace garcht
