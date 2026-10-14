#include "ewmar/backtest.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "ewmar/math.hpp"
#include "ewmar/risk.hpp"

namespace ewmar {

namespace {
double xlogy(double x, double y) { return x == 0.0 ? 0.0 : x * std::log(y); }
}  // namespace

const char* to_string(TrafficLight z) {
    switch (z) {
        case TrafficLight::Green: return "green";
        case TrafficLight::Yellow: return "yellow";
        case TrafficLight::Red: return "red";
    }
    return "?";
}

double binomial_cdf(int k, int n, double p) {
    if (k < 0) return 0.0;
    if (k >= n) return 1.0;
    return math::inc_beta(n - k, k + 1.0, 1.0 - p);  // P(X <= k) = I_{1-p}(n - k, k + 1)
}

void wilson_interval(int k, int n, double level, double& lo, double& hi) {
    if (n <= 0) {
        lo = hi = kNaN;
        return;
    }
    const double z = math::norm_ppf(0.5 + 0.5 * level), z2 = z * z, nn = n, ph = k / nn;
    const double den = 1.0 + z2 / nn;
    const double centre = (ph + z2 / (2.0 * nn)) / den;
    const double half = z * std::sqrt(ph * (1.0 - ph) / nn + z2 / (4.0 * nn * nn)) / den;
    lo = std::max(0.0, centre - half);
    hi = std::min(1.0, centre + half);
}

CoverageTest coverage_test(const std::vector<int>& hits, double p) {
    CoverageTest r;
    r.expected_rate = p;
    r.n = static_cast<int>(hits.size());
    if (r.n == 0) {
        r.rate = r.rate_lo = r.rate_hi = kNaN;
        r.p_uc = r.p_ind = r.p_cc = kNaN;
        return r;
    }
    for (int h : hits) r.exceptions += h != 0;
    const double n = r.n, x = r.exceptions;
    r.rate = x / n;
    wilson_interval(r.exceptions, r.n, 0.95, r.rate_lo, r.rate_hi);
    r.lr_uc = std::max(0.0, -2.0 * (xlogy(n - x, 1.0 - p) + xlogy(x, p) - xlogy(n - x, 1.0 - r.rate) - xlogy(x, r.rate)));
    r.p_uc = math::chi2_sf(r.lr_uc, 1.0);

    double n00 = 0, n01 = 0, n10 = 0, n11 = 0;  // transitions of the hit sequence
    for (size_t t = 1; t < hits.size(); ++t) {
        const bool a = hits[t - 1] != 0, b = hits[t] != 0;
        (a ? (b ? n11 : n10) : (b ? n01 : n00)) += 1.0;
    }
    const double p01 = n00 + n01 > 0 ? n01 / (n00 + n01) : 0.0;
    const double p11 = n10 + n11 > 0 ? n11 / (n10 + n11) : 0.0;
    const double p1 = (n01 + n11) / std::max(1.0, n00 + n01 + n10 + n11);
    const double l0 = xlogy(n00 + n10, 1.0 - p1) + xlogy(n01 + n11, p1);
    const double l1 = xlogy(n00, 1.0 - p01) + xlogy(n01, p01) + xlogy(n10, 1.0 - p11) + xlogy(n11, p11);
    r.lr_ind = std::max(0.0, -2.0 * (l0 - l1));
    r.p_ind = math::chi2_sf(r.lr_ind, 1.0);
    r.lr_cc = r.lr_uc + r.lr_ind;
    r.p_cc = math::chi2_sf(r.lr_cc, 2.0);

    r.binom_cdf = binomial_cdf(r.exceptions, r.n, p);
    r.zone = r.binom_cdf < 0.95 ? TrafficLight::Green : (r.binom_cdf < 0.9999 ? TrafficLight::Yellow : TrafficLight::Red);
    return r;
}

double forecast_cdf(const DistForecast& f, double x) {
    if (!std::isfinite(x)) return kNaN;
    if (!(f.sigma > 0.0)) return x >= 0.0 ? 1.0 : 0.0;
    return math::std_t_cdf(x / f.sigma, f.nu);
}

BerkowitzTest berkowitz_test(const std::vector<double>& pit) {
    BerkowitzTest r;
    std::vector<double> y;
    for (double u : pit)
        if (std::isfinite(u)) y.push_back(math::norm_ppf(std::clamp(u, 1e-12, 1.0 - 1e-12)));
    r.n = static_cast<int>(y.size());
    if (r.n < 3) {
        r.mu = r.rho = r.sigma = r.lr = r.p = kNaN;
        return r;
    }
    // y_t = c + rho y_{t-1} + e_t: conditional ML = OLS; H0: c = 0, rho = 0, var(e) = 1
    const size_t m = y.size() - 1;
    double mx = 0.0, my = 0.0;
    for (size_t t = 1; t < y.size(); ++t) {
        mx += y[t - 1];
        my += y[t];
    }
    mx /= static_cast<double>(m);
    my /= static_cast<double>(m);
    double sxx = 0.0, sxy = 0.0;
    for (size_t t = 1; t < y.size(); ++t) {
        sxx += (y[t - 1] - mx) * (y[t - 1] - mx);
        sxy += (y[t - 1] - mx) * (y[t] - my);
    }
    const double rho = sxx > 0.0 ? sxy / sxx : 0.0, c = my - rho * mx;
    double rss = 0.0, ss0 = 0.0;
    for (size_t t = 1; t < y.size(); ++t) {
        const double e = y[t] - c - rho * y[t - 1];
        rss += e * e;
        ss0 += y[t] * y[t];
    }
    const double md = static_cast<double>(m), s2 = std::max(rss / md, 1e-300);
    const double l1 = -0.5 * md * (math::kLn2Pi + std::log(s2) + 1.0);
    const double l0 = -0.5 * md * math::kLn2Pi - 0.5 * ss0;
    r.rho = rho;
    r.mu = std::fabs(1.0 - rho) > 1e-12 ? c / (1.0 - rho) : kNaN;
    r.sigma = std::sqrt(s2);
    r.lr = std::max(0.0, 2.0 * (l1 - l0));
    r.p = math::chi2_sf(r.lr, 3.0);
    return r;
}

EsTest acerbi_szekely_z2(const std::vector<double>& x, const std::vector<DistForecast>& f, double confidence,
                         int n_sims, std::uint64_t seed) {
    EsTest r;
    const double alpha = 1.0 - confidence;
    std::vector<size_t> idx;
    std::vector<double> q, e;
    double last_nu = kNaN;
    TailFactors tf{};
    for (size_t t = 0; t < x.size() && t < f.size(); ++t) {
        if (!std::isfinite(x[t]) || !(f[t].sigma > 0.0)) continue;
        if (!(f[t].nu == last_nu)) {  // factors only change with nu
            tf = student_factors(f[t].nu, confidence);
            last_nu = f[t].nu;
        }
        idx.push_back(t);
        q.push_back(tf.var);
        e.push_back(tf.es);
    }
    const size_t T = idx.size();
    r.n = static_cast<int>(T);
    if (T == 0) {
        r.z2 = r.p2 = kNaN;
        return r;
    }
    auto z2_of = [&](auto&& draw, int& exc) {
        double s = 0.0;
        exc = 0;
        for (size_t k = 0; k < T; ++k) {
            const double sg = f[idx[k]].sigma, xv = draw(k);
            if (xv < -sg * q[k]) {
                s += xv / (sg * e[k]);
                ++exc;
            }
        }
        return s / (static_cast<double>(T) * alpha) + 1.0;
    };
    r.z2 = z2_of([&](size_t k) { return x[idx[k]]; }, r.exceptions);
    if (n_sims <= 0) {
        r.p2 = kNaN;
        return r;
    }
    // unit-variance draws: t(nu) * sqrt((nu - 2) / nu)
    std::vector<double> scale(T);
    for (size_t k = 0; k < T; ++k) {
        const double nu = f[idx[k]].nu;
        scale[k] = std::isfinite(nu) ? f[idx[k]].sigma * std::sqrt((nu - 2.0) / nu) : f[idx[k]].sigma;
    }
    math::Rng rng(seed);
    int le = 0, dummy = 0;
    for (int s = 0; s < n_sims; ++s) {
        const double z = z2_of([&](size_t k) { return scale[k] * rng.student_t(f[idx[k]].nu); }, dummy);
        le += z <= r.z2;
    }
    r.p2 = (le + 1.0) / (n_sims + 1.0);
    return r;
}

}  // namespace ewmar
