#include "garcht/risk.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

#include "garcht/math.hpp"
#include "garcht/optimize.hpp"
#include "garcht/random.hpp"

namespace garcht {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

bool is_normal(double nu) { return !std::isfinite(nu); }

double body_cdf(double z, double nu) { return is_normal(nu) ? math::norm_cdf(z) : math::std_t_cdf(z, nu); }
double body_ppf(double p, double nu) { return is_normal(nu) ? math::norm_ppf(p) : math::std_t_ppf(p, nu); }

// E[Z; a <= Z <= b] for the unit-variance body.
double body_partial_mean(double a, double b, double nu) {
    if (is_normal(nu)) return math::norm_pdf(a) - math::norm_pdf(b);
    const double k = std::sqrt((nu - 2.0) / nu);
    auto G = [nu](double x) { return std::isinf(x) ? 0.0 : -(nu + x * x) / (nu - 1.0) * math::t_pdf(x, nu); };
    return k * (G(b / k) - G(a / k));
}

double logistic(double x) {
    if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
    const double e = std::exp(x);
    return e / (1.0 + e);
}

double fit_unit_t_nu(const std::vector<double>& u, double nu_min, double nu_max) {
    std::vector<double> u2;
    u2.reserve(u.size());
    for (double v : u)
        if (std::isfinite(v)) u2.push_back(v * v);
    const double cnt = static_cast<double>(u2.size());
    auto negll = [&](double w) {
        const double nu = 2.0 + std::exp(w);
        const double c = math::log_gamma(0.5 * (nu + 1.0)) - math::log_gamma(0.5 * nu) -
                         0.5 * std::log(math::kPi * (nu - 2.0));
        double acc = 0.0;
        const double inv = 1.0 / (nu - 2.0);
        for (double v : u2) acc += std::log(1.0 + v * inv);
        return -(cnt * c - 0.5 * (nu + 1.0) * acc);
    };
    const double lo = std::log(std::max(nu_min - 2.0, 1e-3));
    const double hi = std::log(nu_max - 2.0);
    return 2.0 + std::exp(math::brent_minimize(negll, lo, hi, 1e-8));
}

}  // namespace

TailFactors tail_factors(Dist dist, double nu, double confidence) {
    TailFactors f;
    f.confidence = confidence;
    if (!(confidence > 0.0 && confidence < 1.0)) {
        f.var = f.es = kNaN;
        return f;
    }
    const double p = 1.0 - confidence;
    if (dist == Dist::Normal || !std::isfinite(nu)) {
        f.var = -math::norm_ppf(p);
        f.es = math::norm_pdf(f.var) / p;
        return f;
    }
    if (!(nu > 2.0)) {
        f.var = f.es = kNaN;
        return f;
    }
    const double x = math::t_ppf(p, nu);  // negative
    const double k = std::sqrt((nu - 2.0) / nu);
    f.var = -k * x;
    f.es = k * math::t_pdf(x, nu) / p * (nu + x * x) / (nu - 1.0);
    return f;
}

// ============================================================== ShockDist

ShockDist ShockDist::student(double nu) {
    ShockDist d;
    d.nu = nu;
    return d;
}

bool ShockDist::operator==(const ShockDist& o) const {
    auto same = [](double a, double b) { return a == b || (std::isinf(a) && std::isinf(b) && (a > 0) == (b > 0)); };
    return same(nu, o.nu) && gpd == o.gpd && (!gpd || (u0 == o.u0 && beta == o.beta && xi == o.xi && theta == o.theta));
}

double ShockDist::cdf(double z) const {
    if (std::isnan(z)) return kNaN;
    if (!gpd) return body_cdf(z, nu);
    if (z < -u0) {
        const double y = -z - u0;
        if (std::fabs(xi) < 1e-9) return theta * std::exp(-y / beta);
        const double base = 1.0 + xi * y / beta;
        return base > 0.0 ? theta * std::pow(base, -1.0 / xi) : 0.0;
    }
    const double f0 = body_cdf(-u0, nu);
    return theta + (1.0 - theta) * (body_cdf(z, nu) - f0) / (1.0 - f0);
}

TailFactors ShockDist::factors(double confidence) const {
    if (!gpd) return tail_factors(is_normal(nu) ? Dist::Normal : Dist::StudentT, nu, confidence);
    TailFactors f;
    f.confidence = confidence;
    const double p = 1.0 - confidence;
    if (!(p > 0.0 && p < 1.0)) {
        f.var = f.es = kNaN;
        return f;
    }
    const double tail_mean = u0 + beta / (1.0 - xi);  // E[L | L > u0]
    if (p <= theta) {
        const double r = p / theta;
        f.var = std::fabs(xi) < 1e-9 ? u0 - beta * std::log(r) : u0 + beta / xi * (std::pow(r, -xi) - 1.0);
        f.es = (f.var + beta - xi * u0) / (1.0 - xi);
        return f;
    }
    // quantile inside the body; ES mixes the GPD tail and the body slice [-u0, -VaR]
    const double f0 = body_cdf(-u0, nu);
    const double target = f0 + (p - theta) * (1.0 - f0) / (1.0 - theta);
    const double zq = body_ppf(target, nu);
    f.var = -zq;
    const double slice = (1.0 - theta) / (1.0 - f0) * body_partial_mean(-u0, zq, nu);
    f.es = (theta * tail_mean - slice) / p;
    return f;
}

double ShockDist::sample(Rng& rng) const {
    auto body = [&]() { return is_normal(nu) ? rng.normal() : rng.std_t(nu); };
    if (!gpd) return body();
    const double uu = rng.uniform();
    if (uu < theta) {
        const double r = uu / theta;
        const double y = std::fabs(xi) < 1e-9 ? -beta * std::log(r) : beta / xi * (std::pow(r, -xi) - 1.0);
        return -(u0 + y);
    }
    for (;;) {  // body conditioned on Z >= -u0 (acceptance ~ 1 - theta)
        const double z = body();
        if (z >= -u0) return z;
    }
}

ShockDist fit_student_shock(const std::vector<double>& u, double nu_min, double nu_max) {
    return ShockDist::student(fit_unit_t_nu(u, nu_min, nu_max));
}

ShockDist fit_evt_shock(const std::vector<double>& u, double nu, double tail_fraction, int min_exceedances) {
    ShockDist d = ShockDist::student(nu);
    std::vector<double> loss;
    loss.reserve(u.size());
    for (double v : u)
        if (std::isfinite(v)) loss.push_back(-v);
    const size_t N = loss.size();
    const size_t k = static_cast<size_t>(std::floor(tail_fraction * static_cast<double>(N)));
    if (k < static_cast<size_t>(std::max(min_exceedances, 10)) || k >= N / 2) return d;
    std::sort(loss.begin(), loss.end(), std::greater<double>());
    const double u0 = loss[k];  // (k+1)-th largest: the top k are the exceedances
    if (!(u0 > 0.0)) return d;
    std::vector<double> y(k);
    double m = 0.0, s2 = 0.0;
    for (size_t i = 0; i < k; ++i) {
        y[i] = loss[i] - u0;
        m += y[i];
    }
    m /= static_cast<double>(k);
    for (double v : y) s2 += (v - m) * (v - m);
    s2 /= static_cast<double>(k);
    if (!(m > 0.0)) return d;

    // GPD maximum likelihood over (log beta, xi), xi in (-0.5, 0.95)
    const double xi_lo = -0.5, xi_span = 1.45;
    auto unpack = [&](const std::vector<double>& p, double& beta, double& xi) {
        beta = std::exp(p[0]);
        xi = xi_lo + xi_span * logistic(p[1]);
    };
    auto negll = [&](double beta, double xi) {
        double acc = static_cast<double>(k) * std::log(beta);
        if (std::fabs(xi) < 1e-9) {
            for (double v : y) acc += v / beta;
            return acc;
        }
        for (double v : y) {
            const double b = 1.0 + xi * v / beta;
            if (!(b > 0.0)) return kInf;
            acc += (1.0 + 1.0 / xi) * std::log(b);
        }
        return acc;
    };
    ObjectiveFn obj = [&](const std::vector<double>& p, std::vector<double>& g) {
        double beta, xi;
        unpack(p, beta, xi);
        const double f = negll(beta, xi) / static_cast<double>(k);
        for (size_t j = 0; j < 2; ++j) {  // central differences (2 parameters, k <= ~1000)
            std::vector<double> pp = p, pm = p;
            const double h = 1e-6 * std::max(1.0, std::fabs(p[j]));
            pp[j] += h;
            pm[j] -= h;
            double bp, xp, bm, xm;
            unpack(pp, bp, xp);
            unpack(pm, bm, xm);
            g[j] = (negll(bp, xp) - negll(bm, xm)) / (2.0 * h * static_cast<double>(k));
        }
        return f;
    };
    // method-of-moments start
    double xi0 = std::clamp(0.5 * (1.0 - m * m / std::max(s2, 1e-12)), -0.3, 0.6);
    double beta0 = std::max(0.5 * m * (m * m / std::max(s2, 1e-12) + 1.0), 1e-6);
    const double s = (xi0 - xi_lo) / xi_span;
    BfgsOptions bo;
    bo.max_iter = 200;
    bo.grad_tol = 1e-7;
    const BfgsResult r = minimize_bfgs(obj, {std::log(beta0), std::log(s / (1.0 - s))}, bo);
    // no usable maximum (e.g. a non-finite gradient at the start): keep the t tail rather than
    // passing the method-of-moments start off as a fit
    if (!std::isfinite(r.f) || (!r.converged && r.iterations == 0)) return d;
    double beta, xi;
    unpack(r.x, beta, xi);
    if (!(beta > 0.0) || !(xi < 0.95)) return d;
    d.gpd = true;
    d.u0 = u0;
    d.beta = beta;
    d.xi = xi;
    d.theta = static_cast<double>(k) / static_cast<double>(N);
    return d;
}

// ============================================================== TailTable

void TailTable::set(Dist dist, double nu, const std::vector<double>& confidences) {
    set(ShockDist::student(dist == Dist::Normal ? kInf : nu), confidences);
}

void TailTable::set(const ShockDist& d, const std::vector<double>& confidences) {
    dist_ = d;
    f_.clear();
    f_.reserve(confidences.size());
    for (double c : confidences) f_.push_back(d.factors(c));
}

}  // namespace garcht
