// Unit and integration tests for the GARCH-t risk library.
// Reference values come from SciPy 1.18 and the Python `arch` 8.0 package
// (see tests/reference/crosscheck_arch.py).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "garcht/garcht.hpp"
#include "garcht/random.hpp"

using namespace garcht;

// ------------------------------------------------------------------ mini framework
namespace {
struct TestCase {
    const char* name;
    void (*fn)();
};
std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}
struct Registrar {
    Registrar(const char* n, void (*f)()) { registry().push_back({n, f}); }
};
int g_checks = 0, g_failures = 0, g_test_failures = 0;

bool near(double a, double b, double abs_tol, double rel_tol) {
    if (std::isnan(a) || std::isnan(b)) return false;
    if (a == b) return true;
    return std::fabs(a - b) <= abs_tol + rel_tol * std::fabs(b);
}
}  // namespace

#define TEST(name)                                     \
    static void name();                                \
    static const Registrar registrar_##name(#name, name); \
    static void name()

#define CHECK(cond)                                                             \
    do {                                                                        \
        ++g_checks;                                                             \
        if (!(cond)) {                                                          \
            ++g_failures;                                                       \
            std::printf("    FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
        }                                                                       \
    } while (0)

#define CHECK_NEAR(a, b, abs_tol, rel_tol)                                                     \
    do {                                                                                       \
        ++g_checks;                                                                            \
        const double va_ = (a), vb_ = (b);                                                     \
        if (!near(va_, vb_, (abs_tol), (rel_tol))) {                                           \
            ++g_failures;                                                                      \
            std::printf("    FAIL %s:%d: %s = %.17g, expected %s = %.17g\n", __FILE__, __LINE__, \
                        #a, va_, #b, vb_);                                                     \
        }                                                                                      \
    } while (0)

#define CHECK_REL(a, b, rel) CHECK_NEAR(a, b, 0.0, rel)

// ------------------------------------------------------------------ helpers
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

std::vector<double> simulate_garch(const GarchParams& p, size_t n, std::uint64_t seed,
                                   size_t burn = 500) {
    Rng rng(seed);
    std::vector<double> y(n);
    double h = p.unconditional_variance();
    for (size_t t = 0; t < n + burn; ++t) {
        const double e = std::sqrt(h) * rng.std_t(p.nu);
        if (t >= burn) y[t - burn] = p.mu + e;
        h = p.omega + p.alpha * e * e + p.beta * h;
    }
    return y;
}

std::vector<double> deterministic_series() {  // same formula as crosscheck_arch.py
    std::vector<double> y(500);
    for (size_t t = 0; t < y.size(); ++t) {
        const double td = static_cast<double>(t);
        const double s = std::sin(1.7 * td);
        y[t] = 0.8 * std::sin(0.37 * td + 0.2) * (1.0 + 0.6 * std::cos(0.051 * td)) + 0.3 * s * s * s;
    }
    return y;
}

bool is_psd_corr(const std::vector<double>& R, int n) {
    std::vector<double> L(static_cast<size_t>(n * n), 0.0);
    for (int j = 0; j < n; ++j) {
        if (std::fabs(R[static_cast<size_t>(j * n + j)] - 1.0) > 1e-12) return false;
        double d = R[static_cast<size_t>(j * n + j)];
        for (int k = 0; k < j; ++k) d -= L[static_cast<size_t>(j * n + k)] * L[static_cast<size_t>(j * n + k)];
        if (d < -1e-12) return false;
        L[static_cast<size_t>(j * n + j)] = std::sqrt(std::max(d, 0.0));
        for (int i = j + 1; i < n; ++i) {
            double s = R[static_cast<size_t>(i * n + j)];
            if (std::fabs(s - R[static_cast<size_t>(j * n + i)]) > 1e-12) return false;
            for (int k = 0; k < j; ++k) s -= L[static_cast<size_t>(i * n + k)] * L[static_cast<size_t>(j * n + k)];
            L[static_cast<size_t>(i * n + j)] = L[static_cast<size_t>(j * n + j)] > 1e-14 ? s / L[static_cast<size_t>(j * n + j)] : 0.0;
        }
    }
    return true;
}

}  // namespace

// ================================================================== math

TEST(normal_quantile_matches_scipy) {
    CHECK_REL(math::norm_ppf(0.975), 1.959963984540054, 1e-15);
    CHECK_REL(math::norm_ppf(0.99), 2.3263478740408408, 1e-15);
    CHECK_REL(math::norm_ppf(0.999), 3.090232306167813, 1e-15);
    CHECK_REL(math::norm_ppf(1e-10), -6.361340902404056, 1e-15);
    CHECK_REL(math::norm_ppf(0.3), -0.5244005127080409, 1e-15);
    CHECK_REL(math::norm_ppf(0.05), -1.6448536269514729, 1e-15);
    CHECK_REL(math::norm_ppf(1e-300), -37.0470962993612, 1e-13);
    CHECK(math::norm_ppf(0.5) == 0.0);
    CHECK(std::isinf(math::norm_ppf(0.0)) && math::norm_ppf(0.0) < 0);
    CHECK(std::isnan(math::norm_ppf(1.5)));
    for (double p = 1e-12; p < 1.0; p = p < 0.01 ? p * 3.7 : p + 0.0137)
        CHECK_NEAR(math::norm_cdf(math::norm_ppf(p)), p, 0.0, 2e-14);
}

TEST(student_t_quantile_matches_scipy) {
    CHECK_REL(math::t_ppf(0.975, 5), 2.5705818356363146, 1e-13);
    CHECK_REL(math::t_ppf(0.99, 10), 2.7637694581126957, 1e-13);
    CHECK_REL(math::t_ppf(0.975, 30), 2.0422724563012378, 1e-13);
    CHECK_REL(math::t_ppf(0.995, 3), 5.840909309733355, 1e-13);
    CHECK_REL(math::t_ppf(0.01, 4), -3.746947387979197, 1e-13);
    CHECK_REL(math::t_ppf(0.001, 2.5), -13.822193110865964, 1e-13);
    CHECK_REL(math::t_ppf(1e-8, 3), -479.5250695797382, 1e-12);
    CHECK_REL(math::t_ppf(0.05, 100), -1.6602343260853396, 1e-13);
    CHECK_REL(math::t_ppf(0.01, 1000), -2.330082674755513, 1e-13);
    CHECK_REL(math::t_ppf(0.025, 2.1), -4.112152758173881, 1e-13);
    CHECK_REL(math::t_ppf(0.01, 7.3), -2.9639404475838975, 1e-13);
    CHECK_REL(math::t_ppf(1e-6, 1.0), -318309.8861827435, 1e-12);
    CHECK_REL(math::t_ppf(0.3, 0.5), -1.0095258786071655, 1e-12);
    // nu -> infinity converges to the Normal quantile
    CHECK_REL(math::t_ppf(0.99, 1e9), math::norm_ppf(0.99), 1e-8);
}

TEST(student_t_quantile_closed_forms) {
    for (double p : {1e-9, 1e-5, 0.001, 0.01, 0.05, 0.2, 0.45, 0.6, 0.9, 0.999}) {
        // Cauchy: tan(pi (p - 1/2)) = -cot(pi p), written to avoid the pole's cancellation
        CHECK_REL(math::t_ppf(p, 1.0), p < 0.5 ? -1 / std::tan(math::kPi * p) : 1 / std::tan(math::kPi * (1 - p)), 1e-12);
        CHECK_REL(math::t_ppf(p, 2.0), (2 * p - 1) / std::sqrt(2 * p * (1 - p)), 1e-12);
        const double a = 4 * p * (1 - p);
        const double q = std::cos(std::acos(std::sqrt(a)) / 3) / std::sqrt(a);
        CHECK_REL(math::t_ppf(p, 4.0), (p < 0.5 ? -2.0 : 2.0) * std::sqrt(q - 1), 1e-11);
    }
    // round trip over a grid of tails and degrees of freedom
    for (double nu : {2.05, 2.5, 3.0, 4.5, 7.0, 12.0, 30.0, 120.0, 500.0})
        for (double p : {1e-10, 1e-6, 1e-4, 0.001, 0.01, 0.025, 0.05, 0.1, 0.3, 0.49})
            CHECK_NEAR(math::t_cdf(math::t_ppf(p, nu), nu), p, 0.0, 1e-11);
}

TEST(student_t_cdf_matches_scipy) {
    CHECK_REL(math::t_cdf(-3.0, 4), 0.019970984035859413, 1e-13);
    CHECK_REL(math::t_cdf(2.5, 6.5), 0.9782352425757278, 1e-13);
    CHECK_REL(math::t_cdf(-50, 3), 8.808576020635984e-06, 1e-12);
    CHECK_REL(math::t_cdf(0.1, 10), 0.5388396407177959, 1e-13);
    CHECK_REL(math::t_cdf(-1000.0, 2.2), 1.4402168061261954e-07, 1e-11);
}

TEST(gamma_functions_match_scipy) {
    const double x[] = {0.5, 1.0, 2.5, 7.3, 15.5, 100.25, 250.0, 1e-3};
    const double lg[] = {0.5723649429247,   0.0,                0.2846828704729192, 7.147892523022249,
                         26.536914491115613, 360.28455963776423, 1128.5237708729908, 6.907178885383853};
    const double dg[] = {-1.9635100260214235, -0.5772156649015329, 0.7031566406452432,
                         1.917820335637986,   2.7082352425903653,  4.602671243274712,
                         5.519459584531046,   -1000.5755719318103};
    for (int i = 0; i < 8; ++i) {
        CHECK_NEAR(math::log_gamma(x[i]), lg[i], 2e-14, 1e-14);
        CHECK_NEAR(math::digamma(x[i]), dg[i], 1e-14, 1e-14);
    }
}

TEST(incomplete_beta_chi2_binomial_match_scipy) {
    CHECK_REL(math::inc_beta(0.5, 5.0, 0.3), 0.9347377538310915, 1e-13);
    CHECK_REL(math::inc_beta(2.5, 0.5, 0.9), 0.48958974456442755, 1e-13);
    CHECK_REL(math::inc_beta(50, 0.5, 0.99), 0.3173043978741973, 1e-12);
    CHECK_REL(math::inc_beta(3, 4, 0.5), 0.65625, 1e-14);
    CHECK_REL(math::chi2_sf(3.84, 1), 0.05004352124870519, 1e-12);
    CHECK_REL(math::chi2_sf(5.99, 2), 0.05003662708658629, 1e-12);
    CHECK_REL(math::chi2_sf(10.0, 3), 0.01856613546304325, 1e-12);
    CHECK_REL(math::chi2_sf(0.5, 1), 0.47950012218695337, 1e-12);
    CHECK_REL(math::chi2_sf(25, 2), 3.7266531720786718e-06, 1e-12);
    CHECK_REL(binomial_cdf(4, 250, 0.01), 0.8921876269036251, 1e-12);
    CHECK_REL(binomial_cdf(5, 250, 0.01), 0.9588168159301517, 1e-12);
    CHECK_REL(binomial_cdf(9, 250, 0.01), 0.9997498099312595, 1e-12);
    CHECK_REL(binomial_cdf(10, 250, 0.01), 0.999946101370953, 1e-12);
    CHECK_REL(binomial_cdf(30, 2500, 0.01), 0.8644505420558679, 1e-11);
}

TEST(standardized_t_density_has_unit_variance) {
    for (double nu : {5.0, 8.0, 30.0}) {
        // Simpson on [-L, L]; tails beyond L are negligible for these nu.
        const double L = 400.0;
        const int N = 400000;
        const double h = 2 * L / N;
        double mass = 0.0, var = 0.0;
        for (int i = 0; i <= N; ++i) {
            const double z = -L + i * h;
            const double w = (i == 0 || i == N) ? 1.0 : (i % 2 ? 4.0 : 2.0);
            const double f = math::std_t_pdf(z, nu);
            mass += w * f;
            var += w * z * z * f;
        }
        CHECK_NEAR(mass * h / 3, 1.0, 1e-9, 0.0);
        CHECK_NEAR(var * h / 3, 1.0, 2e-6, 0.0);
        CHECK_NEAR(math::std_t_cdf(math::std_t_ppf(0.01, nu), nu), 0.01, 0.0, 1e-12);
    }
}

TEST(var_es_tail_factors_match_scipy) {
    struct Case { double c, nu, var, es; };
    const Case cases[] = {{0.99, 5, 2.606463569384279, 3.448836760048015},
                          {0.975, 4, 1.9632431614775572, 2.8238712518154174},
                          {0.95, 10, 1.6211145108730005, 2.1541393786578658},
                          {0.99, 2.5, 2.3939840948021334, 4.0657775735052715}};
    for (const Case& k : cases) {
        const TailFactors f = tail_factors(Dist::StudentT, k.nu, k.c);
        CHECK_REL(f.var, k.var, 1e-12);
        CHECK_REL(f.es, k.es, 1e-12);
    }
    CHECK_REL(tail_factors(Dist::Normal, kInf, 0.95).es, 2.0627128075074257, 1e-13);
    CHECK_REL(tail_factors(Dist::Normal, kInf, 0.975).es, 2.3378027922014133, 1e-13);
    CHECK_REL(tail_factors(Dist::Normal, kInf, 0.99).es, 2.665214220345806, 1e-13);
    CHECK_REL(tail_factors(Dist::StudentT, 1e8, 0.99).var, math::norm_ppf(0.99), 1e-6);
    // ES > VaR, and the t tail is heavier than the Normal at high confidence
    const TailFactors t5 = tail_factors(Dist::StudentT, 5, 0.99);
    CHECK(t5.es > t5.var);
    CHECK(t5.var > tail_factors(Dist::Normal, kInf, 0.99).var);
    CHECK(std::isnan(tail_factors(Dist::StudentT, 1.5, 0.99).var));
    const RiskMeasure m = risk_measure(0.001, 0.02, t5);
    CHECK_REL(m.var, -0.001 + 0.02 * t5.var, 1e-15);
}

TEST(brent_and_bfgs_minimize_test_functions) {
    const double xm = math::brent_minimize([](double x) { return (x - 1.234) * (x - 1.234) + 3; }, -10, 10);
    CHECK_NEAR(xm, 1.234, 1e-7, 0.0);  // a minimizer is only determined to ~sqrt(eps)
    ObjectiveFn rosen = [](const std::vector<double>& x, std::vector<double>& g) {
        const double a = 1 - x[0], b = x[1] - x[0] * x[0];
        g[0] = -2 * a - 400 * x[0] * b;
        g[1] = 200 * b;
        return a * a + 100 * b * b;
    };
    BfgsOptions o;
    o.max_iter = 1000;
    const BfgsResult r = minimize_bfgs(rosen, {-1.2, 1.0}, o);
    CHECK(r.converged);
    CHECK_NEAR(r.x[0], 1.0, 1e-6, 0.0);
    CHECK_NEAR(r.x[1], 1.0, 1e-6, 0.0);
    // objective that is +inf outside its domain (x > 0)
    ObjectiveFn barrier = [](const std::vector<double>& x, std::vector<double>& g) {
        if (x[0] <= 0) return kInf;
        g[0] = 1 - 1 / x[0];
        return x[0] - std::log(x[0]);
    };
    const BfgsResult rb = minimize_bfgs(barrier, {20.0}, o);
    CHECK_NEAR(rb.x[0], 1.0, 1e-6, 0.0);
}

// ================================================================== likelihood

TEST(loglik_matches_arch_package) {
    const std::vector<double> y = deterministic_series();
    struct Case { Dist d; double mu; double nu; double ll; double h_last; };
    const Case cases[] = {{Dist::StudentT, 0.0, 6.0, -490.5887597006622, 1.024763348608011},
                          {Dist::Normal, 0.0, kInf, -496.3600964261322, 1.024763348608011},
                          {Dist::StudentT, 0.05, 6.0, -491.5007802956677, 1.002049287138952},
                          {Dist::Normal, 0.05, kInf, -497.18669109836156, 1.002049287138952}};
    for (const Case& c : cases) {
        const GarchParams p{c.mu, 0.05, 0.08, 0.88, c.nu};
        // arch builds the backcast from the residuals at its starting values (y - mean(y))
        double ybar = 0.0;
        for (double v : y) ybar += v / static_cast<double>(y.size());
        std::vector<double> r(y.size()), h(y.size());
        for (size_t t = 0; t < y.size(); ++t) r[t] = y[t] - (c.mu != 0.0 ? ybar : 0.0);
        const double bc = garch_backcast(r.data(), r.size());
        CHECK_REL(garch_loglik(y.data(), y.size(), c.d, p, bc, nullptr, h.data()), c.ll, 1e-12);
        CHECK_REL(h.back(), c.h_last, 1e-12);
    }
}

TEST(analytic_gradient_matches_finite_differences) {
    std::vector<double> y = simulate_garch({0.03, 0.05, 0.09, 0.88, 5.0}, 3000, 7);
    for (size_t t = 100; t < y.size(); t += 97) y[t] = kNaN;  // exercise missing-data path
    const double bc = garch_backcast(y.data(), y.size());
    for (Dist d : {Dist::StudentT, Dist::Normal}) {
        const GarchParams p{0.02, 0.06, 0.11, 0.84, d == Dist::StudentT ? 6.5 : kInf};
        double g[5];
        garch_loglik(y.data(), y.size(), d, p, bc, g);
        for (int k = 0; k < (d == Dist::StudentT ? 5 : 4); ++k) {
            GarchParams pp = p, pm = p;
            double* fp[5] = {&pp.mu, &pp.omega, &pp.alpha, &pp.beta, &pp.nu};
            double* fm[5] = {&pm.mu, &pm.omega, &pm.alpha, &pm.beta, &pm.nu};
            const double step = 1e-6 * std::max(1.0, std::fabs(*fp[k]));
            *fp[k] += step;
            *fm[k] -= step;
            const double fd = (garch_loglik(y.data(), y.size(), d, pp, bc) -
                               garch_loglik(y.data(), y.size(), d, pm, bc)) / (2 * step);
            CHECK_NEAR(g[k], fd, 1e-4, 1e-6);
        }
    }
}

TEST(missing_observations_use_conditional_expectation) {
    std::vector<double> y = deterministic_series();
    y[10] = kNaN;
    y[11] = kNaN;
    const GarchParams p{0.0, 0.05, 0.08, 0.88, 6.0};
    const double bc = garch_backcast(y.data(), y.size());
    std::vector<double> h(y.size());
    const double ll = garch_loglik(y.data(), y.size(), Dist::StudentT, p, bc, nullptr, h.data());
    // manual recursion
    double hh = p.omega + (p.alpha + p.beta) * bc, acc = 0.0;
    for (size_t t = 0; t < y.size(); ++t) {
        CHECK_REL(h[t], hh, 1e-14);
        double e2 = hh;
        if (std::isfinite(y[t])) {
            acc += math::std_t_log_pdf(y[t] / std::sqrt(hh), p.nu) - 0.5 * std::log(hh);
            e2 = y[t] * y[t];
        }
        hh = p.omega + p.alpha * e2 + p.beta * hh;
    }
    CHECK_REL(ll, acc, 1e-12);
    CHECK(std::isinf(garch_loglik(y.data(), y.size(), Dist::StudentT, {0, -1, 0.1, 0.8, 6}, bc)));
}

// ================================================================== estimation

TEST(fit_recovers_garch_t_parameters) {
    const GarchParams truth{0.0, 0.02, 0.08, 0.90, 6.0};
    for (std::uint64_t seed : {1u, 2u, 3u}) {
        const std::vector<double> y = simulate_garch(truth, 20000, seed);
        const FitResult f = fit_garch(y, GarchSpec{});
        CHECK(f.status == FitStatus::Ok);
        CHECK_NEAR(f.params.alpha, truth.alpha, 0.015, 0.0);
        CHECK_NEAR(f.params.beta, truth.beta, 0.02, 0.0);
        CHECK_NEAR(f.params.nu, truth.nu, 1.0, 0.0);
        CHECK_NEAR(f.params.unconditional_variance(), truth.unconditional_variance(), 0.0, 0.15);
        // truth inside +-4 standard errors (classic and robust)
        CHECK(std::fabs(f.params.alpha - truth.alpha) < 4 * f.std_err.alpha);
        CHECK(std::fabs(f.params.beta - truth.beta) < 4 * f.std_err.beta);
        CHECK(std::fabs(f.params.nu - truth.nu) < 4 * f.robust_std_err.nu);
        CHECK(f.std_err.alpha > 0 && f.robust_std_err.alpha > 0 && std::isnan(f.std_err.mu));
        CHECK(f.n_obs == 20000 && f.n_params == 4);
    }
}

TEST(fit_recovers_normal_and_constant_mean) {
    const GarchParams tn{0.0, 0.01, 0.05, 0.93, kInf};
    const FitResult fn = fit_garch(simulate_garch(tn, 20000, 11), GarchSpec{Dist::Normal});
    CHECK(fn.status == FitStatus::Ok);
    CHECK_NEAR(fn.params.alpha, 0.05, 0.012, 0.0);
    CHECK_NEAR(fn.params.beta, 0.93, 0.015, 0.0);
    CHECK(std::isinf(fn.params.nu) && fn.n_params == 3);

    const GarchParams tm{0.05, 0.02, 0.10, 0.85, 8.0};
    const FitResult fm =
        fit_garch(simulate_garch(tm, 20000, 12), GarchSpec{Dist::StudentT, MeanModel::Constant});
    CHECK(fm.status == FitStatus::Ok);
    CHECK_NEAR(fm.params.mu, 0.05, 4 * fm.std_err.mu, 0.0);
    CHECK_NEAR(fm.params.alpha, 0.10, 0.02, 0.0);
    CHECK_NEAR(fm.params.beta, 0.85, 0.03, 0.0);
    CHECK_NEAR(fm.params.nu, 8.0, 2.0, 0.0);
}

TEST(fit_is_scale_invariant) {
    const std::vector<double> y = simulate_garch({0.0, 0.02, 0.07, 0.91, 5.0}, 5000, 21);
    std::vector<double> ys(y.size());
    const double c = 1e-4;  // e.g. percent returns -> fractions of basis points
    for (size_t i = 0; i < y.size(); ++i) ys[i] = c * y[i];
    const FitResult a = fit_garch(y, GarchSpec{}), b = fit_garch(ys, GarchSpec{});
    CHECK(a.status == FitStatus::Ok && b.status == FitStatus::Ok);
    CHECK_NEAR(a.params.alpha, b.params.alpha, 1e-5, 0.0);
    CHECK_NEAR(a.params.beta, b.params.beta, 1e-5, 0.0);
    CHECK_NEAR(a.params.nu, b.params.nu, 1e-3, 0.0);
    CHECK_REL(b.params.omega, a.params.omega * c * c, 1e-4);
    CHECK_REL(b.loglik, a.loglik - 5000 * std::log(c), 1e-9);  // density of c*y = f(y)/c
    CHECK_REL(b.next_variance, a.next_variance * c * c, 1e-4);
}

TEST(fit_gradient_vanishes_at_optimum_and_warm_start_agrees) {
    const std::vector<double> y = simulate_garch({0.0, 0.03, 0.06, 0.92, 7.0}, 8000, 31);
    const FitResult f = fit_garch(y, GarchSpec{});
    CHECK(f.status == FitStatus::Ok);
    double g[5];
    garch_loglik(y.data(), y.size(), Dist::StudentT, f.params, f.backcast, g);
    // elasticities of the *mean* log-likelihood ~ 0 (interior optimum)
    const double n = static_cast<double>(y.size());
    CHECK_NEAR(g[1] * f.params.omega / n, 0.0, 5e-6, 0.0);
    CHECK_NEAR(g[2] * f.params.alpha / n, 0.0, 5e-6, 0.0);
    CHECK_NEAR(g[3] * f.params.beta / n, 0.0, 5e-6, 0.0);
    CHECK_NEAR(g[4] * f.params.nu / n, 0.0, 5e-6, 0.0);
    FitOptions o;
    GarchParams w{0.0, 0.05, 0.15, 0.70, 15.0};
    o.warm_start = &w;
    const FitResult f2 = fit_garch(y, GarchSpec{}, o);
    CHECK_NEAR(f2.loglik, f.loglik, 1e-6, 0.0);
    CHECK_NEAR(f2.params.beta, f.params.beta, 1e-4, 0.0);
}

TEST(fit_is_a_local_maximum) {
    const std::vector<double> y = simulate_garch({0.0, 0.05, 0.10, 0.85, 5.0}, 4000, 99);
    const FitResult f = fit_garch(y, GarchSpec{});
    CHECK(f.status == FitStatus::Ok);
    for (double da : {-0.005, 0.005})
        for (double db : {-0.005, 0.005}) {
            GarchParams p = f.params;
            p.alpha += da;
            p.beta += db;
            CHECK(garch_loglik(y.data(), y.size(), Dist::StudentT, p, f.backcast) < f.loglik);
        }
}

TEST(variance_targeting_constrains_omega) {
    const std::vector<double> y = simulate_garch({0.0, 0.02, 0.08, 0.90, 6.0}, 10000, 41);
    GarchSpec spec;
    spec.variance_targeting = true;
    const FitResult f = fit_garch(y, spec);
    CHECK(f.status == FitStatus::Ok);
    double ss = 0;
    for (double v : y) ss += v * v;
    CHECK_REL(f.params.omega, (1 - f.params.alpha - f.params.beta) * ss / static_cast<double>(y.size()), 1e-10);
    CHECK(f.n_params == 3 && std::isnan(f.std_err.omega) && f.std_err.alpha > 0);
    CHECK_NEAR(f.params.alpha, 0.08, 0.02, 0.0);
}

TEST(fit_handles_degenerate_and_dirty_inputs) {
    std::vector<double> flat(1000, 0.001);
    GarchSpec cm{Dist::StudentT, MeanModel::Constant};
    CHECK(fit_garch(flat, cm).status == FitStatus::DegenerateData);
    CHECK(fit_garch(std::vector<double>(1000, 0.0), GarchSpec{}).status == FitStatus::DegenerateData);
    CHECK(fit_garch(std::vector<double>(50, 0.01), GarchSpec{}).status == FitStatus::InsufficientData);
    CHECK(fit_garch(std::vector<double>(1000, kNaN), GarchSpec{}).status == FitStatus::InsufficientData);
    CHECK(fit_garch(std::vector<double>{}, GarchSpec{}).status == FitStatus::InsufficientData);

    const GarchParams truth{0.0, 0.02, 0.08, 0.90, 6.0};
    std::vector<double> y = simulate_garch(truth, 10000, 51);
    const FitResult clean = fit_garch(y, GarchSpec{});
    for (size_t t = 3; t < y.size(); t += 20) y[t] = kNaN;  // 5% missing
    y[17] = std::numeric_limits<double>::infinity();
    const FitResult dirty = fit_garch(y, GarchSpec{});
    CHECK(dirty.status == FitStatus::Ok);
    CHECK_NEAR(dirty.params.alpha, clean.params.alpha, 0.02, 0.0);
    CHECK_NEAR(dirty.params.beta, clean.params.beta, 0.03, 0.0);

    y[5000] = 1000.0 * std::sqrt(truth.unconditional_variance());  // absurd bad tick
    const FitResult spike = fit_garch(y, GarchSpec{});
    CHECK(spike.usable() && spike.params.is_valid(Dist::StudentT));
    CHECK(std::isfinite(spike.next_variance) && spike.next_variance > 0);

    // i.i.d. data: no volatility clustering; the fit must still be valid
    Rng rng(61);
    std::vector<double> iid(5000);
    for (double& v : iid) v = 0.01 * rng.normal();
    const FitResult fi = fit_garch(iid, GarchSpec{});
    CHECK(fi.usable() && fi.params.is_valid(Dist::StudentT));
    CHECK(fi.params.alpha < 0.05);
    CHECK(fi.params.nu > 20);  // Normal data -> large nu

    // near-integrated volatility
    const FitResult fp = fit_garch(simulate_garch({0.0, 0.001, 0.05, 0.945, 6.0}, 20000, 71), GarchSpec{});
    CHECK(fp.usable() && fp.params.persistence() < 1.0 && fp.params.persistence() > 0.98);
}

// ================================================================== filter

TEST(filter_reproduces_in_sample_path_and_forecasts) {
    const GarchParams truth{0.0, 0.02, 0.08, 0.90, 6.0};
    const std::vector<double> y = simulate_garch(truth, 3000, 81);
    const FitResult f = fit_garch(std::vector<double>(y.begin(), y.begin() + 2000), GarchSpec{});
    std::vector<double> h(3000);
    garch_variance_path(y.data(), y.size(), f.params, f.backcast, h.data());
    CHECK_REL(h[2000], f.next_variance, 1e-14);
    GarchFilter flt(f.params, f.next_variance);
    for (size_t t = 2000; t < 3000; ++t) {
        CHECK_REL(flt.variance(), h[t], 1e-12);
        const double z = flt.update(y[t]);
        CHECK_REL(z, y[t] / std::sqrt(h[t]), 1e-12);
    }
    // multi-step forecasts vs brute-force iteration of E[h_{t+k}]
    double hk = flt.variance(), cum = 0.0;
    for (int k = 1; k <= 50; ++k) {
        CHECK_REL(flt.forecast_variance(k), hk, 1e-12);
        cum += hk;
        CHECK_REL(flt.cumulative_variance(k), cum, 1e-12);
        hk = f.params.omega + f.params.persistence() * hk;
    }
    // missing data and robust clipping
    GarchFilter a = flt, b = flt;
    a.update(kNaN);
    b.update_missing();
    CHECK_REL(a.variance(), b.variance(), 1e-15);
    GarchFilter c1 = flt, c2 = flt;
    const double big = 50.0 * std::sqrt(flt.variance());
    c1.update(big);
    c2.update(big, 6.0);
    CHECK(c2.variance() < c1.variance());
    CHECK_REL(c2.variance(), f.params.omega + (f.params.alpha * 36.0 + f.params.beta) * flt.variance(), 1e-12);
    // integrated (EWMA) filter: omega = 0, alpha + beta = 1
    GarchFilter ew(GarchParams{0.0, 0.0, 0.06, 0.94, kInf}, 2.0);
    CHECK_REL(ew.forecast_variance(10), 2.0, 1e-15);
    CHECK_REL(ew.cumulative_variance(10), 20.0, 1e-15);
}

// ================================================================== seasonality

TEST(diurnal_profile_recovers_u_shape) {
    const int K = 28;
    std::vector<double> truth(K);
    double ms = 0;
    for (int k = 0; k < K; ++k) {
        truth[k] = 1.0 + 1.2 * std::exp(-k / 2.5) + 0.5 * std::exp(-(K - 1 - k) / 3.0);
        ms += truth[k] * truth[k];
    }
    for (double& v : truth) v /= std::sqrt(ms / K);
    Rng rng(91);
    std::vector<double> r;
    std::vector<int> slot;
    for (int d = 0; d < 3000; ++d)
        for (int k = 0; k < K; ++k) {
            r.push_back(0.002 * truth[k] * rng.std_t(6.0));
            slot.push_back(k);
        }
    r[100] = kNaN;
    const DiurnalProfile p = DiurnalProfile::estimate(r.data(), slot.data(), r.size(), K);
    double check = 0;
    for (int k = 0; k < K; ++k) {
        CHECK_NEAR(p[k], truth[k], 0.0, 0.04);
        check += p[k] * p[k];
    }
    CHECK_REL(check / K, 1.0, 1e-12);
    const DiurnalProfile flat(K);
    CHECK(flat[5] == 1.0 && flat.size() == K);
}

// ================================================================== portfolio engine

namespace {
EngineConfig small_config() {
    EngineConfig c;
    c.bars_per_day = 4;
    c.window_bars = 4000;
    c.min_fit_bars = 400;
    c.threads = 2;
    return c;
}
}  // namespace

TEST(engine_single_asset_matches_univariate_formula) {
    EngineConfig cfg = small_config();
    cfg.diurnal = false;
    IntradayRiskEngine eng({"AAA"}, cfg);
    const std::vector<double> y = simulate_garch({0.0, 2e-7, 0.07, 0.90, 5.0}, 3000, 101);
    const double x[1] = {-2.5e6};  // short position
    CHECK(!eng.ready());
    CHECK(std::isnan(eng.compute(0, x).sigma));
    for (size_t t = 0; t < y.size(); ++t) eng.on_bar(static_cast<int>(t % 4), &y[t]);
    const RefitReport rep = eng.refit(x);
    CHECK(rep.ready && eng.ready() && rep.n_fitted == 1 && rep.window_bars == 3000);
    const PortfolioRisk r = eng.compute(0, x);
    const AssetModel& a = eng.asset(0);
    const double sd = std::sqrt(a.filter.variance());
    CHECK_REL(r.sigma, 2.5e6 * sd, 1e-12);
    CHECK_REL(r.sigma_contrib[0], r.sigma, 1e-12);
    for (size_t k = 0; k < cfg.confidences.size(); ++k) {
        const TailFactors f = r.tail.factors(cfg.confidences[k]);  // t body + GPD tail (EVT)
        CHECK_REL(r.measures[k].var, r.sigma * f.var, 1e-12);
        CHECK_REL(r.measures[k].es, r.sigma * f.es, 1e-12);
    }
    CHECK(r.nu > 2.5 && r.nu < 12);  // calibrated tail close to the simulated nu = 5
    // live update moves the forecast exactly like the univariate filter
    GarchFilter ref = a.filter;
    const double next = 0.004;
    eng.on_bar(0, &next);
    ref.update(next);
    CHECK_REL(eng.asset(0).filter.variance(), ref.variance(), 1e-14);
}

TEST(engine_hedges_and_correlations) {
    EngineConfig cfg = small_config();
    IntradayRiskEngine eng({"A", "B", "C"}, cfg);
    Rng rng(111);
    const GarchParams g{0.0, 2e-7, 0.06, 0.92, 6.0};
    double h = g.unconditional_variance();
    for (int t = 0; t < 3000; ++t) {
        const double e = std::sqrt(h) * rng.std_t(6.0);
        h = g.omega + g.alpha * e * e + g.beta * h;
        const double s = 1.0 + 0.8 * ((t % 4) == 0);  // simple intraday pattern
        const double c = std::sqrt(h) * rng.normal() * s;
        const double r[3] = {e * s, e * s, c};  // A and B identical, C independent
        eng.on_bar(t % 4, r);
    }
    const double x_long[3] = {1e6, 1e6, 0.0};
    eng.refit(x_long);
    const std::vector<double>& R = eng.correlation();
    CHECK_NEAR(R[1], 1.0, 1e-9, 0.0);
    CHECK_NEAR(R[2], 0.0, 0.3, 0.0);  // EWMA(0.99): effective sample ~200 bars
    CHECK(is_psd_corr(R, 3));
    CHECK(eng.diurnal()[0] > 1.3 * eng.diurnal()[1]);

    const PortfolioRisk both = eng.compute(1, x_long);
    const double x_a[3] = {1e6, 0.0, 0.0};
    const PortfolioRisk one = eng.compute(1, x_a);
    CHECK_REL(both.sigma, 2 * one.sigma, 1e-6);  // perfectly correlated: additive
    const double x_hedge[3] = {1e6, -1e6, 0.0};
    CHECK(eng.compute(1, x_hedge).sigma < 1e-3 * one.sigma);  // perfect hedge
    const double x_mix[3] = {1e6, -3e5, 7e5};
    const PortfolioRisk mix = eng.compute(2, x_mix);
    double sum = 0;
    for (double v : mix.sigma_contrib) sum += v;
    CHECK_REL(sum, mix.sigma, 1e-12);
    CHECK(mix.measures[2].var > mix.measures[0].var && mix.measures[2].es > mix.measures[2].var);
    // first bar of the day is riskier through the diurnal factor
    CHECK(eng.compute(0, x_mix).sigma > eng.compute(2, x_mix).sigma);

    // live updates with missing data keep everything finite and R a valid correlation
    for (int t = 0; t < 200; ++t) {
        double r[3] = {0.001 * rng.normal(), 0.001 * rng.normal(), 0.001 * rng.normal()};
        if (t % 7 == 0) r[1] = kNaN;
        eng.on_bar(t % 4, r);
    }
    const PortfolioRisk after = eng.compute(3, x_mix);
    CHECK(std::isfinite(after.sigma) && std::isfinite(after.measures[0].var));
    CHECK(is_psd_corr(eng.correlation(), 3));
    const double nu_before = eng.portfolio_nu();
    const double nu_after = eng.recalibrate_tail(x_hedge);
    CHECK(std::isfinite(nu_before) && std::isfinite(nu_after));
}

TEST(engine_dead_feed_does_not_block_correlations) {
    EngineConfig cfg = small_config();
    cfg.correlation = CorrelationMode::Ewma;  // deterministic live update for the last check
    IntradayRiskEngine eng({"A", "DEAD", "B"}, cfg);
    Rng rng(131);
    for (int t = 0; t < 2000; ++t) {
        const double f = 0.002 * rng.std_t(6.0);
        const double r[3] = {f + 0.001 * rng.normal(), kNaN, f + 0.001 * rng.normal()};
        eng.on_bar(t % 4, r);
    }
    const double x[3] = {1e6, 0.0, 1e6};
    const RefitReport rep = eng.refit(x);
    CHECK(rep.ready && rep.n_inactive == 1 && rep.n_fallback == 1 && rep.complete_rows == 2000);
    CHECK(!eng.asset(1).active && eng.asset(0).active && eng.asset(2).active);
    const std::vector<double>& R = eng.correlation();
    CHECK(R[2] > 0.6);                            // A-B correlation still estimated (true 0.8)
    CHECK(R[1] == 0.0 && R[5] == 0.0 && R[4] == 1.0);  // dead asset: identity block
    CHECK(is_psd_corr(R, 3));
    const PortfolioRisk risk = eng.compute(1, x);
    CHECK(std::isfinite(risk.sigma) && risk.sigma > 0);
    // live bars keep updating the correlation of the active pair
    const std::vector<double> before = R;
    const double r[3] = {0.01, kNaN, -0.01};
    eng.on_bar(1, r);
    CHECK(eng.correlation()[2] < before[2]);
}

TEST(engine_normal_spec_and_constant_correlation) {
    EngineConfig cfg = small_config();
    cfg.spec.dist = Dist::Normal;
    cfg.correlation = CorrelationMode::Constant;
    IntradayRiskEngine eng({"A", "B"}, cfg);
    Rng rng(121);
    for (int t = 0; t < 2000; ++t) {
        const double f = 0.002 * rng.normal();
        const double r[2] = {f + 0.001 * rng.normal(), 0.5 * f + 0.002 * rng.normal()};
        eng.on_bar(t % 4, r);
    }
    const double x[2] = {5e5, 5e5};
    eng.refit(x);
    const PortfolioRisk r = eng.compute(0, x);
    CHECK(std::isinf(r.nu));
    CHECK_REL(r.measures[2].var, r.sigma * math::norm_ppf(0.99), 1e-12);
    const std::vector<double> R0 = eng.correlation();
    const double z[2] = {0.01, -0.01};
    eng.on_bar(1, z);
    CHECK(eng.correlation() == R0);  // constant mode: no intraday change
}

TEST(engine_validates_configuration) {
    bool threw = false;
    try {
        IntradayRiskEngine e({}, EngineConfig{});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    threw = false;
    EngineConfig bad;
    bad.confidences = {0.99, 1.0};
    try {
        IntradayRiskEngine e({"A"}, bad);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    IntradayRiskEngine ok({"A"}, small_config());
    threw = false;
    const double r = 0.0;
    try {
        ok.on_bar(4, &r);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    CHECK(threw);
}

// ================================================================== backtesting

TEST(coverage_tests_match_reference) {
    std::vector<int> hits(250, 0);
    for (int i = 0; i < 8; ++i) hits[static_cast<size_t>(i * 30)] = 1;
    const CoverageTest k = coverage_test(hits, 0.01);
    CHECK(k.exceptions == 8 && k.n == 250);
    CHECK_REL(k.lr_uc, 7.733550724494528, 1e-12);
    CHECK_REL(k.p_uc, 0.005420405194127773, 1e-10);
    CHECK(k.zone == TrafficLight::Yellow);
    const CoverageTest z = coverage_test(std::vector<int>(250, 0), 0.01);
    CHECK_REL(z.lr_uc, 5.025167926750726, 1e-12);
    CHECK(z.zone == TrafficLight::Green);

    std::vector<int> cl(100, 0);
    for (int i : {10, 11, 12, 40, 41, 70, 90, 91, 92, 93}) cl[static_cast<size_t>(i)] = 1;
    const CoverageTest c = coverage_test(cl, 0.05);
    CHECK_REL(c.lr_uc, 4.130843782549277, 1e-12);
    CHECK_REL(c.lr_ind, 18.7083159942539, 1e-12);
    CHECK_REL(c.p_ind, 1.5231675555657085e-05, 1e-9);
    CHECK_REL(c.p_cc, 1.0978410578674944e-05, 1e-9);
    std::vector<int> red(250, 0);
    for (int i = 0; i < 10; ++i) red[static_cast<size_t>(i * 20)] = 1;
    CHECK(coverage_test(red, 0.01).zone == TrafficLight::Red);
}

// ================================================================== market data

TEST(timestamps_slots_and_returns) {
    int d, m, s;
    CHECK(parse_timestamp("2026-03-02 10:15:00", d, m, &s) && d == 20260302 && m == 615 && s == 0);
    CHECK(parse_timestamp("2026-12-31T17:00", d, m) && d == 20261231 && m == 1020);
    CHECK(!parse_timestamp("2026-13-01 10:00:00", d, m));
    CHECK(!parse_timestamp("garbage", d, m));
    const SessionSpec ss;
    CHECK(bar_slot(615, ss) == 0 && bar_slot(1020, ss) == 27);
    CHECK(bar_slot(600, ss) == -1 && bar_slot(1035, ss) == -1 && bar_slot(620, ss) == -1);

    const std::string path = "garcht_test_bars.csv";
    {
        std::ofstream f(path);
        f << "timestamp,ticker,open,high,low,close,volume\n"
          << "2026-03-02 10:15:00,AAA,10.00,10.2,9.9,10.10,100\n"
          << "2026-03-02 10:15:00,BBB,20.00,20.1,19.9,20.00,100\n"
          << "2026-03-02 10:15:00,AAA,99.00,99,99,99.00,100\n"  // duplicate: ignored
          << "2026-03-02 10:30:00,BBB,20.00,20.2,19.9,20.20,100\n"  // AAA missing here
          << "2026-03-02 10:45:00,AAA,10.30,10.4,10.2,10.40,100\n"
          << "2026-03-02 10:45:00,BBB,20.20,20.3,20.1,abc,100\n"  // bad row
          << "2026-03-02 10:45:00,ZZZ,1,1,1,1,1\n"  // unknown ticker
          << "2026-03-03 10:15:00,AAA,10.80,10.9,10.7,10.90,100\n"
          << "2026-03-03 10:15:00,BBB,0,0,0,0,100\n";  // non-positive price
    }
    const BarPanel p = load_bars_csv(path, {"AAA", "BBB"});
    std::remove(path.c_str());
    CHECK(p.num_bars() == 4 && p.num_assets() == 2);
    CHECK(p.duplicates == 1 && p.bad_rows == 1 && p.unknown_tickers == 1 && p.rows_read == 9);
    CHECK(p.at(p.close, 0, 0) == 10.10);
    std::vector<double> r;
    std::vector<int> sl;
    bar_returns(p, ss, r, sl);
    CHECK(sl[0] == 0 && sl[1] == 1 && sl[2] == 2 && sl[3] == 0);
    CHECK_REL(r[0], std::log(10.10 / 10.00), 1e-15);                     // first bar: close/open
    CHECK(std::isnan(r[2]));                                            // AAA missing at 10:30
    CHECK_REL(r[4], std::log(10.40 / 10.10) / std::sqrt(2.0), 1e-15);   // 2-bar return, scaled
    CHECK_REL(r[3], std::log(20.20 / 20.00), 1e-15);
    CHECK(std::isnan(r[5]));                                            // bad row -> missing
    CHECK_REL(r[6], std::log(10.90 / 10.80), 1e-15);                     // new day: overnight excluded
    CHECK(std::isnan(r[7]));                                            // zero price
}

TEST(random_generator_is_deterministic_and_calibrated) {
    Rng a(5), b(5);
    for (int i = 0; i < 100; ++i) CHECK(a.next_u64() == b.next_u64());
    Rng r(6);
    const int N = 1000000;
    double s1 = 0, s2 = 0;
    for (int i = 0; i < N; ++i) {
        const double z = r.std_t(8.0);
        s1 += z;
        s2 += z * z;
    }
    CHECK_NEAR(s1 / N, 0.0, 0.005, 0.0);
    CHECK_NEAR(s2 / N, 1.0, 0.01, 0.0);
}


// ================================================================== data input (v2)

TEST(csv_rfc4180_semicolons_and_decimal_commas) {
    const CsvTable a = parse_csv("a,b,c\n1,\"x, y\",3\n\"he said \"\"hi\"\"\",\"multi\nline\",  6 \n# comment\n\n");
    CHECK(a.delimiter == ',' && a.header.size() == 3 && a.rows.size() == 2);
    CHECK(a.rows[0][1] == "x, y");
    CHECK(a.rows[1][0] == "he said \"hi\"" && a.rows[1][1] == "multi\nline" && a.rows[1][2] == "6");
    const CsvTable b = parse_csv("\xEF\xBB\xBF" "Data;Ticker;Close\r\n2026-07-17 10:15;PETR4;38,50\r\n");
    CHECK(b.delimiter == ';' && b.column("ticker") == 1 && b.column("CLOSE") == 2);
    double v;
    CHECK(parse_number(b.rows[0][2], v) && v == 38.5);
    CHECK(parse_number("1.234,56", v) && std::fabs(v - 1234.56) < 1e-9);
    CHECK(parse_number("1,234.56", v) && std::fabs(v - 1234.56) < 1e-9);
    CHECK(parse_number("1.234.567", v) && v == 1234567.0);
    CHECK(parse_number("-1.5e-3", v) && v == -1.5e-3);
    CHECK(parse_number("", v) && std::isnan(v));
    CHECK(parse_number("NA", v) && std::isnan(v));
    CHECK(!parse_number("12abc", v));
    CHECK(!parse_number("1.2.3,4,5", v));
}

TEST(timestamps_formats_zones_and_ordering) {
    TimePoint a, b, c, d, e;
    CHECK(parse_time("2026-07-17 10:15:00", a) && a.ymd == 20260717 && a.minute == 615);
    CHECK(parse_time("17/07/2026 10:15", b) && b.key() == a.key());
    CHECK(parse_time("2026-07-17T13:15:00Z", c) && c.key() == a.key());  // UTC -> B3 (UTC-3)
    CHECK(parse_time("2026-07-17T10:15:00-03:00", d) && d.key() == a.key());
    CHECK(parse_time("2026-07-17 10:15:00.250", e) && e.key() == a.key() + 250 && e.str() == "2026-07-17 10:15:00");
    TimePoint f;
    CHECK(parse_time("2026-01-01T01:30:00Z", f) && f.ymd == 20251231 && f.minute == 22 * 60 + 30);  // day rollover
    TimePoint g;
    CHECK(parse_time("2024-02-29 09:00", g) && !parse_time("2026-02-29 09:00", g));  // leap years
    CHECK(!parse_time("2026-07-17", g) && !parse_time("2026-07-17 25:00", g) && !parse_time("17/07/26 10:00", g));
    TimePoint h1, h2;
    CHECK(parse_time("2026-07-17 09:59:59", h1) && parse_time("2026-07-17 10:00:00", h2) && h1.key() < h2.key());
    TimePoint i1, i2;  // key ordering across months / years
    CHECK(parse_time("2026-12-31 17:00", i1) && parse_time("2027-01-04 10:15", i2) && i1.key() < i2.key());
}

TEST(panel_from_unsorted_mixed_rows_and_session_detection) {
    const std::string path = "garcht_test_mixed.csv";
    {
        std::ofstream f(path);
        f << "datetime;symbol;close;open\n"  // semicolons, decimal commas, mixed formats, unsorted
          << "18/07/2026 10:30;aaa;10,40;10,30\n"
          << "2026-07-17 10:30:00;AAA;10,20;10,10\n"
          << "2026-07-17T13:15:00Z;AAA;10,10;10,00\n"  // = 10:15 local
          << "2026-07-18 10:15;AAA;10,30;10,25\n"
          << "2026-07-18 10:45;AAA;10,35;10,40\n"
          << "2026-07-18 11:00;ZZZ;1;1\n";  // unknown ticker: must not create a bar
    }
    const BarPanel p = load_bars_csv(path, {"AAA"});
    std::remove(path.c_str());
    CHECK(p.num_bars() == 5 && p.unknown_tickers == 1 && p.bad_rows == 0);
    CHECK(p.timestamps[0] == "2026-07-17 10:15:00" && p.timestamps[4] == "2026-07-18 10:45:00");
    for (size_t t = 1; t < p.num_bars(); ++t) CHECK(p.time_key[t] > p.time_key[t - 1]);
    const SessionSpec s = detect_session(p);
    CHECK(s.bar_minutes == 15 && s.open_minute == 600);
    std::vector<double> rr, gaps;
    std::vector<int> sl;
    bar_returns(p, s, rr, sl);
    overnight_returns(p, sl, gaps);
    CHECK(std::isnan(gaps[0]));
    CHECK_REL(gaps[2], std::log(10.25 / 10.20), 1e-14);  // first bar of day 2: open / previous close
}

// ================================================================== backtest (v2)

TEST(wilson_berkowitz_match_reference) {
    double lo, hi;
    wilson_interval(8, 250, 0.95, lo, hi);
    CHECK_REL(lo, 0.01630236549364963, 1e-12);
    CHECK_REL(hi, 0.06186240283408649, 1e-12);
    wilson_interval(0, 100, 0.95, lo, hi);
    CHECK_NEAR(lo, 0.0, 1e-15, 0.0);
    CHECK_REL(hi, 0.03699349820698568, 1e-12);
    std::vector<double> pit(300);
    for (size_t t = 0; t < pit.size(); ++t) pit[t] = math::norm_cdf(0.8 * std::sin(0.7 * static_cast<double>(t)) + 0.3);
    const BerkowitzTest b = berkowitz_test(pit);
    CHECK_REL(b.rho, 0.7677539883990782, 1e-10);
    CHECK_REL(b.sigma, 0.3646776162591978, 1e-10);
    CHECK_REL(b.lr, 428.3508736636876, 1e-10);
    // a well-specified forecast passes, a too-narrow one fails
    Rng rng(201);
    std::vector<double> good, narrow;
    for (int t = 0; t < 3000; ++t) {
        const double x = 2.0 * rng.std_t(5.0);
        good.push_back(forecast_cdf(DistForecast(0.0, 2.0, 5.0), x));
        narrow.push_back(forecast_cdf(DistForecast(0.0, 1.6, 5.0), x));
    }
    CHECK(berkowitz_test(good).p > 0.01);
    CHECK(berkowitz_test(narrow).p < 1e-6);
}

TEST(acerbi_szekely_detects_thin_tails) {
    Rng rng(211);
    const int T = 3000;
    std::vector<double> x(T);
    for (double& v : x) v = rng.std_t(3.5);  // heavy-tailed outcomes, unit variance
    std::vector<DistForecast> right(T, DistForecast(0.0, 1.0, 3.5)), normal(T, DistForecast(0.0, 1.0, kInf));
    const EsTest ok = acerbi_szekely_test(x, right, 0.975, 1000, 1);
    const EsTest bad = acerbi_szekely_test(x, normal, 0.975, 1000, 1);
    const EsTest bad99 = acerbi_szekely_test(x, normal, 0.99, 1000, 1);
    CHECK(ok.p1 > 0.01 && ok.p2 > 0.01);
    // At 97.5% the unit-variance t has *fewer* exceedances of the Normal VaR but larger
    // losses: Z2 (frequency x size) is nearly blind, Z1 (size given an exception) rejects.
    CHECK(bad.z1 < 0.0 && bad.p1 < 0.01);
    CHECK(bad99.z2 < 0.0 && bad99.p2 < 0.01);  // further in the tail Z2 rejects as well
    CHECK(ok.n == T && ok.exceptions > 0);
}

// ================================================================== tail distributions

TEST(shock_distribution_evt_is_consistent) {
    ShockDist d = ShockDist::student(6.0);
    d.gpd = true;
    d.u0 = 1.6;
    d.beta = 0.55;
    d.xi = 0.2;
    d.theta = 0.08;
    for (double c : {0.95, 0.975, 0.99, 0.999}) {
        const TailFactors f = d.factors(c);
        CHECK_NEAR(d.cdf(-f.var), 1.0 - c, 1e-12, 1e-9);  // quantile <-> cdf
        CHECK(f.es > f.var);
        // ES = average of the quantile function over the tail (midpoint rule in probability)
        const int N = 200000;
        double acc = 0.0;
        for (int k = 0; k < N; ++k) acc += d.factors(1.0 - (1.0 - c) * (k + 0.5) / N).var;
        CHECK_REL(f.es, acc / N, 2e-3);
    }
    CHECK_NEAR(d.cdf(-d.u0), d.theta, 1e-12, 0.0);  // continuity at the threshold
    const TailFactors body = d.factors(0.85);        // quantile inside the t body
    CHECK_NEAR(d.cdf(-body.var), 0.15, 1e-12, 1e-9);
    Rng rng(221);
    int below = 0;
    const int M = 400000;
    for (int k = 0; k < M; ++k) below += d.sample(rng) < -d.factors(0.99).var;
    CHECK_NEAR(below / static_cast<double>(M), 0.01, 0.0006, 0.0);
    CHECK(ShockDist::student(5.0) == ShockDist::student(5.0) && !(d == ShockDist::student(6.0)));
}

TEST(evt_fit_recovers_gpd_tail) {
    ShockDist truth = ShockDist::student(8.0);
    truth.gpd = true;
    truth.u0 = 1.4;
    truth.beta = 0.6;
    truth.xi = 0.25;
    truth.theta = 0.10;
    Rng rng(231);
    std::vector<double> u(200000);
    for (double& v : u) v = truth.sample(rng);
    const ShockDist fit = fit_evt_shock(u, 8.0, 0.10, 50);
    CHECK(fit.gpd);
    CHECK_NEAR(fit.xi, 0.25, 0.04, 0.0);
    CHECK_NEAR(fit.factors(0.99).var, truth.factors(0.99).var, 0.0, 0.02);
    CHECK_NEAR(fit.factors(0.99).es, truth.factors(0.99).es, 0.0, 0.03);
    const ShockDist few = fit_evt_shock(std::vector<double>(300, 0.5), 8.0, 0.10, 50);  // too few exceedances
    CHECK(!few.gpd && few.nu == 8.0);
    const ShockDist t = fit_student_shock(std::vector<double>(u.begin(), u.begin() + 20000), 2.1, 300.0);
    CHECK(t.nu > 3.0 && t.nu < 15.0 && !t.gpd);
}

// ================================================================== revaluation

TEST(black_scholes_matches_reference) {
    CHECK_REL(bs_price(OptionType::Call, 100, 100, 1.0, 0.2, 0.05), 10.450583572185565, 1e-12);
    CHECK_REL(bs_price(OptionType::Put, 100, 100, 1.0, 0.2, 0.05), 5.573526022256971, 1e-12);
    CHECK_REL(bs_delta(OptionType::Call, 100, 100, 1.0, 0.2, 0.05), 0.6368306511756191, 1e-12);
    CHECK_REL(bs_gamma(100, 100, 1.0, 0.2, 0.05), 0.018762017345846895, 1e-12);
    CHECK_REL(bs_price(OptionType::Put, 95, 105, 0.5, 0.35, 0.14, 0.03), 11.742034872020731, 1e-12);
    CHECK_REL(bs_delta(OptionType::Put, 95, 105, 0.5, 0.35, 0.14, 0.03), -0.5155025309926707, 1e-12);
    // put-call parity and finite-difference greeks
    for (double s : {80.0, 100.0, 125.0}) {
        const double c = bs_price(OptionType::Call, s, 100, 0.3, 0.3, 0.1, 0.02);
        const double p = bs_price(OptionType::Put, s, 100, 0.3, 0.3, 0.1, 0.02);
        CHECK_REL(c - p, s * std::exp(-0.02 * 0.3) - 100 * std::exp(-0.1 * 0.3), 1e-11);
        const double h = 1e-4 * s;
        const double fd = (bs_price(OptionType::Call, s + h, 100, 0.3, 0.3, 0.1, 0.02) -
                           bs_price(OptionType::Call, s - h, 100, 0.3, 0.3, 0.1, 0.02)) / (2 * h);
        CHECK_REL(bs_delta(OptionType::Call, s, 100, 0.3, 0.3, 0.1, 0.02), fd, 1e-7);
    }
    CHECK(bs_price(OptionType::Call, 110, 100, 0.0, 0.3, 0.1) == 10.0);  // expired: intrinsic
    CHECK(bs_price(OptionType::Put, 110, 100, -1.0, 0.3, 0.1) == 0.0);
}

TEST(monte_carlo_matches_analytic_for_stocks_and_captures_gamma) {
    // two correlated stocks, multivariate t: MC VaR/ES of a linear book ~ analytic t values
    JointForecast jf;
    jf.mu = {0.0, 0.0};
    jf.sigma = {0.004, 0.006};
    jf.corr = {1.0, 0.6, 0.6, 1.0};
    jf.nu = 5.0;
    MonteCarloRevaluation mc(200000, 3);
    mc.simulate(jf);
    std::vector<Position> book(2);
    book[0].asset = 0;
    book[0].quantity = 10000;
    book[1].asset = 1;
    book[1].quantity = -4000;
    const std::vector<double> spot{50.0, 80.0};
    const McRiskResult r = mc.evaluate(book, spot, 0.0, {0.95, 0.99});
    const double x0 = 10000 * 50.0, x1 = -4000 * 80.0;
    const double sd = std::sqrt(x0 * x0 * 0.004 * 0.004 + x1 * x1 * 0.006 * 0.006 + 2 * 0.6 * x0 * x1 * 0.004 * 0.006);
    for (size_t k = 0; k < 2; ++k) {
        const TailFactors f = tail_factors(Dist::StudentT, 5.0, r.measures[k].confidence);
        // exp(r) - 1 vs r: second-order difference only
        CHECK_NEAR(r.measures[k].var, sd * f.var, 0.0, 0.02);
        CHECK_NEAR(r.measures[k].es, sd * f.es, 0.0, 0.03);
    }
    // sample correlation of the scenarios
    const std::vector<double>& sc = mc.scenarios();
    double s00 = 0, s11 = 0, s01 = 0;
    for (int s = 0; s < mc.num_scenarios(); ++s) {
        s00 += sc[2 * s] * sc[2 * s];
        s11 += sc[2 * s + 1] * sc[2 * s + 1];
        s01 += sc[2 * s] * sc[2 * s + 1];
    }
    CHECK_NEAR(s01 / std::sqrt(s00 * s11), 0.6, 0.01, 0.0);
    CHECK_NEAR(std::sqrt(s00 / mc.num_scenarios()), 0.004, 0.0, 0.01);

    // short straddle: delta-neutral, so a linear model sees ~no risk; full revaluation does
    std::vector<Position> straddle(2);
    for (int k = 0; k < 2; ++k) {
        straddle[static_cast<size_t>(k)].kind = Position::Kind::Option;
        straddle[static_cast<size_t>(k)].asset = 0;
        straddle[static_cast<size_t>(k)].quantity = -100;
        straddle[static_cast<size_t>(k)].multiplier = 100;
        straddle[static_cast<size_t>(k)].strike = 50;
        straddle[static_cast<size_t>(k)].expiry = 2.0 / 252;  // two days left: large gamma
        straddle[static_cast<size_t>(k)].vol = 0.3;
    }
    straddle[0].type = OptionType::Call;
    straddle[1].type = OptionType::Put;
    double delta_exposure = 0.0;
    for (const Position& p : straddle) delta_exposure += position_delta_exposure(p, 50.0);
    JointForecast big = jf;
    big.sigma = {0.02, 0.02};
    mc.simulate(big);
    const McRiskResult s = mc.evaluate(straddle, spot, 0.0, {0.99});
    CHECK(std::fabs(delta_exposure) < 0.05 * 100 * 100 * 50);  // nearly delta neutral
    CHECK(s.measures[0].var > 3.0 * std::fabs(delta_exposure) * 0.02 * 3.0);  // gamma loss dominates
    std::vector<double> A = {1, 1, 1, 1}, L;  // perfectly correlated: semi-definite Cholesky
    CHECK(cholesky_psd(A, 2, L) && L[0] == 1.0 && L[2] == 1.0 && L[3] == 0.0);
}

// ================================================================== engine (v2)

TEST(dcc_estimation_recovers_parameters) {
    // three assets with DCC(1,1) correlation dynamics, a = 0.05, b = 0.90
    const double a = 0.05, b = 0.90;
    const int n = 3, T = 6000;
    const double Rbar[9] = {1, 0.5, 0.3, 0.5, 1, 0.4, 0.3, 0.4, 1};
    std::vector<double> Q(Rbar, Rbar + 9);
    Rng rng(241);
    EngineConfig cfg;
    cfg.bars_per_day = 4;
    cfg.window_bars = T;
    cfg.min_fit_bars = 500;
    cfg.diurnal = false;
    cfg.threads = 2;
    cfg.correlation = CorrelationMode::Dcc;
    IntradayRiskEngine eng({"A", "B", "C"}, cfg);
    std::vector<double> L;
    for (int t = 0; t < T; ++t) {
        std::vector<double> R(9);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) R[i * n + j] = Q[i * n + j] / std::sqrt(Q[i * n + i] * Q[j * n + j]);
        cholesky_psd(R, n, L);
        double g[3] = {rng.normal(), rng.normal(), rng.normal()}, z[3];
        for (int i = 0; i < n; ++i) {
            z[i] = 0.0;
            for (int k = 0; k <= i; ++k) z[i] += L[i * n + k] * g[k];
        }
        const double r[3] = {0.002 * z[0], 0.003 * z[1], 0.0025 * z[2]};
        eng.on_bar(t % 4, r);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) Q[i * n + j] = (1 - a - b) * Rbar[i * n + j] + a * z[i] * z[j] + b * Q[i * n + j];
    }
    const double x[3] = {1e6, 1e6, 1e6};
    const RefitReport rep = eng.refit(x);
    CHECK(rep.dcc_estimated);
    CHECK_NEAR(rep.dcc_a, a, 0.02, 0.0);
    CHECK_NEAR(rep.dcc_b, b, 0.06, 0.0);
    CHECK(is_psd_corr(eng.correlation(), 3));
}

TEST(engine_overnight_tail_calibration_and_threads) {
    EngineConfig cfg = small_config();
    cfg.tail_min_oos = 300;
    IntradayRiskEngine eng({"A", "B"}, cfg);
    Rng rng(251);
    const double gap_sd = 0.01;
    for (int d = 0; d < 400; ++d) {
        const double gaps[2] = {gap_sd * rng.std_t(5.0), gap_sd * rng.std_t(5.0)};
        eng.on_overnight(gaps);
        for (int k = 0; k < 4; ++k) {
            const double f = 0.002 * rng.std_t(6.0);
            const double r[2] = {f + 0.001 * rng.normal(), 0.5 * f + 0.0015 * rng.normal()};
            eng.on_bar(k, r);
        }
    }
    const double x[2] = {1e6, -4e5};
    eng.refit(x);
    const OvernightRisk on = eng.compute_overnight(x);
    CHECK(on.available && on.sigma_gap > 0 && on.sigma_bar > 0);
    CHECK_REL(on.sigma * on.sigma, on.sigma_gap * on.sigma_gap + on.sigma_bar * on.sigma_bar, 1e-12);
    CHECK_NEAR(on.sigma_gap, gap_sd * std::sqrt(1e12 + 1.6e11), 0.0, 0.5);  // EWMA estimate: rough
    CHECK(on.nu > 2.5 && on.measures[2].es > on.measures[2].var);

    // out-of-sample calibration: feed outcomes from a t(4) and check nu_p moves there
    PortfolioRisk fc;
    fc.mean = 0.0;
    fc.sigma = 1.0;
    for (int k = 0; k < 2000; ++k) eng.observe_outcome(rng.std_t(4.0), fc);
    CHECK(eng.oos_outcomes() == 2000);
    // the tail in force was fitted before enough out-of-sample outcomes existed, so a
    // rebalance re-calibrates it; once it is out-of-sample only, rebalances keep it
    const double nu_oos = eng.recalibrate_tail(x);
    CHECK_NEAR(nu_oos, 4.0, 1.0, 0.0);
    CHECK(eng.recalibrate_tail(x) == nu_oos);
    const RefitReport rep = eng.refit(x);
    CHECK_NEAR(rep.portfolio_nu, 4.0, 1.0, 0.0);

    // concurrent readers while the stream keeps updating
    std::vector<std::thread> readers;
    std::atomic<int> bad{0};
    std::atomic<bool> stop{false};
    for (int th = 0; th < 3; ++th)
        readers.emplace_back([&] {
            PortfolioRisk r;
            while (!stop) {
                eng.compute(1, x, r);
                if (!std::isfinite(r.sigma) || !(r.measures[0].var > 0)) ++bad;
                const JointForecast jf = eng.joint_forecast(2);
                if (jf.sigma.size() != 2) ++bad;
            }
        });
    for (int t = 0; t < 3000; ++t) {
        const double r[2] = {0.001 * rng.normal(), 0.001 * rng.normal()};
        eng.on_bar(t % 4, r);
        if (t % 1000 == 999) eng.refit(x);
    }
    stop = true;
    for (auto& th : readers) th.join();
    CHECK(bad == 0);
}

TEST(engine_handles_tick_discrete_assets) {
    EngineConfig cfg = small_config();
    IntradayRiskEngine eng({"LIQ", "TICK"}, cfg);
    Rng rng(261);
    double p = 3.20, logp = std::log(3.20);
    const GarchParams g{0.0, 2e-7, 0.07, 0.91, 5.0};
    double h = g.unconditional_variance();
    for (int t = 0; t < 3000; ++t) {
        const double e = std::sqrt(h) * rng.std_t(5.0);
        h = g.omega + g.alpha * e * e + g.beta * h;
        logp += 0.6 * e;
        const double p_new = std::round(std::exp(logp) * 100.0) / 100.0;  // R$ 0.01 tick at R$ 3
        const double r[2] = {e, std::log(p_new / p)};
        p = p_new;
        eng.on_bar(t % 4, r);
    }
    const double x[2] = {1e6, 1e6};
    eng.refit(x);
    const AssetModel tick = eng.asset(1), liq = eng.asset(0);
    CHECK(tick.zero_share > 0.25 && tick.robust_fit && !liq.robust_fit);
    CHECK(tick.params.persistence() < 0.9995 && tick.params.nu >= 3.0);
    const double uvol = std::sqrt(tick.params.unconditional_variance());
    CHECK(uvol < 0.01);  // a sensible 15-min volatility, not an exploding one
}

TEST(per_asset_diurnal_profiles_shrink_to_pooled) {
    const int K = 8, D = 600, n = 4;
    std::vector<double> r, prof_a(K), prof_b(K);
    std::vector<int> slot;
    for (int k = 0; k < K; ++k) {
        prof_a[static_cast<size_t>(k)] = 1.0 + 1.0 * std::exp(-k / 1.5);  // assets 0,1: open-heavy
        prof_b[static_cast<size_t>(k)] = 1.0 + 1.0 * std::exp(-(K - 1 - k) / 1.5);  // assets 2,3: close-heavy
    }
    Rng rng(271);
    for (int d = 0; d < D; ++d)
        for (int k = 0; k < K; ++k) {
            slot.push_back(k);
            for (int i = 0; i < n; ++i)
                r.push_back(0.002 * (i < 2 ? prof_a : prof_b)[static_cast<size_t>(k)] * rng.std_t(6.0));
        }
    const auto per = DiurnalProfile::estimate_per_asset(r.data(), slot.data(), static_cast<size_t>(K * D), n, K);
    CHECK(per[0][0] > 1.3 * per[0][K - 1]);  // open-heavy kept
    CHECK(per[2][K - 1] > 1.3 * per[2][0]);  // close-heavy kept
    for (int i = 0; i < n; ++i) {
        double ms = 0;
        for (int k = 0; k < K; ++k) ms += per[static_cast<size_t>(i)][k] * per[static_cast<size_t>(i)][k];
        CHECK_REL(ms / K, 1.0, 1e-12);
    }
    // identical assets: per-asset profiles collapse onto the common one (pure noise is shrunk)
    std::vector<double> r2;
    for (int d = 0; d < D; ++d)
        for (int k = 0; k < K; ++k)
            for (int i = 0; i < n; ++i) r2.push_back(0.002 * prof_a[static_cast<size_t>(k)] * rng.std_t(6.0));
    const auto same = DiurnalProfile::estimate_per_asset(r2.data(), slot.data(), static_cast<size_t>(K * D), n, K);
    for (int k = 0; k < K; ++k) CHECK_NEAR(same[0][k], same[3][k], 0.06, 0.0);
}

// ================================================================== synthetic market + replay

TEST(synthetic_market_is_deterministic_and_realistic) {
    SyntheticConfig c;
    c.days = 100;
    const SyntheticMarket a = generate_market(c), b = generate_market(c);
    CHECK(a.rows.size() == b.rows.size() && a.rows[1234].close == b.rows[1234].close);
    CHECK(a.assets.size() == 15 && a.fills.size() > 10 && a.options.size() == 4);
    // circuit breaker on the crash day: no prints at slots 1 and 2 (15 assets x 2 bars missing)
    CHECK(a.rows.size() == static_cast<size_t>(100 * 28 * 15 - 2 * 15));
    // tick discreteness on the R$ 3 stock
    size_t zeros = 0, total = 0;
    double prev = -1.0;
    for (const BarRow& row : a.rows)
        if (row.ticker == "XTEL3") {
            if (prev > 0) {
                ++total;
                zeros += row.close == prev;
            }
            prev = row.close;
        }
    CHECK(zeros > total / 4);
    SyntheticConfig s = c;
    s.realistic = false;
    s.degraded_copy = false;
    const SyntheticMarket m = generate_market(s);
    CHECK(m.rows.size() == static_cast<size_t>(100 * 28 * 15) && m.degraded_rows.empty());
}

TEST(replay_end_to_end_with_options_and_overnight) {
    SyntheticConfig c;
    c.days = 75;
    c.degraded_copy = false;
    const SyntheticMarket m = generate_market(c);
    ReplayInput in;
    in.tickers = m.tickers();
    in.bars = build_panel(m.rows, in.tickers);
    in.initial_quantity = m.initial_quantity;
    in.fills = m.fills;
    in.options = m.options;
    ReplayConfig rc;
    rc.window_days = 40;
    rc.mc_scenarios = 2000;
    rc.engine.threads = 2;
    const ReplayResult r = run_replay(in, rc);
    CHECK(r.session.bars_per_day == 28 && r.sessions == 75);
    CHECK(r.forecasts.size() > 30 * 28);
    CHECK(r.overnight.size() > 25);
    CHECK(r.model_name == "GARCH-t/EVT");
    bool finite = true;
    for (const ForecastRecord& f : r.forecasts)
        finite = finite && std::isfinite(f.pnl) && f.models[0].measures[2].var > 0 && f.book_mc.size() == 3 &&
                 std::isfinite(f.book_pnl) && f.book_mc[2].var > 0 && f.book_delta[2].var > 0;
    CHECK(finite);
    const BacktestReport bt = evaluate_backtest(r, rc.engine.confidences, 200);
    CHECK(bt.rows.size() == 3 * 4 + 3);  // 2 intraday models + 2 book models, x3 levels; overnight x3
    CHECK(std::isfinite(bt.berkowitz[0].test.p) && std::isfinite(bt.berkowitz_overnight.p));
    CHECK(r.var_models.size() == 2 && r.var_models[0].main && r.var_models[1].name == "EWMA-N (ref)");
    CHECK(bt.berkowitz.size() == 2 && bt.berkowitz[1].model == "EWMA-N (ref)");
    // the option revaluation on a worker thread gives the records of the synchronous one, bit for bit
    ReplayConfig sync = rc;
    sync.mc_async = false;
    const ReplayResult rs = run_replay(in, sync);
    bool same = rs.forecasts.size() == r.forecasts.size();
    for (size_t i = 0; same && i < r.forecasts.size(); ++i) {
        const ForecastRecord &a = r.forecasts[i], &b = rs.forecasts[i];
        same = a.book_mc.size() == b.book_mc.size() && a.book_mc_sigma == b.book_mc_sigma &&
               a.models[0].measures[2].var == b.models[0].measures[2].var;
        for (size_t k = 0; same && k < a.book_mc.size(); ++k)
            same = a.book_mc[k].var == b.book_mc[k].var && a.book_mc[k].es == b.book_mc[k].es;
    }
    CHECK(same);
    bool threw = false;
    try {
        ReplayConfig too_long = rc;
        too_long.window_days = 80;
        run_replay(in, too_long);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

TEST(resampled_finer_samples_reach_any_grid) {
    // 5-second samples of one stock aggregate into 1-minute bars: first open, high, low, last close, volume
    std::vector<BarRow> fine;
    for (int k = 1; k <= 24; ++k) {  // 10:00:05 .. 10:02:00
        BarRow r;
        r.time.ymd = 20260105;
        r.time.minute = 600 + (5 * k) / 60;
        r.time.millis = ((5 * k) % 60) * 1000;
        r.ticker = "AAA";
        r.open = r.close = 10.0 + 0.01 * k;
        r.high = r.close + 0.005;
        r.low = r.close - 0.005;
        r.volume = 1.0;
        fine.push_back(r);
    }
    std::reverse(fine.begin(), fine.end());  // the file order does not matter
    const SessionSpec one_minute{600, 1, 420};
    const BarPanel p = build_panel(resample_rows(fine, one_minute), {"AAA"});
    CHECK(p.num_bars() == 2 && p.minute[0] == 601 && p.minute[1] == 602);
    CHECK_REL(p.open[0], 10.01, 1e-12);   // 10:00:05
    CHECK_REL(p.close[0], 10.12, 1e-12);  // 10:01:00 closes the first bar
    CHECK_REL(p.high[0], 10.125, 1e-12);
    CHECK_REL(p.low[0], 10.005, 1e-12);
    CHECK(p.volume[0] == 12.0 && p.volume[1] == 12.0);
    CHECK_REL(p.open[1], 10.13, 1e-12);

    // the synthetic market split into 5-minute sub-bars and resampled to its 15-minute grid is the
    // same panel, and the replay on it the same replay
    SyntheticConfig c;
    c.days = 45;
    c.degraded_copy = false;
    const SyntheticMarket m = generate_market(c);
    std::vector<BarRow> sub;
    for (const BarRow& r : m.rows) {
        double prev = r.open;
        for (int j = 0; j < 3; ++j) {  // 10:00-10:15 -> 10:05, 10:10, 10:15
            BarRow s = r;
            const double cl = j == 2 ? r.close : r.open + (r.close - r.open) * (j + 1) / 3.0;
            s.open = prev;
            s.close = cl;
            s.high = j == 0 ? r.high : std::max(prev, cl);
            s.low = j == 0 ? r.low : std::min(prev, cl);
            s.volume = r.volume / 3.0;
            s.time.minute = r.time.minute - 10 + 5 * j;
            prev = cl;
            sub.push_back(s);
        }
    }
    const std::vector<std::string> tickers = m.tickers();
    const BarPanel original = build_panel(m.rows, tickers);
    const SessionSpec grid = detect_session(original);
    const BarPanel back = build_panel(resample_rows(sub, grid), tickers);
    bool same = back.timestamps == original.timestamps && back.close == original.close && back.open == original.open &&
                back.high == original.high && back.low == original.low;
    for (size_t k = 0; same && k < back.volume.size(); ++k) same = std::fabs(back.volume[k] - original.volume[k]) < 1e-6;
    CHECK(same);
}


// ================================================================== regression tests (review)

TEST(bar_returns_count_gaps_in_grid_slots) {
    BarPanel p;
    p.tickers = {"A", "B"};
    std::vector<BarRow> rows;
    auto add = [&](const char* ts, const char* tk, double o, double c) {
        BarRow r;
        CHECK(parse_time(ts, r.time));
        r.ticker = tk;
        r.open = o;
        r.close = c;
        r.high = std::max(o, c);
        r.low = std::min(o, c);
        r.volume = 1;
        rows.push_back(r);
    };
    add("2026-07-17 10:15", "A", 10.0, 10.1);
    add("2026-07-17 10:15", "B", 20.0, 20.1);
    // 10:30 never printed (halt); 10:45 prints; an off-grid 10:52 row for A only
    add("2026-07-17 10:45", "A", 10.3, 10.4);
    add("2026-07-17 10:45", "B", 20.3, 20.4);
    add("2026-07-17 10:52", "A", 10.4, 10.9);
    add("2026-07-17 11:00", "A", 10.4, 10.5);
    add("2026-07-17 11:00", "B", 20.4, 20.5);
    add("2026-07-17 11:00:30", "B", 99.0, 99.0);  // same bar, same minute: a duplicate
    // next session opens late (11:15): no clean overnight gap
    add("2026-07-20 11:15", "A", 10.6, 10.7);
    add("2026-07-20 11:15", "B", 20.6, 20.7);
    const BarPanel panel = build_panel(rows, {"A", "B"});
    CHECK(panel.duplicates == 1 && panel.num_bars() == 5);
    const SessionSpec ss;
    std::vector<double> r, gaps;
    std::vector<int> sl;
    bar_returns(panel, ss, r, sl);
    // rows: 10:15, 10:45, 10:52 (off-grid), 11:00, 11:15 next day
    CHECK(sl[0] == 0 && sl[1] == 2 && sl[2] == -1 && sl[3] == 3 && sl[4] == 4);
    CHECK_REL(r[1 * 2 + 0], std::log(10.4 / 10.1) / std::sqrt(2.0), 1e-14);  // two slots
    CHECK_REL(r[1 * 2 + 1], std::log(20.4 / 20.1) / std::sqrt(2.0), 1e-14);
    CHECK(std::isnan(r[2 * 2 + 0]));                                       // off-grid row ignored
    CHECK_REL(r[3 * 2 + 0], std::log(10.5 / 10.4), 1e-14);                 // one bar, not /sqrt(2)
    CHECK_REL(r[4 * 2 + 0], std::log(10.7 / 10.6), 1e-14);                 // new session: from its open
    overnight_returns(panel, sl, gaps);
    CHECK(std::isnan(gaps[4 * 2 + 0]) && std::isnan(gaps[4 * 2 + 1]));    // late opening: no gap
}

TEST(replay_scores_only_printed_bars_and_drops_expired_options) {
    SyntheticConfig c;
    c.days = 100;  // includes the crash day (95) with its circuit breaker
    c.degraded_copy = false;
    SyntheticMarket m = generate_market(c);
    ReplayInput in;
    in.tickers = m.tickers();
    in.bars = build_panel(m.rows, in.tickers);
    in.initial_quantity = m.initial_quantity;
    in.fills = m.fills;
    in.options = m.options;
    std::string crash;
    for (const ScenarioEvent& e : m.events)
        if (e.text.rfind("market crash", 0) == 0) crash = e.timestamp.substr(0, 10);
    // make one option expire inside the backtest
    in.options[0].expiry_ymd = std::stoi(crash.substr(0, 4) + crash.substr(5, 2) + crash.substr(8, 2)) - 5;
    ReplayConfig rc;
    rc.window_days = 60;
    rc.mc_scenarios = 1000;
    rc.engine.threads = 2;
    const ReplayResult r = run_replay(in, rc);
    CHECK(r.unscored == 1);  // the 10:15 forecast for 10:30 on the halted day
    bool halted_scored = false;
    for (const ForecastRecord& f : r.forecasts)
        halted_scored = halted_scored || f.target_time == crash + " 10:30:00" || f.target_time == crash + " 10:45:00" ||
                        (f.target_time == crash + " 11:00:00" && f.target_slot != 3);
    CHECK(!halted_scored);
    // after expiry the expired option contributes nothing: book P&L = stocks + live options
    int checked = 0;
    for (const ForecastRecord& f : r.forecasts) {
        if (f.forecast_time.substr(0, 10) <= crash.substr(0, 8) + "01") continue;
        if (!std::isfinite(f.book_pnl)) continue;
        ++checked;
        CHECK(std::fabs(f.book_pnl - f.pnl) < 0.5 * std::fabs(f.book_mc[2].var) + 1e6);  // bounded, no jumps
    }
    CHECK(checked > 0);
}

TEST(numerics_edge_cases_from_review) {
    // Monte Carlo VaR is the exact order statistic
    MonteCarloRevaluation mc(10000, 5);
    JointForecast jf;
    jf.mu = {0.0};
    jf.sigma = {0.01};
    jf.corr = {1.0};
    jf.nu = 6.0;
    mc.simulate(jf);
    std::vector<Position> pos(1);
    pos[0].asset = 0;
    pos[0].quantity = 1.0;
    const McRiskResult res = mc.evaluate(pos, {1.0}, 0.0, {0.95, 0.99});
    std::vector<double> losses;
    for (double r : mc.scenarios()) losses.push_back(-(std::exp(r) - 1.0));
    std::sort(losses.begin(), losses.end(), std::greater<double>());
    CHECK(res.measures[0].var == losses[499]);  // 500th largest of 10000 (was the 501st)
    CHECK(res.measures[1].var == losses[99]);
    // NaN inputs do not become intrinsic values
    CHECK(std::isnan(bs_price(OptionType::Call, 100, 90, 0.5, kNaN, 0.05)));
    CHECK(std::isnan(bs_delta(OptionType::Put, 100, 90, kNaN, 0.3, 0.05)));
    // Cholesky rejects a non-PSD matrix with a zero pivot
    std::vector<double> L;
    CHECK(!cholesky_psd({1, 1, 0, 1, 1, 0.5, 0, 0.5, 1}, 3, L));
    CHECK(cholesky_psd({1, 1, 0, 1, 1, 0, 0, 0, 1}, 3, L));
    // Student-t for infinite / huge nu
    for (double x : {-3.0, -0.5, 1.7}) {
        CHECK(math::t_cdf(x, kInf) == math::norm_cdf(x));
        // t(nu) = Phi(x) - phi(x)(x^3 + x)/(4 nu) + O(nu^-2): checks the large-nu accuracy
        const double first_order = math::norm_cdf(x) - math::norm_pdf(x) * (x * x * x + x) / 4e9;
        CHECK_NEAR(math::t_cdf(x, 1e9), first_order, 1e-15, 0.0);
        CHECK_REL(math::std_t_cdf(x, kInf), math::norm_cdf(x), 1e-15);
    }
    CHECK_NEAR(math::t_cdf(math::t_ppf(0.01, 5e6), 5e6), 0.01, 0.0, 1e-12);
    CHECK_REL(math::t_pdf(0.3, kInf), math::norm_pdf(0.3), 1e-15);
    // Brent with a NaN starting point
    const double xm = math::brent_minimize(
        [](double v) { return v < 0.5 ? std::nan("") : (v - 0.9) * (v - 0.9); }, 0.0, 1.0);
    CHECK_NEAR(xm, 0.9, 1e-6, 0.0);
    // GARCH forecasts for persistence above 1 (explosive parameters)
    GarchFilter f(GarchParams{0.0, 0.1, 0.3, 0.71, 6.0}, 1.0);
    double h = 1.0;
    for (int k = 1; k <= 11; ++k) {
        CHECK_REL(f.forecast_variance(k), h, 1e-12);
        h = 0.1 + 1.01 * h;
    }
    // uniform stays strictly inside (0, 1)
    Rng rng(9);
    double lo = 1.0, hi = 0.0;
    for (int k = 0; k < 1000000; ++k) {
        const double u = rng.uniform();
        lo = std::min(lo, u);
        hi = std::max(hi, u);
    }
    CHECK(lo > 0.0 && hi < 1.0);
}

TEST(engine_robust_filter_survives_refit) {
    const GarchParams g{0.0, 2e-7, 0.07, 0.90, 6.0};
    const std::vector<double> y = simulate_garch(g, 3000, 301);
    auto run = [&](double z_clip, bool bad_tick) {
        EngineConfig cfg = small_config();
        cfg.diurnal = false;
        cfg.z_clip = z_clip;
        IntradayRiskEngine eng({"A"}, cfg);
        for (size_t t = 0; t < y.size(); ++t) {
            double v = y[t];
            if (bad_tick && t + 1 == y.size()) v = 50.0 * std::sqrt(g.unconditional_variance());
            eng.on_bar(static_cast<int>(t % 4), &v);
        }
        const double x[1] = {1e6};
        eng.refit(x);
        return eng.asset(0).filter.variance();
    };
    const double clean = run(6.0, false), robust = run(6.0, true), raw = run(0.0, true);
    // a 50-sigma tick capped at 6 sigma still raises h by about beta + 36 alpha (~3.4x), as the
    // live filter would; uncapped it raises it by ~alpha * 2500 (~175x)
    CHECK(robust < 6.0 * clean);
    CHECK(raw > 50.0 * clean);
}

TEST(engine_proxies_assets_without_history) {
    EngineConfig cfg = small_config();
    IntradayRiskEngine eng({"A", "B", "NEW"}, cfg);
    Rng rng(311);
    for (int d = 0; d < 300; ++d) {
        const double gaps[3] = {0.01 * rng.normal(), 0.01 * rng.normal(), kNaN};
        eng.on_overnight(gaps);
        for (int k = 0; k < 4; ++k) {
            const double f = 0.002 * rng.normal();
            const double r[3] = {f + 0.001 * rng.normal(), f + 0.003 * rng.normal(), kNaN};
            eng.on_bar(k, r);
        }
    }
    const double x[3] = {0.0, 0.0, 1e6};
    eng.refit(x);
    const AssetModel nw = eng.asset(2), a = eng.asset(0), b = eng.asset(1);
    CHECK(nw.source == ModelSource::Fallback && !nw.active);
    CHECK(nw.params.unconditional_variance() >=
          std::max(a.params.unconditional_variance(), b.params.unconditional_variance()) * 0.999);
    const PortfolioRisk risk = eng.compute(1, x);
    CHECK(risk.sigma > 1e6 * 0.002);  // far from zero risk
    const OvernightRisk on = eng.compute_overnight(x);
    CHECK(on.available && on.sigma_gap > 1e6 * 0.002);  // proxy gap risk, not zero
}

TEST(correlation_shrinkage_and_auction_bars) {
    // regular bars: strong common factor; auction bars (slots 0 and 3): idiosyncratic noise
    auto make = [](double shrink, bool auction) {
        EngineConfig cfg = small_config();
        cfg.corr_shrinkage = shrink;
        cfg.auction_correlation = auction;
        cfg.diurnal = false;
        auto eng = std::make_unique<IntradayRiskEngine>(std::vector<std::string>{"A", "B", "C"}, cfg);
        Rng rng(321);
        for (int t = 0; t < 3600; ++t) {
            const int k = t % 4;
            const double f = 0.002 * rng.normal();
            const double idio = (k == 0 || k == 3) ? 0.004 : 0.001;
            const double r[3] = {f + idio * rng.normal(), f + idio * rng.normal(), f + idio * rng.normal()};
            eng->on_bar(k, r);
        }
        const double x[3] = {1e6, 1e6, 1e6};
        eng->refit(x);
        return eng;
    };
    const auto base = make(0.0, true), shrunk = make(0.3, true), mixed = make(0.0, false);
    const std::vector<double> R0 = base->correlation(), R1 = shrunk->correlation(), Rm = mixed->correlation();
    CHECK_NEAR(R1[1], 0.7 * R0[1], 1e-9, 0.0);  // shrinkage now applies in EWMA mode
    CHECK(is_psd_corr(R1, 3));
    const std::vector<double> c = base->slot_correlation_scale();
    CHECK(c[0] < 0.5 && c[3] < 0.5 && c[1] == 1.0 && c[2] == 1.0);  // auction bars detected
    CHECK(R0[1] > Rm[1] + 0.05);  // continuous-trading correlation no longer diluted
    const double x[3] = {1e6, 1e6, 1e6};
    CHECK(base->compute(0, x).sigma < base->compute(1, x).sigma);  // less correlated at the open
    const std::vector<double> cm = mixed->slot_correlation_scale();
    for (double v : cm) CHECK(v == 1.0);
}

// ------------------------------------------------------------------ second review round

TEST(engine_fallback_proxy_is_rederived_each_refit) {
    EngineConfig cfg = small_config();
    cfg.window_bars = 1000;
    cfg.threads = 1;
    cfg.diurnal = false;
    IntradayRiskEngine eng({"A", "B", "NEW"}, cfg);
    Rng rng(7);
    for (int t = 0; t < 1000; ++t) {
        const double r[3] = {0.001 * rng.normal(), 0.001 * rng.normal(), kNaN};
        eng.on_bar(t % 4, r);
    }
    const double x[3] = {0.0, 0.0, 1e6};
    eng.refit(x);
    CHECK(eng.asset(2).source == ModelSource::Fallback);
    const double v1 = eng.asset(2).params.unconditional_variance();
    // the peers' volatility x5; NEW lists late (60 bars: too few to fit) with a 2% bar volatility
    for (int t = 0; t < 1000; ++t) {
        const double r[3] = {0.005 * rng.normal(), 0.005 * rng.normal(), t >= 940 ? 0.02 * rng.normal() : kNaN};
        eng.on_bar(t % 4, r);
    }
    const RefitReport rep = eng.refit(x);
    const AssetModel nw = eng.asset(2);
    CHECK(nw.source == ModelSource::Fallback && rep.n_fallback == 1 && rep.n_previous == 0);
    CHECK(nw.params.unconditional_variance() > 50.0 * v1);  // re-derived (own variance ~4e-4)
    CHECK(nw.params.unconditional_variance() >= eng.asset(0).params.unconditional_variance());
}

TEST(engine_halted_asset_does_not_freeze_correlations) {
    EngineConfig cfg = small_config();
    cfg.window_bars = 2000;
    cfg.threads = 1;
    cfg.diurnal = false;
    IntradayRiskEngine eng({"A", "B", "C"}, cfg);
    Rng rng(5);
    int t = 0;
    auto bar = [&](double rho, bool halted) {
        const double f = rng.normal();
        const double r[3] = {0.002 * f, 0.002 * (rho * f + std::sqrt(1.0 - rho * rho) * rng.normal()),
                             halted ? kNaN : 0.002 * rng.normal()};
        eng.on_bar(t++ % 4, r);
    };
    for (int k = 0; k < 2000; ++k) bar(0.8, false);
    const double x[3] = {1e6, 1e6, 0.0};
    eng.refit(x);
    const double before = eng.correlation()[1];
    CHECK(before > 0.7);
    // C halted (still "active": observed in most of the window) while the A-B correlation flips
    for (int k = 0; k < 400; ++k) bar(-0.6, true);
    const double live = eng.correlation()[1];
    // live updates go on with C imputed (frozen at `before` until now); with the estimated
    // lambda ~0.9987, 200 continuous-trading bars move it about a quarter of the way
    CHECK(live < before - 0.15);
    const RefitReport rep = eng.refit(x);
    CHECK(eng.asset(2).active && rep.complete_rows <= 1600);
    CHECK(eng.correlation()[1] < -0.3);  // refit: pairwise likelihood + imputed path
    CHECK(is_psd_corr(eng.correlation(), 3));
}

TEST(engine_missed_bar_variance_carried_to_forecast) {
    EngineConfig cfg = small_config();
    cfg.diurnal = false;
    cfg.threads = 1;
    IntradayRiskEngine eng({"A", "B"}, cfg);
    Rng rng(9);
    for (int t = 0; t < 2000; ++t) {
        const double r[2] = {0.001 * rng.normal(), 0.001 * rng.normal()};
        eng.on_bar(t % 4, r);
    }
    const double x[2] = {1e6, 1e6};
    eng.refit(x);
    const double r0[2] = {0.0005, -0.0005};
    eng.on_bar(0, r0);
    const double one = eng.compute(1, x).asset_sigma[0];
    const double miss[2] = {kNaN, 0.0005};
    eng.on_bar(1, miss);  // A printed at slot 0, not at slot 1: its marked P&L spans two bars
    const PortfolioRisk two = eng.compute(2, x);
    CHECK(two.asset_sigma[0] > 1.3 * one);  // ~sqrt(2): the missed bar's variance is added
    CHECK_REL(two.asset_sigma[1], std::sqrt(eng.asset(1).filter.variance()), 1e-12);  // B printed
    eng.on_bar(2, r0);  // printed again: one bar
    CHECK(eng.compute(3, x).asset_sigma[0] < 1.1 * one);
    eng.on_bar(3, miss);
    eng.on_bar(0, miss);  // new session, A not yet printed: P&L referenced to its open
    CHECK(eng.compute(1, x).asset_sigma[0] < 1.1 * one);
}

namespace {
// implied overnight correlation of A and B from 1- and 2-asset books
double implied_gap_corr(const IntradayRiskEngine& eng, int n, int a, int b) {
    std::vector<double> xa(static_cast<size_t>(n), 0.0), xb = xa, xab = xa;
    xa[static_cast<size_t>(a)] = xab[static_cast<size_t>(a)] = 1e6;
    xb[static_cast<size_t>(b)] = xab[static_cast<size_t>(b)] = 1e6;
    const double va = std::pow(eng.compute_overnight(xa.data()).sigma_gap, 2);
    const double vb = std::pow(eng.compute_overnight(xb.data()).sigma_gap, 2);
    const double vab = std::pow(eng.compute_overnight(xab.data()).sigma_gap, 2);
    return (vab - va - vb) / (2.0 * std::sqrt(va * vb));
}
}  // namespace

TEST(overnight_correlation_short_histories_and_proxies) {
    EngineConfig cfg = small_config();
    cfg.window_bars = 1200;
    cfg.threads = 1;
    cfg.diurnal = false;
    cfg.estimate_overnight_lambda = false;
    // (1) a third asset with only a few gaps must not wipe out the A-B gap correlation
    for (int new_gaps : {0, 10, 25}) {
        IntradayRiskEngine eng({"A", "B", "NEW"}, cfg);
        Rng rng(11);
        for (int d = 0; d < 300; ++d) {
            const double f = rng.normal();
            const double gaps[3] = {0.01 * (std::sqrt(0.9) * f + std::sqrt(0.1) * rng.normal()),
                                    0.01 * (std::sqrt(0.9) * f + std::sqrt(0.1) * rng.normal()),
                                    d >= 300 - new_gaps ? 0.01 * rng.normal() : kNaN};
            eng.on_overnight(gaps);
            for (int k = 0; k < 4; ++k) {
                const double r[3] = {0.001 * rng.normal(), 0.001 * rng.normal(), 0.001 * rng.normal()};
                eng.on_bar(k, r);
            }
        }
        const double x[3] = {1e6, 1e6, 0.0};
        eng.refit(x);
        CHECK(implied_gap_corr(eng, 3, 0, 1) > 0.8);  // true 0.9 (intraday correlation 0)
    }
    // (2) a proxied asset (no gaps) must keep the gap covariance PSD: no hedge illusion
    cfg.overnight_lambda = 0.999;
    IntradayRiskEngine eng({"ETF", "B", "C", "NEW"}, cfg);
    Rng rng(3);
    for (int d = 0; d < 250; ++d) {
        const double b = rng.normal(), c = rng.normal();
        const double etf = (b + c) / std::sqrt(2.0) + 0.05 * rng.normal();
        const double gaps[4] = {0.01 * etf, 0.01 * b, 0.01 * c, kNaN};
        eng.on_overnight(gaps);
        for (int k = 0; k < 4; ++k) {
            const double r[4] = {0.001 * rng.normal(), 0.001 * rng.normal(), 0.001 * rng.normal(),
                                 0.001 * rng.normal()};
            eng.on_bar(k, r);
        }
    }
    std::vector<double> x0(4, 0.0);
    eng.refit(x0.data());
    std::vector<double> C(16);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) C[static_cast<size_t>(i * 4 + j)] = i == j ? 1.0 : implied_gap_corr(eng, 4, i, j);
    for (double& v : C) v = std::round(v * 1e9) / 1e9;  // symmetric to rounding
    CHECK(is_psd_corr(C, 4));
    std::vector<double> sd(4);
    for (int i = 0; i < 4; ++i) {
        std::vector<double> e(4, 0.0);
        e[static_cast<size_t>(i)] = 1.0;
        sd[static_cast<size_t>(i)] = eng.compute_overnight(e.data()).sigma_gap;
    }
    // long ETF, short B and C, a small long NEW: positive gap risk (was clamped to 0)
    const double y[4] = {std::sqrt(2.0) / sd[0], -1.0 / sd[1], -1.0 / sd[2], 0.3 / sd[3]};
    CHECK(eng.compute_overnight(y).sigma_gap > 0.2);
}

TEST(student_t_far_tails_and_large_nu) {
    // references: regularized incomplete beta in mpmath (40+ digits), see tests/reference
    struct Ref {
        double nu, x, F;
    };
    const Ref refs[] = {
#include "reference/t_cdf_reference.inc"
    };
    for (const Ref& r : refs) CHECK_REL(math::t_cdf(r.x, r.nu), r.F, 5e-14);
    // quantile round trips, far tails included (Newton on ln F: no crawling in the tail)
    for (double nu : {2.5, 5.0, 30.0, 45.0, 300.0, 5000.0, 1e6, 5e6})
        for (double p : {1e-300, 1e-86, 1e-20, 1e-6, 0.01, 0.2, 0.4999}) {
            const double q = math::t_ppf(p, nu);
            CHECK_REL(math::t_cdf(q, nu), p, 1e-12);
        }
    CHECK(math::t_cdf(1e200, 5.0) == 1.0);
    const double far = math::t_cdf(-1e200, 0.5);  // power-law tail ~3.21e-101
    CHECK(far > 3.1e-101 && far < 3.3e-101);
    CHECK(math::chi2_sf(kInf, 1.0) == 0.0);
}

TEST(per_asset_diurnal_zero_slot_takes_common_shape) {
    const size_t T = 8 * 60, n = 4;
    std::vector<double> r(T * n);
    std::vector<int> slot(T);
    Rng rng(17);
    for (size_t t = 0; t < T; ++t) {
        slot[t] = static_cast<int>(t % 8);
        for (size_t i = 0; i < n; ++i) r[t * n + i] = (i == 0 && slot[t] == 0) ? 0.0 : 0.001 * rng.normal();
    }
    const std::vector<DiurnalProfile> prof = DiurnalProfile::estimate_per_asset(r.data(), slot.data(), T, n, 8);
    CHECK(prof[0][0] > 0.5);  // no 1e-6 factor (a later 0.2% move would deseasonalize to 2000)
    bool threw = false;
    try {
        DiurnalProfile::estimate(r.data(), slot.data(), T, -1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

TEST(csv_and_input_edge_cases_from_review) {
    const CsvTable t = parse_csv("a;b\n1;2\n;;\n3;4\n");
    CHECK(t.rows.size() == 2);  // a line of delimiters is blank
    const CsvTable one = parse_csv("ret\n0,0012\n-1,5\n");  // one column, decimal commas
    double v = 0.0;
    CHECK(one.rows.size() == 2 && one.rows[0].size() == 1 && parse_number(one.rows[0][0], v) && v == 0.0012);
    TimePoint tp;
    CHECK(parse_time("2026-10-09 13:15:00-03", tp) && tp.ymd == 20261009 && tp.minute == 13 * 60 + 15);
    CHECK(parse_time("2026-10-09t16:15:00Z", tp) && tp.minute == 13 * 60 + 15);
    CHECK(is_b3_holiday(20260216) && is_b3_holiday(20271115) && !is_b3_holiday(20260218));
    const std::string path = "garcht_test_fills.csv";
    {
        std::ofstream f(path);
        f << "timestamp,ticker,side,quantity\n2026-03-02 10:20:00,AAA,SHORT,100\n";
    }
    bool threw = false;
    try {
        load_fills_csv(path);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);  // an unknown side is not guessed
    {
        std::ofstream f(path);
        f << "timestamp,ticker,side,quantity\n2026-03-02 10:20:00,AAA,V,100\n2026-03-02 10:21:00,AAA,,-5\n";
    }
    const std::vector<Fill> fl = load_fills_csv(path);
    CHECK(fl.size() == 2 && fl[0].quantity == -100.0 && fl[1].quantity == -5.0);
    std::remove(path.c_str());
    // Acerbi-Szekely without simulations: statistics only, no fake p-values
    std::vector<double> xs(200, 0.0);
    std::vector<DistForecast> fs(200);
    for (auto& f : fs) {
        f.mean = 0.0;
        f.sigma = 1.0;
        f.shock = ShockDist::student(6.0);
    }
    const EsTest es = acerbi_szekely_test(xs, fs, 0.975, 0, 1);
    CHECK(std::isfinite(es.z2) && std::isnan(es.p2));
}

// ================================================================== live feed

TEST(live_feed_reproduces_the_file_replay) {
    // The same market replayed from a whole file and from a feed, bar by bar after a history: the
    // forecasts, overnight forecasts and daily refits must be identical, including a session the
    // feed cuts 6 bars early (its end is only learnt when the next session starts).
    SyntheticConfig c;
    c.days = 45;
    c.degraded_copy = false;
    const SyntheticMarket m = generate_market(c);
    std::vector<BarRow> rows = m.rows;
    std::stable_sort(rows.begin(), rows.end(), [](const BarRow& a, const BarRow& b) { return a.time.key() < b.time.key(); });
    std::vector<int> days;
    for (const BarRow& r : rows)
        if (days.empty() || r.time.ymd != days.back()) days.push_back(r.time.ymd);
    const int live_from = days[30], cut_day = days[36];
    int cut_minute = 0;
    for (const BarRow& r : rows)
        if (r.time.ymd == cut_day) cut_minute = std::max(cut_minute, r.time.minute);
    cut_minute -= 6 * 15;
    rows.erase(std::remove_if(rows.begin(), rows.end(),
                              [&](const BarRow& r) { return r.time.ymd == cut_day && r.time.minute > cut_minute; }),
               rows.end());

    ReplayInput full;
    full.tickers = m.tickers();
    full.bars = build_panel(rows, full.tickers);
    full.initial_quantity = m.initial_quantity;
    full.fills = m.fills;
    ReplayConfig rc;
    rc.window_days = 20;
    rc.engine.threads = 2;
    const ReplayResult batch = run_replay(full, rc);

    ReplayInput history = full;
    std::vector<BarRow> past;
    for (const BarRow& r : rows)
        if (r.time.ymd < live_from) past.push_back(r);
    history.bars = build_panel(past, history.tickers);
    std::vector<std::string> lines = {"timestamp,ticker,open,high,low,close,volume"};
    char buf[256];
    std::string clock;
    for (const BarRow& r : rows) {
        if (r.time.ymd < live_from) continue;
        const std::string ts = r.time.str();
        if (!clock.empty() && ts != clock) lines.push_back("#clock " + clock);
        clock = ts;
        std::snprintf(buf, sizeof buf, "%s,%s,%.17g,%.17g,%.17g,%.17g,%.17g", ts.c_str(), r.ticker.c_str(), r.open,
                      r.high, r.low, r.close, r.volume);
        lines.push_back(buf);
    }
    lines.push_back("#clock " + clock);
    lines.push_back("#end");
    size_t next = 0;
    LiveStats st;
    size_t live_bars = 0;
    const ReplayResult live = run_live(history, rc, [&](std::string& l) {
        if (next >= lines.size()) return false;
        l = lines[next++];
        return true;
    }, st, [&](const BarPanel&, size_t, const ForecastRecord* f) { live_bars += f != nullptr; });

    CHECK(st.late == 0 && st.duplicates == 0 && st.bad == 0 && st.unknown == 0);
    CHECK(live_bars > 0 && st.bars == live_bars);  // every live bar is on the grid and issues a forecast
    CHECK(batch.forecasts.size() == live.forecasts.size() && batch.forecasts.size() > 200);
    bool same = batch.forecasts.size() == live.forecasts.size();
    for (size_t i = 0; same && i < batch.forecasts.size(); ++i) {
        const ForecastRecord &a = batch.forecasts[i], &b = live.forecasts[i];
        same = a.forecast_time == b.forecast_time && a.target_time == b.target_time && a.pnl == b.pnl &&
               a.models[0].sigma == b.models[0].sigma && a.models[0].measures[2].var == b.models[0].measures[2].var && a.models[1].measures[2].var == b.models[1].measures[2].var;
    }
    CHECK(same);
    bool same_on = batch.overnight.size() == live.overnight.size();
    for (size_t i = 0; same_on && i < batch.overnight.size(); ++i)
        same_on = batch.overnight[i].forecast_time == live.overnight[i].forecast_time &&
                  batch.overnight[i].risk.sigma == live.overnight[i].risk.sigma && batch.overnight[i].pnl == live.overnight[i].pnl;
    CHECK(same_on);
    CHECK(batch.daily.size() == live.daily.size());
    CHECK(batch.unscored == live.unscored && batch.sessions == live.sessions);

    // A row that arrives after the feed's clock passed its bar is late: dropped and counted.
    std::vector<std::string> late_lines = lines;
    const auto first_clock = std::find_if(late_lines.begin(), late_lines.end(), [](const std::string& l) { return l.rfind("#clock", 0) == 0; });
    const std::string moved = *(first_clock - 1);
    late_lines.erase(first_clock - 1);
    late_lines.insert(std::find_if(late_lines.begin(), late_lines.end(), [](const std::string& l) { return l.rfind("#clock", 0) == 0; }) + 1, moved);
    late_lines.insert(late_lines.begin() + 5, late_lines[4]);  // and a duplicated row
    next = 0;
    LiveStats st2;
    run_live(history, rc, [&](std::string& l) {
        if (next >= late_lines.size()) return false;
        l = late_lines[next++];
        return true;
    }, st2);
    CHECK(st2.late == 1 && st2.duplicates == 1);

    // With the jump filter on, the live mode filters bar by bar as the file replay does.
    rc.jump_filter = true;
    const ReplayResult batch_f = run_replay(full, rc);
    next = 0;
    LiveStats st3;
    const ReplayResult live_f = run_live(history, rc, [&](std::string& l) {
        if (next >= lines.size()) return false;
        l = lines[next++];
        return true;
    }, st3);
    bool same_f = batch_f.forecasts.size() == live_f.forecasts.size() && batch_f.jump_events.size() == live_f.jump_events.size();
    for (size_t i = 0; same_f && i < batch_f.forecasts.size(); ++i)
        same_f = batch_f.forecasts[i].pnl == live_f.forecasts[i].pnl && batch_f.forecasts[i].models[0].measures[2].var == live_f.forecasts[i].models[0].measures[2].var;
    CHECK(same_f);

    // A feed line is split as the file reader splits a CSV: a spreadsheet export (BOM, ';', quoted
    // fields, decimal commas) is read the same way live.
    char delim = 0;
    const std::vector<std::string> head = split_feed_line("\xEF\xBB\xBFticker;timestamp;close", delim);
    CHECK(delim == ';' && head.size() == 3 && head[0] == "ticker" && head[1] == "timestamp");
    const std::vector<std::string> row = split_feed_line("\"AAA\";2026-01-02 10:15:00;\"38,50\"", delim);
    double px = 0.0;
    CHECK(row.size() == 3 && row[0] == "AAA" && parse_number(row[2], px) && px == 38.5);
}

TEST(jump_filter_bad_tick_genuine_shock_and_market_move) {
    // 5 stocks, 12 sessions of 28 bars of small moves; in session 10: a bad tick on stock 0 (slot 8,
    // +8%, back at the next bar), a genuine -8% level shift on stock 1 (slot 14), a -5% move of
    // every stock together (slot 20) and a genuine +8% move of stock 3 at the session's last bar;
    // in session 11, a bad tick on stock 2 at the session's first bar.
    const size_t n = 5;
    std::vector<BarRow> rows;
    std::vector<double> px(n, 50.0);
    unsigned s = 12345;
    auto noise = [&]() {
        s = s * 1103515245u + 12345u;
        return (static_cast<double>((s >> 8) & 0xFFFF) / 65535.0 - 0.5) * 0.004;  // +-0.2%
    };
    const char* names[] = {"AAA", "BBB", "CCC", "DDD", "EEE"};
    for (int d = 0; d < 12; ++d)
        for (int k = 0; k < 28; ++k)
            for (size_t i = 0; i < n; ++i) {
                double close = px[i] * std::exp(noise());
                if (d == 10 && k == 14 && i == 1) close *= 0.92;
                if (d == 10 && k == 20) close *= 0.95;
                if (d == 10 && k == 27 && i == 3) close *= 1.08;
                BarRow r;
                r.time = {20261005 + d, 615 + 15 * k, 0};
                r.ticker = names[i];
                r.open = px[i];
                r.close = close;
                r.high = std::max(r.open, close);
                r.low = std::min(r.open, close);
                r.volume = 1000;
                px[i] = close;
                if (d == 10 && k == 8 && i == 0) r.close *= 1.08;  // the print is wrong, the price is not
                if (d == 11 && k == 0 && i == 2) r.close *= 1.08;
                rows.push_back(r);
            }
    const std::vector<std::string> tickers(names, names + n);
    const BarPanel raw = build_panel(rows, tickers);
    const SessionSpec session = detect_session(raw);
    BarPanel p = raw;
    JumpFilter filter(n, session);
    for (size_t t = 0; t < p.num_bars(); ++t) filter.apply(p, t);
    const size_t bar = 10 * 28;
    CHECK(std::isnan(p.close[(bar + 8) * n + 0]));                      // bad tick held ...
    CHECK(p.close[(bar + 9) * n + 0] == raw.close[(bar + 9) * n + 0]);  // ... and the next bar kept
    CHECK(std::isnan(p.close[(bar + 14) * n + 1]));                     // genuine shift held one bar,
    CHECK(p.close[(bar + 15) * n + 1] == raw.close[(bar + 15) * n + 1]);  // then enters
    for (size_t i = 0; i < n; ++i) CHECK(p.close[(bar + 20) * n + i] == raw.close[(bar + 20) * n + i]);  // market move kept
    CHECK(std::isnan(p.close[(bar + 27) * n + 3]));                     // held at the last bar: undecided
    CHECK(std::isnan(p.close[(bar + 28) * n + 2]));                     // bad tick at a first bar held ...
    CHECK(p.close[(bar + 29) * n + 2] == raw.close[(bar + 29) * n + 2]);  // ... and decided at the next
    int held = 0, bad = 0, accepted = 0, market = 0, undecided = 0;
    for (const JumpEvent& e : filter.events()) {
        held += e.action == "HELD";
        bad += e.action == "BAD_TICK" && (e.ticker == "AAA" || e.ticker == "CCC");
        accepted += e.action == "SHOCK_ACCEPTED" && e.ticker == "BBB";
        market += e.action == "MARKET_MOVE";
        undecided += e.action == "UNDECIDED" && e.ticker == "DDD";
    }
    CHECK(held == 4 && bad == 2 && accepted == 1 && market == 5 && undecided == 1);
    CHECK(filter.events().size() == 13);
    // causal: a different future (every later price moved) leaves every decision up to the cut as it was
    size_t same = 0, total = 0;
    for (size_t cut : {bar + 8, bar + 14, bar + 20}) {
        BarPanel other = raw;
        for (size_t k = (cut + 1) * n; k < other.close.size(); ++k) {
            other.close[k] *= 1.3;
            other.open[k] *= 0.8;
        }
        JumpFilter f2(n, session);
        for (size_t t = 0; t < other.num_bars(); ++t) f2.apply(other, t);
        for (size_t k = 0; k < (cut + 1) * n; ++k, ++total) {
            const double a = other.close[k], b = p.close[k];
            same += (std::isnan(a) && std::isnan(b)) || a == b;
        }
    }
    CHECK(same == total);
    // an accepted jump enters the model at full size: the whole move since the last good close
    std::vector<double> ret;
    std::vector<int> slots;
    bar_returns(p, session, ret, slots);
    const double spread = ret[(bar + 15) * n + 1];  // across the held close: scaled by 1/sqrt(2)
    filter.restore_accepted(ret);
    const double full = std::log(raw.close[(bar + 15) * n + 1] / raw.close[(bar + 13) * n + 1]);
    CHECK(std::fabs(ret[(bar + 15) * n + 1] - full) < 1e-15);
    CHECK(std::fabs(spread * std::sqrt(2.0) - full) < 1e-12);

    // The filter shapes the model's view only: with it on and off, the replay scores the same
    // forecasts on the same P&L (the prices as printed), while the bad tick moves the VaR off.
    ReplayInput in;
    in.tickers = tickers;
    in.bars = raw;
    in.initial_quantity.assign(n, 100.0);
    ReplayConfig rc;
    rc.window_days = 5;
    rc.overnight = false;
    const ReplayResult off = run_replay(in, rc);
    rc.jump_filter = true;
    const ReplayResult on = run_replay(in, rc);
    CHECK(off.forecasts.size() == on.forecasts.size() && !on.forecasts.empty());
    bool same_pnl = off.forecasts.size() == on.forecasts.size();
    size_t var_differs = 0;
    for (size_t i = 0; same_pnl && i < on.forecasts.size(); ++i) {
        same_pnl = off.forecasts[i].target_time == on.forecasts[i].target_time && off.forecasts[i].pnl == on.forecasts[i].pnl;
        var_differs += off.forecasts[i].models[0].measures[0].var != on.forecasts[i].models[0].measures[0].var;
    }
    CHECK(same_pnl);
    CHECK(var_differs > 0);
    CHECK(on.jump_events.size() == filter.events().size());

    // Regression: with the filter on, the two bad ticks must not damage the risk estimate (their
    // printed P&L must not reach the tail calibration either). The same market without them gives
    // VaRs within 5% at all but the bars of the ticks (marked at the printed price); with the filter
    // off the ticks move the VaR more than 5% for many bars.
    ReplayInput clean_in = in;
    clean_in.bars.close[(bar + 8) * n + 0] /= 1.08;
    clean_in.bars.close[(bar + 28) * n + 2] /= 1.08;
    rc.jump_filter = true;
    const ReplayResult clean_on = run_replay(clean_in, rc);
    rc.jump_filter = false;
    const ReplayResult clean_off = run_replay(clean_in, rc);
    auto damaged = [](const ReplayResult& a, const ReplayResult& b) {
        size_t k = 0;
        for (size_t i = 0; i < a.forecasts.size() && i < b.forecasts.size(); ++i)
            k += std::fabs(a.forecasts[i].models[0].measures[2].var / b.forecasts[i].models[0].measures[2].var - 1.0) > 0.05;
        return k;
    };
    CHECK(clean_on.forecasts.size() == on.forecasts.size());
    CHECK(damaged(on, clean_on) <= 2);
    CHECK(damaged(off, clean_off) >= 5);
}

// ------------------------------------------------------------------ runner
int main(int argc, char** argv) {
    const std::string filter = argc > 1 ? argv[1] : "";
    int run = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (const TestCase& tc : registry()) {
        if (!filter.empty() && std::string(tc.name).find(filter) == std::string::npos) continue;
        const int before = g_failures;
        const auto s = std::chrono::steady_clock::now();
        tc.fn();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s).count();
        ++run;
        const bool ok = g_failures == before;
        if (!ok) ++g_test_failures;
        std::printf("[%s] %-58s %8.1f ms\n", ok ? " OK " : "FAIL", tc.name, ms);
    }
    const double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\n%d tests, %d checks, %d failed checks, %d failed tests (%.2f s)\n", run, g_checks,
                g_failures, g_test_failures, total);
    return g_failures == 0 ? 0 : 1;
}
