// Special functions and probability distributions used by the GARCH-t engine.
//
// Everything here is self-contained (no external dependencies), thread-safe
// (no global state such as `signgam`) and accurate to ~1e-13 relative error in the ranges
// used by the risk engine (Student-t CDF/quantile: a few 1e-14 for every nu; binomial CDF
// for n ~ 1e5: ~1e-10, ample for the traffic light).
#pragma once

#include <cstddef>

namespace garcht::math {

inline constexpr double kPi = 3.14159265358979323846264338327950288;
inline constexpr double kLn2Pi = 1.83787706640934548356065947281123527;  // ln(2*pi)
inline constexpr double kInvSqrt2Pi = 0.39894228040143267793994605993438187;

// ---------------------------------------------------------------- gamma family
double log_gamma(double x);  // ln Gamma(x), x > 0 (Lanczos, thread-safe)
double digamma(double x);    // psi(x) = d/dx ln Gamma(x), x > 0

// Regularized incomplete beta I_x(a, b), a, b > 0, x in [0, 1].
double inc_beta(double a, double b, double x);

// Regularized upper incomplete gamma Q(a, x), a > 0, x >= 0.
double inc_gamma_upper(double a, double x);

// ---------------------------------------------------------------- Normal
double norm_pdf(double x);
double norm_cdf(double x);
double norm_ppf(double p);  // inverse CDF, p in (0, 1)

// ---------------------------------------------------------------- Student-t
// "Classic" Student-t with nu degrees of freedom (variance nu / (nu - 2)).
double t_pdf(double x, double nu);
double t_log_pdf(double x, double nu);
double t_cdf(double x, double nu);
double t_ppf(double p, double nu);  // inverse CDF, p in (0, 1), nu > 0

// Standardized Student-t: unit variance, requires nu > 2.
//   Z = T * sqrt((nu - 2) / nu),  T ~ t(nu)
double std_t_pdf(double z, double nu);
double std_t_log_pdf(double z, double nu);
double std_t_cdf(double z, double nu);
double std_t_ppf(double p, double nu);

// ---------------------------------------------------------------- chi-square
double chi2_sf(double x, double df);  // survival function P(X > x)

// ---------------------------------------------------------------- 1-D optimizer
// Brent's method: minimizes f on [a, b]. Returns the argmin.
template <class F>
double brent_minimize(F&& f, double a, double b, double tol = 1e-10, int max_iter = 200);

}  // namespace garcht::math

// ---------------------------------------------------------------- template impl
#include <cmath>

namespace garcht::math {

template <class F>
double brent_minimize(F&& f, double a, double b, double tol, int max_iter) {
    const double golden = 0.3819660112501051;
    double x = a + golden * (b - a), w = x, v = x;
    double fx = f(x);
    if (!std::isfinite(fx)) fx = HUGE_VAL;  // a NaN start would block every comparison
    double fw = fx, fv = fx;
    double d = 0.0, e = 0.0;
    for (int it = 0; it < max_iter; ++it) {
        const double m = 0.5 * (a + b);
        const double tol1 = tol * std::fabs(x) + 1e-12;
        const double tol2 = 2.0 * tol1;
        if (std::fabs(x - m) <= tol2 - 0.5 * (b - a)) break;
        bool golden_step = true;
        if (std::fabs(e) > tol1) {  // try parabolic interpolation
            double r = (x - w) * (fx - fv);
            double q = (x - v) * (fx - fw);
            double p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if (q > 0.0) p = -p;
            q = std::fabs(q);
            const double e_prev = e;
            e = d;
            if (std::fabs(p) < std::fabs(0.5 * q * e_prev) && p > q * (a - x) && p < q * (b - x)) {
                d = p / q;
                const double u = x + d;
                if (u - a < tol2 || b - u < tol2) d = (x < m) ? tol1 : -tol1;
                golden_step = false;
            }
        }
        if (golden_step) {
            e = (x < m) ? b - x : a - x;
            d = golden * e;
        }
        const double u = (std::fabs(d) >= tol1) ? x + d : x + (d > 0 ? tol1 : -tol1);
        double fu = f(u);
        if (!std::isfinite(fu)) fu = HUGE_VAL;
        if (fu <= fx) {
            (u < x ? b : a) = x;
            v = w; fv = fw;
            w = x; fw = fx;
            x = u; fx = fu;
        } else {
            (u < x ? a : b) = u;
            if (fu <= fw || w == x) {
                v = w; fv = fw;
                w = u; fw = fu;
            } else if (fu <= fv || v == x || v == w) {
                v = u; fv = fu;
            }
        }
    }
    return x;
}

}  // namespace garcht::math
