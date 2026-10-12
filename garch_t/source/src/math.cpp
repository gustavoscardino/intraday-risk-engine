#include "garcht/math.hpp"

#include <array>
#include <cmath>
#include <limits>

namespace garcht::math {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kEps = 1e-15;
constexpr double kFpMin = 1e-300;

// Continued fraction for the incomplete beta (modified Lentz).
double beta_cf(double a, double b, double x) {
    const double qab = a + b, qap = a + 1.0, qam = a - 1.0;
    double c = 1.0;
    double d = 1.0 - qab * x / qap;
    if (std::fabs(d) < kFpMin) d = kFpMin;
    d = 1.0 / d;
    double h = d;
    for (int m = 1; m <= 20000; ++m) {
        const double m2 = 2.0 * m;
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if (std::fabs(d) < kFpMin) d = kFpMin;
        c = 1.0 + aa / c;
        if (std::fabs(c) < kFpMin) c = kFpMin;
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (std::fabs(d) < kFpMin) d = kFpMin;
        c = 1.0 + aa / c;
        if (std::fabs(c) < kFpMin) c = kFpMin;
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        if (std::fabs(del - 1.0) < kEps) break;
    }
    return h;
}

// Lower-tail Normal quantile for p in (0, 0.5]: Wichura's AS241 (PPND16).
double norm_ppf_lower(double p) {
    const double q = p - 0.5;
    double x;
    if (std::fabs(q) <= 0.425) {
        const double r = 0.180625 - q * q;
        x = q *
            (((((((2.5090809287301226727e+3 * r + 3.3430575583588128105e+4) * r +
                  6.7265770927008700853e+4) * r + 4.5921953931549871457e+4) * r +
                1.3731693765509461125e+4) * r + 1.9715909503065514427e+3) * r +
              1.3314166789178437745e+2) * r + 3.3871328727963666080e+0) /
            (((((((5.2264952788528545610e+3 * r + 2.8729085735721942674e+4) * r +
                  3.9307895800092710610e+4) * r + 2.1213794301586595867e+4) * r +
                5.3941960214247511077e+3) * r + 6.8718700749205790830e+2) * r +
              4.2313330701600911252e+1) * r + 1.0);
    } else {
        double r = std::sqrt(-std::log(p));
        if (r <= 5.0) {
            r -= 1.6;
            x = (((((((7.74545014278341407640e-4 * r + 2.27238449892691845833e-2) * r +
                      2.41780725177450611770e-1) * r + 1.27045825245236838258e+0) * r +
                    3.64784832476320460504e+0) * r + 5.76949722146069140550e+0) * r +
                  4.63033784615654529590e+0) * r + 1.42343711074968357734e+0) /
                (((((((1.05075007164441684324e-9 * r + 5.47593808499534494600e-4) * r +
                      1.51986665636164571966e-2) * r + 1.48103976427480074590e-1) * r +
                    6.89767334985100004550e-1) * r + 1.67638483018380384940e+0) * r +
                  2.05319162663775882187e+0) * r + 1.0);
        } else {
            r -= 5.0;
            x = (((((((2.01033439929228813265e-7 * r + 2.71155556874348757815e-5) * r +
                      1.24266094738807843860e-3) * r + 2.65321895265761230930e-2) * r +
                    2.96560571828504891230e-1) * r + 1.78482653991729133580e+0) * r +
                  5.46378491116411436990e+0) * r + 6.65790464350110377720e+0) /
                (((((((2.04426310338993978564e-15 * r + 1.42151175831644588870e-7) * r +
                      1.84631831751005468180e-5) * r + 7.86869131145613259100e-4) * r +
                    1.48753612908506148525e-2) * r + 1.36929880922735805310e-1) * r +
                  5.99832206555887937690e-1) * r + 1.0);
        }
        x = -x;
    }
    // One Halley refinement step against the erfc-based CDF (accurate in the lower tail).
    const double e = norm_cdf(x) - p;
    const double u = e / norm_pdf(x);
    if (std::isfinite(u)) x -= u / (1.0 + 0.5 * x * u);
    return x;
}

// ln Gamma(a + 1/2) - ln Gamma(a) - ln(a)/2 to full precision for all a > 0 (the direct
// difference of two large log-gammas would lose ~log10(a) digits). Recurrence
// rem(a) = rem(a + 1) + ln(a (a + 1) / (a + 1/2)^2) / 2 up to a >= 16, then the asymptotic series
// sum_k (2^(1-k) - 2) B_k / (k (k - 1) a^(k-1)) over even k (next term < 1e-19 at a = 16).
double log_gamma_half_ratio_rem(double a) {
    if (a < 1.0) return log_gamma(a + 0.5) - log_gamma(a) - 0.5 * std::log(a);
    double acc = 0.0;
    for (; a < 16.0; a += 1.0) acc += 0.5 * std::log1p(-0.25 / ((a + 0.5) * (a + 0.5)));
    const double r = 1.0 / a, r2 = r * r;
    return acc +
           r * (-1.0 / 8.0 +
                r2 * (1.0 / 192.0 +
                      r2 * (-1.0 / 640.0 +
                            r2 * (17.0 / 14336.0 +
                                  r2 * (-31.0 / 18432.0 +
                                        r2 * (691.0 / 180224.0 +
                                              r2 * (-5461.0 / 425984.0 + r2 * (929569.0 / 15728640.0))))))));
}

// ln Gamma(a + 1/2) - ln Gamma(a)
double log_gamma_half_ratio(double a) { return 0.5 * std::log(a) + log_gamma_half_ratio_rem(a); }

double t_log_norm_const(double nu) {  // ln of t(nu) density normalizing constant
    return log_gamma_half_ratio(0.5 * nu) - 0.5 * std::log(nu * kPi);
}

// ln B(a, b) with an accurate path for b = 1/2 (Student-t incomplete betas).
double log_beta(double a, double b) {
    if (b == 0.5) return log_gamma(0.5) - log_gamma_half_ratio(a);
    if (a == 0.5) return log_gamma(0.5) - log_gamma_half_ratio(b);
    return log_gamma(a) + log_gamma(b) - log_gamma(a + b);
}

// Student-t lower tail P(T <= -sqrt(x2)) = I_z(nu/2, 1/2) / 2, z = nu / (nu + x2), for nu >= 30 and
// z >= 0.7: DiDonato & Morris (1992, ACM TOMS 708) BGRAT expansion for I_z(a, b) with a large and
// b <= 1, written with J_n scaled by h = u^b e^-u / Gamma(b) so nothing overflows in the far tail.
// The continued fraction is in its slowly converging regime here and its round-off grows like
// nu * eps (1e-9 relative at nu = 1e7); this series is accurate to a few eps for every nu.
double t_tail_large_nu(double nu, double x2) {
    constexpr double b = 0.5;
    constexpr int kTerms = 30;
    static const std::array<double, kTerms + 1> fact_odd = [] {  // (2m + 1)!
        std::array<double, kTerms + 1> f{};
        f[0] = 1.0;
        for (int m = 1; m <= kTerms; ++m) f[m] = f[m - 1] * (2.0 * m) * (2.0 * m + 1.0);
        return f;
    }();
    const double a = 0.5 * nu, t = a - 0.25;               // t = a + (b - 1) / 2
    const double lx = std::log1p(-x2 / (nu + x2));          // ln z, z >= 0.7
    const double u = -t * lx;
    const double su = std::sqrt(u);
    const double h = su * std::exp(-u) / std::sqrt(kPi);    // u^b e^-u / Gamma(b)
    // Gamma(a + b) / (Gamma(a) t^b)
    const double pref = std::exp(log_gamma_half_ratio_rem(a) - 0.5 * std::log1p(-0.25 / a));
    double p[kTerms];
    p[0] = 1.0;
    double J = std::erfc(su);  // h * J_0 = Q(1/2, u)
    double sum = J;
    const double lx2 = 0.25 * lx * lx, t4 = 4.0 * t * t;
    double lxp = 1.0, b2n = b;
    for (int n = 1; n < kTerms; ++n) {
        double pn = 0.0;
        for (int m = 1; m < n; ++m) pn += (m * b - n) * p[n - m] / fact_odd[m];
        p[n] = pn / n + (b - 1.0) / fact_odd[n];
        J = (b2n * (b2n + 1.0) * J + h * (u + b2n + 1.0) * lxp) / t4;
        lxp *= lx2;
        b2n += 2.0;
        const double r = p[n] * J;
        sum += r;
        if (std::fabs(r) <= 1e-17 * std::fabs(sum)) break;
    }
    return 0.5 * pref * sum;
}

}  // namespace

// ============================================================== gamma family

double log_gamma(double x) {
    if (!(x > 0.0)) {
        if (x == 0.0) return std::numeric_limits<double>::infinity();
        if (std::isnan(x)) return kNaN;
        // Reflection formula for negative non-integers.
        const double s = std::sin(kPi * x);
        if (s == 0.0) return std::numeric_limits<double>::infinity();
        return std::log(kPi / std::fabs(s)) - log_gamma(1.0 - x);
    }
    if (x < 0.5) return std::log(kPi / std::sin(kPi * x)) - log_gamma(1.0 - x);
    if (x > 15.0) {
        // Stirling series: very accurate for large arguments.
        const double inv = 1.0 / x, inv2 = inv * inv;
        const double series =
            inv * (1.0 / 12.0 -
                   inv2 * (1.0 / 360.0 -
                           inv2 * (1.0 / 1260.0 - inv2 * (1.0 / 1680.0 - inv2 * (1.0 / 1188.0)))));
        return (x - 0.5) * std::log(x) - x + 0.5 * kLn2Pi + series;
    }
    // Lanczos approximation, g = 7, n = 9.
    static constexpr double c[9] = {0.99999999999980993,     676.5203681218851,
                                    -1259.1392167224028,     771.32342877765313,
                                    -176.61502916214059,     12.507343278686905,
                                    -0.13857109526572012,    9.9843695780195716e-6,
                                    1.5056327351493116e-7};
    const double xm1 = x - 1.0;
    double a = c[0];
    const double t = xm1 + 7.5;
    for (int i = 1; i < 9; ++i) a += c[i] / (xm1 + i);
    return 0.5 * kLn2Pi + (xm1 + 0.5) * std::log(t) - t + std::log(a);
}

double digamma(double x) {
    if (!(x > 0.0)) return kNaN;
    double result = 0.0;
    while (x < 10.0) {  // recurrence psi(x) = psi(x + 1) - 1/x
        result -= 1.0 / x;
        x += 1.0;
    }
    const double inv = 1.0 / x, inv2 = inv * inv;
    result += std::log(x) - 0.5 * inv -
              inv2 * (1.0 / 12.0 -
                      inv2 * (1.0 / 120.0 -
                              inv2 * (1.0 / 252.0 -
                                      inv2 * (1.0 / 240.0 -
                                              inv2 * (1.0 / 132.0 - inv2 * (691.0 / 32760.0))))));
    return result;
}

// I_x(a, b) given both x and its complement xc = 1 - x computed independently: when x is
// within ~1e-8 of 1, forming 1 - x by subtraction would lose half the digits.
static double inc_beta_xc(double a, double b, double x, double xc) {
    if (!(a > 0.0) || !(b > 0.0) || std::isnan(x) || std::isnan(xc)) return kNaN;
    if (x <= 0.0) return 0.0;
    if (xc <= 0.0) return 1.0;
    // ln x and ln(1 - x), each from whichever of x, xc is small (log1p for the one near 1):
    // with a or b ~ 1e9 an absolute error of 1e-16 in a logarithm would cost 8 digits
    const double lx = x < 0.5 ? std::log(x) : std::log1p(-xc);
    const double lxc = xc < 0.5 ? std::log(xc) : std::log1p(-x);
    const double log_front = -log_beta(a, b) + a * lx + b * lxc;
    if (x < (a + 1.0) / (a + b + 2.0)) return std::exp(log_front) * beta_cf(a, b, x) / a;
    return 1.0 - std::exp(log_front) * beta_cf(b, a, xc) / b;
}

double inc_beta(double a, double b, double x) {
    if (!(a > 0.0) || !(b > 0.0) || std::isnan(x)) return kNaN;
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;
    const double log_front = -log_beta(a, b) + a * std::log(x) + b * std::log1p(-x);
    if (x < (a + 1.0) / (a + b + 2.0)) return std::exp(log_front) * beta_cf(a, b, x) / a;
    return 1.0 - std::exp(log_front) * beta_cf(b, a, 1.0 - x) / b;
}

double inc_gamma_upper(double a, double x) {
    if (!(a > 0.0) || std::isnan(x)) return kNaN;
    if (x <= 0.0) return 1.0;
    if (std::isinf(x)) return 0.0;
    const double log_front = -x + a * std::log(x) - log_gamma(a);
    if (x < a + 1.0) {  // series for P(a, x), then Q = 1 - P
        double ap = a, sum = 1.0 / a, del = sum;
        for (int n = 0; n < 100000; ++n) {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if (std::fabs(del) < std::fabs(sum) * kEps) break;
        }
        return 1.0 - sum * std::exp(log_front);
    }
    // Continued fraction for Q(a, x).
    double b = x + 1.0 - a, c = 1.0 / kFpMin, d = 1.0 / b, h = d;
    for (int i = 1; i <= 100000; ++i) {
        const double an = -i * (i - a);
        b += 2.0;
        d = an * d + b;
        if (std::fabs(d) < kFpMin) d = kFpMin;
        c = b + an / c;
        if (std::fabs(c) < kFpMin) c = kFpMin;
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        if (std::fabs(del - 1.0) < kEps) break;
    }
    return std::exp(log_front) * h;
}

// ============================================================== Normal

double norm_pdf(double x) { return kInvSqrt2Pi * std::exp(-0.5 * x * x); }

double norm_cdf(double x) { return 0.5 * std::erfc(-x * 0.70710678118654752440); }

double norm_ppf(double p) {
    if (std::isnan(p) || p < 0.0 || p > 1.0) return kNaN;
    if (p == 0.0) return -std::numeric_limits<double>::infinity();
    if (p == 1.0) return std::numeric_limits<double>::infinity();
    if (p > 0.5) return -norm_ppf_lower(1.0 - p);  // 1 - p is exact for p >= 0.5
    return norm_ppf_lower(p);
}

// ============================================================== Student-t

double t_log_pdf(double x, double nu) {
    if (std::isinf(nu) && nu > 0) return -0.5 * (kLn2Pi + x * x);  // Normal limit
    return t_log_norm_const(nu) - 0.5 * (nu + 1.0) * std::log1p(x * x / nu);
}

double t_pdf(double x, double nu) { return std::exp(t_log_pdf(x, nu)); }

double t_cdf(double x, double nu) {
    if (std::isnan(x) || !(nu > 0.0)) return kNaN;
    if (std::isinf(nu)) return norm_cdf(x);  // Normal limit
    if (std::isinf(x)) return x > 0 ? 1.0 : 0.0;
    if (x == 0.0) return 0.5;
    const double x2 = x * x;
    if (!std::isfinite(x2)) {
        // |x| > 1e154 >> sqrt(nu): power-law tail K |x|^-nu / nu, exact to O(nu / x^2)
        if (nu > 1e100) return x > 0 ? 1.0 : 0.0;
        const double tail = std::exp(t_log_norm_const(nu) + 0.5 * (nu - 1.0) * std::log(nu) -
                                     nu * std::log(std::fabs(x)));
        return x > 0.0 ? 1.0 - tail : tail;
    }
    if (nu >= 30.0 && x2 <= 0.4 * nu) {
        const double tail = t_tail_large_nu(nu, x2);
        return x > 0.0 ? 1.0 - tail : tail;
    }
    if (x2 < 1e-4 * nu) {
        // Near zero: P(|T| < |x|) = I_{x^2/(nu+x^2)}(1/2, nu/2) keeps full precision
        // (the tail argument nu/(nu+x^2) would round to 1). The result is ~0.5 here,
        // so the subtraction below cannot cancel.
        const double w = x2 / (nu + x2);
        const double half_central = 0.5 * inc_beta_xc(0.5, 0.5 * nu, w, nu / (nu + x2));
        return x >= 0.0 ? 0.5 + half_central : 0.5 - half_central;
    }
    // Tail: P(T < -|x|) = 0.5 * I_{nu/(nu+x^2)}(nu/2, 1/2), computed directly so that
    // small tail probabilities keep full *relative* precision.
    const double z = nu / (nu + x2);
    const double tail = 0.5 * inc_beta_xc(0.5 * nu, 0.5, z, x2 / (nu + x2));
    return x > 0.0 ? 1.0 - tail : tail;
}

double t_ppf(double p, double nu) {
    if (std::isnan(p) || p < 0.0 || p > 1.0 || !(nu > 0.0)) return kNaN;
    if (p == 0.0) return -std::numeric_limits<double>::infinity();
    if (p == 1.0) return std::numeric_limits<double>::infinity();
    if (p > 0.5) return -t_ppf(1.0 - p, nu);  // exact complement
    if (p == 0.5) return 0.0;

    const double z = norm_ppf(p);
    // Cornish-Fisher expansion of the t quantile around the Normal one.
    const double z2 = z * z, z3 = z2 * z, z5 = z3 * z2, z7 = z5 * z2, z9 = z7 * z2;
    const double x_cf = z + (z3 + z) / (4.0 * nu) + (5 * z5 + 16 * z3 + 3 * z) / (96.0 * nu * nu) +
                        (3 * z7 + 19 * z5 + 17 * z3 - 15 * z) / (384.0 * nu * nu * nu) +
                        (79 * z9 + 776 * z7 + 1482 * z5 - 1920 * z3 - 945 * z) /
                            (92160.0 * nu * nu * nu * nu);
    if (nu > 1e7) return x_cf;  // error O(nu^-5): far below double precision

    // Power-law tail approximation: F(x) ~ K |x|^-nu / nu for x -> -inf.
    const double log_k = t_log_norm_const(nu) + 0.5 * (nu + 1.0) * std::log(nu);
    const double x_tail = -std::exp((log_k - std::log(nu) - std::log(p)) / nu);

    auto log_err = [&](double x) {
        const double F = t_cdf(x, nu);
        return (F > 0.0 && std::isfinite(x)) ? std::fabs(std::log(F / p)) : HUGE_VAL;
    };
    double x = (log_err(x_cf) <= log_err(x_tail)) ? x_cf : x_tail;
    if (!(x < 0.0) || !std::isfinite(x)) x = -1.0;

    // Bracket the root: F(lo) <= p <= F(hi).
    double lo, hi = 0.0;
    if (t_cdf(x, nu) <= p) {
        lo = x;
    } else {
        hi = x;
        lo = 2.0 * x;
        for (int i = 0; i < 2000 && t_cdf(lo, nu) > p; ++i) {
            hi = lo;
            lo *= 2.0;
            if (!std::isfinite(lo)) return lo;
        }
        x = hi;  // the bracket end nearest the start (the start itself unless it was expanded)
    }

    // Safeguarded Newton iterations on g(x) = ln F(x) - ln p, g' = f / F: nearly linear in the
    // tail, where F spans hundreds of orders of magnitude across the bracket (Newton or Halley on
    // F - p crawl there: from a point with F << p their steps are ~|x| / nu).
    const double log_p = std::log(p);
    for (int it = 0; it < 400; ++it) {
        const double F = t_cdf(x, nu);
        if (F == p) break;
        if (F < p) lo = x; else hi = x;
        const double f = t_pdf(x, nu);
        double x_new = HUGE_VAL;  // F underflowed / no density: bisection
        if (F > 0.0 && f > 0.0 && std::isfinite(f)) {
            const double ratio = F / p;  // ln(F/p) keeps full precision near the root
            x_new = x - (std::isfinite(ratio) ? std::log(ratio) : std::log(F) - log_p) * (F / f);
        }
        if (!(x_new > lo && x_new < hi)) {
            // Bisection; geometric in the far tail (bracket spans orders of magnitude).
            x_new = (hi < -1.0) ? -std::sqrt(lo * hi) : 0.5 * (lo + hi);
        }
        const bool done = std::fabs(x_new - x) <= 4e-16 * std::fmax(1.0, std::fabs(x));
        x = x_new;
        if (done) break;
    }
    return x;
}

double std_t_log_pdf(double z, double nu) {
    if (std::isinf(nu) && nu > 0) return -0.5 * (kLn2Pi + z * z);
    return log_gamma_half_ratio(0.5 * nu) - 0.5 * std::log(kPi * (nu - 2.0)) -
           0.5 * (nu + 1.0) * std::log1p(z * z / (nu - 2.0));
}

double std_t_pdf(double z, double nu) { return std::exp(std_t_log_pdf(z, nu)); }

double std_t_cdf(double z, double nu) {
    if (std::isinf(nu) && nu > 0) return norm_cdf(z);
    return t_cdf(z * std::sqrt(nu / (nu - 2.0)), nu);
}

double std_t_ppf(double p, double nu) {
    if (std::isinf(nu) && nu > 0) return norm_ppf(p);
    return t_ppf(p, nu) * std::sqrt((nu - 2.0) / nu);
}

// ============================================================== chi-square

double chi2_sf(double x, double df) {
    if (std::isnan(x) || !(df > 0.0)) return kNaN;
    if (x <= 0.0) return 1.0;
    return inc_gamma_upper(0.5 * df, 0.5 * x);
}

}  // namespace garcht::math
