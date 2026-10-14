#include "ewmar/math.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ewmar::math {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kEps = 1e-15;
constexpr double kTiny = 1e-300;

// Continued fraction of the incomplete beta function (modified Lentz method).
double beta_cf(double a, double b, double x) {
    const double qab = a + b, qap = a + 1.0, qam = a - 1.0;
    double c = 1.0, d = 1.0 - qab * x / qap;
    if (std::fabs(d) < kTiny) d = kTiny;
    d = 1.0 / d;
    double h = d;
    for (int m = 1; m <= 500; ++m) {
        const double m2 = 2.0 * m;
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if (std::fabs(d) < kTiny) d = kTiny;
        c = 1.0 + aa / c;
        if (std::fabs(c) < kTiny) c = kTiny;
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (std::fabs(d) < kTiny) d = kTiny;
        c = 1.0 + aa / c;
        if (std::fabs(c) < kTiny) c = kTiny;
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        if (std::fabs(del - 1.0) < kEps) break;
    }
    return h;
}

// ln of the prefactor x^a (1-x)^b / (a B(a, b))
double beta_front_log(double a, double b, double x) {
    return std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) + a * std::log(x) + b * std::log1p(-x);
}

}  // namespace

// ============================================================== Normal

double norm_pdf(double x) { return std::exp(-0.5 * x * x - 0.5 * kLn2Pi); }

double norm_cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

double norm_ppf(double p) {
    if (std::isnan(p) || p < 0.0 || p > 1.0) return kNaN;
    if (p == 0.0) return -kInf;
    if (p == 1.0) return kInf;
    if (p > 0.5) return -norm_ppf(1.0 - p);
    // Acklam's rational approximation (relative error 1.2e-9) ...
    static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                               1.383577518672690e+02,  -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                               6.680131188771972e+01,  -1.328068155288572e+01};
    static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                               -2.549732539343734e+00, 4.374664141464968e+00,  2.938163982698783e+00};
    static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
                               3.754408661907416e+00};
    double x;
    if (p < 0.02425) {
        const double q = std::sqrt(-2.0 * std::log(p));
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    } else {
        const double q = p - 0.5, r = q * q;
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    }
    // ... refined by Halley steps on the exact cdf (full double precision)
    for (int it = 0; it < 2; ++it) {
        const double e = norm_cdf(x) - p;
        const double u = e * std::sqrt(2.0 * kPi) * std::exp(0.5 * x * x);
        x -= u / (1.0 + 0.5 * x * u);
    }
    return x;
}

// ============================================================== incomplete functions

double inc_beta(double a, double b, double x) {
    if (std::isnan(x) || !(a > 0.0) || !(b > 0.0)) return kNaN;
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;
    const double front = beta_front_log(a, b, x);
    if (x < (a + 1.0) / (a + b + 2.0)) return std::exp(front) * beta_cf(a, b, x) / a;
    return 1.0 - std::exp(front) * beta_cf(b, a, 1.0 - x) / b;
}

double inc_gamma_p(double a, double x) {
    if (std::isnan(x) || !(a > 0.0)) return kNaN;
    if (x <= 0.0) return 0.0;
    if (x < a + 1.0) {  // series
        double ap = a, sum = 1.0 / a, del = sum;
        for (int n = 0; n < 1000; ++n) {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if (std::fabs(del) < std::fabs(sum) * kEps) break;
        }
        return sum * std::exp(-x + a * std::log(x) - std::lgamma(a));
    }
    return 1.0 - inc_gamma_q(a, x);
}

double inc_gamma_q(double a, double x) {
    if (std::isnan(x) || !(a > 0.0)) return kNaN;
    if (x <= 0.0) return 1.0;
    if (x < a + 1.0) return 1.0 - inc_gamma_p(a, x);
    // continued fraction (modified Lentz)
    double b = x + 1.0 - a, c = 1.0 / kTiny, d = 1.0 / b, h = d;
    for (int i = 1; i < 1000; ++i) {
        const double an = -i * (i - a);
        b += 2.0;
        d = an * d + b;
        if (std::fabs(d) < kTiny) d = kTiny;
        c = b + an / c;
        if (std::fabs(c) < kTiny) c = kTiny;
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        if (std::fabs(del - 1.0) < kEps) break;
    }
    return std::exp(-x + a * std::log(x) - std::lgamma(a)) * h;
}

double chi2_sf(double x, double k) {
    if (std::isnan(x)) return kNaN;
    if (x <= 0.0) return 1.0;
    return inc_gamma_q(0.5 * k, 0.5 * x);
}

// ============================================================== Student t

double t_logpdf(double x, double nu) {
    if (!std::isfinite(nu)) return -0.5 * x * x - 0.5 * kLn2Pi;
    return std::lgamma(0.5 * (nu + 1.0)) - std::lgamma(0.5 * nu) - 0.5 * std::log(nu * kPi) -
           0.5 * (nu + 1.0) * std::log1p(x * x / nu);
}

double t_pdf(double x, double nu) { return std::exp(t_logpdf(x, nu)); }

double t_cdf(double x, double nu) {
    if (std::isnan(x) || !(nu > 0.0)) return kNaN;
    if (!std::isfinite(nu)) return norm_cdf(x);
    if (x == kInf) return 1.0;
    if (x == -kInf) return 0.0;
    const double x2 = x * x, xb = nu / (nu + x2), a = 0.5 * nu;
    // tail P(T < -|x|) = I_xb(nu/2, 1/2)/2 is accurate while the incomplete beta takes its direct
    // continued fraction; nearer the centre, P(|T| < |x|) = I_{1-xb}(1/2, nu/2) avoids 1 - (~1)
    if (xb < (a + 1.0) / (a + 2.5)) {
        const double tail = 0.5 * inc_beta(a, 0.5, xb);
        return x < 0.0 ? tail : 1.0 - tail;
    }
    const double c = 0.5 * inc_beta(0.5, a, x2 / (nu + x2));
    return x < 0.0 ? 0.5 - c : 0.5 + c;
}

double t_ppf(double p, double nu) {
    if (std::isnan(p) || p < 0.0 || p > 1.0 || !(nu > 0.0)) return kNaN;
    if (!std::isfinite(nu)) return norm_ppf(p);
    if (p == 0.0) return -kInf;
    if (p == 1.0) return kInf;
    if (p == 0.5) return 0.0;
    if (p > 0.5) return -t_ppf(1.0 - p, nu);
    // Cornish-Fisher start from the Normal quantile, then safeguarded Newton on the cdf
    const double z = norm_ppf(p);
    double x = z + (z * z * z + z) / (4.0 * nu) + (5.0 * std::pow(z, 5) + 16.0 * z * z * z + 3.0 * z) / (96.0 * nu * nu);
    if (!(x < 0.0)) x = z;
    double hi = 0.0, lo = std::min(x, -1.0);
    while (t_cdf(lo, nu) > p) {
        hi = lo;
        lo *= 2.0;
        if (lo < -1e300) return -kInf;
    }
    x = std::clamp(x, lo, hi);
    for (int it = 0; it < 200; ++it) {
        const double f = t_cdf(x, nu) - p;
        if (f > 0.0) hi = x;
        else lo = x;
        const double pdf = t_pdf(x, nu);
        double xn = pdf > 0.0 ? x - f / pdf : 0.5 * (lo + hi);
        if (!(xn > lo && xn < hi)) xn = 0.5 * (lo + hi);
        if (std::fabs(xn - x) <= 1e-15 * (1.0 + std::fabs(x))) return xn;
        x = xn;
    }
    return x;
}

double std_t_logpdf(double z, double nu) {
    if (!std::isfinite(nu)) return -0.5 * z * z - 0.5 * kLn2Pi;
    const double c = std::sqrt(nu / (nu - 2.0));
    return t_logpdf(z * c, nu) + std::log(c);
}

double std_t_cdf(double z, double nu) {
    if (!std::isfinite(nu)) return norm_cdf(z);
    return t_cdf(z * std::sqrt(nu / (nu - 2.0)), nu);
}

// ============================================================== Brent minimization

double brent_minimize(const std::function<double(double)>& f, double lo, double hi, double tol, double* fmin,
                      int max_iter) {
    const double golden = 0.3819660112501051;  // (3 - sqrt 5) / 2
    double a = std::min(lo, hi), b = std::max(lo, hi);
    double x = a + golden * (b - a), w = x, v = x;
    double fx = f(x), fw = fx, fv = fx;
    double d = 0.0, e = 0.0;
    for (int it = 0; it < max_iter; ++it) {
        const double xm = 0.5 * (a + b);
        const double tol1 = tol * std::fabs(x) + 1e-12, tol2 = 2.0 * tol1;
        if (std::fabs(x - xm) <= tol2 - 0.5 * (b - a)) break;
        bool golden_step = true;
        if (std::fabs(e) > tol1) {  // try a parabola through x, w, v
            const double r = (x - w) * (fx - fv);
            double q = (x - v) * (fx - fw);
            double p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if (q > 0.0) p = -p;
            q = std::fabs(q);
            const double e_old = e;
            e = d;
            if (std::fabs(p) < std::fabs(0.5 * q * e_old) && p > q * (a - x) && p < q * (b - x)) {
                d = p / q;
                const double u = x + d;
                if (u - a < tol2 || b - u < tol2) d = xm >= x ? tol1 : -tol1;
                golden_step = false;
            }
        }
        if (golden_step) {
            e = x >= xm ? a - x : b - x;
            d = golden * e;
        }
        const double u = std::fabs(d) >= tol1 ? x + d : x + (d >= 0.0 ? tol1 : -tol1);
        const double fu = f(u);
        if (fu <= fx) {
            if (u >= x) a = x;
            else b = x;
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if (u < x) a = u;
            else b = u;
            if (fu <= fw || w == x) {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if (fu <= fv || v == x || v == w) {
                v = u;
                fv = fu;
            }
        }
    }
    if (fmin) *fmin = fx;
    return x;
}

// ============================================================== random numbers

namespace {
std::uint64_t splitmix64(std::uint64_t& s) {
    std::uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
}  // namespace

Rng::Rng(std::uint64_t seed) {
    for (auto& s : s_) s = splitmix64(seed);
}

std::uint64_t Rng::next() {
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

double Rng::uniform() { return (static_cast<double>(next() >> 11) + 0.5) * 0x1.0p-53; }

double Rng::normal() {
    if (has_spare_) {
        has_spare_ = false;
        return spare_;
    }
    double u, v, s;
    do {  // Marsaglia polar method
        u = 2.0 * uniform() - 1.0;
        v = 2.0 * uniform() - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    const double m = std::sqrt(-2.0 * std::log(s) / s);
    spare_ = v * m;
    has_spare_ = true;
    return u * m;
}

double Rng::gamma(double shape) {
    if (shape < 1.0) return gamma(shape + 1.0) * std::pow(uniform(), 1.0 / shape);
    const double d = shape - 1.0 / 3.0, c = 1.0 / std::sqrt(9.0 * d);  // Marsaglia-Tsang
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

double Rng::student_t(double nu) {
    if (!std::isfinite(nu)) return normal();
    return normal() / std::sqrt(2.0 * gamma(0.5 * nu) / nu);
}

}  // namespace ewmar::math
