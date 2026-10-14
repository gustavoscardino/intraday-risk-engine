// Special functions, distributions, a one-dimensional minimizer and a random number generator.
// Everything the EWMA engine and its backtest need, with no external dependencies.
#pragma once

#include <cstdint>
#include <functional>

namespace ewmar::math {

constexpr double kPi = 3.14159265358979323846;
constexpr double kLn2Pi = 1.83787706640934548356;  // ln(2 pi)

// ---------------------------------------------------------------- Normal
double norm_pdf(double x);
double norm_cdf(double x);
double norm_ppf(double p);  // inverse of norm_cdf; -inf / +inf at 0 / 1, NaN outside [0, 1]

// ---------------------------------------------------------------- incomplete functions
double inc_beta(double a, double b, double x);  // regularized I_x(a, b)
double inc_gamma_p(double a, double x);         // regularized lower P(a, x)
double inc_gamma_q(double a, double x);         // regularized upper Q(a, x) = 1 - P(a, x)
double chi2_sf(double x, double k);             // P(X > x), X ~ chi2(k)

// ---------------------------------------------------------------- Student t (standard: scale 1)
double t_pdf(double x, double nu);
double t_logpdf(double x, double nu);
double t_cdf(double x, double nu);
double t_ppf(double p, double nu);  // inverse of t_cdf

// Unit-variance t (nu > 2): Z = T * sqrt((nu - 2) / nu), T ~ t(nu).
double std_t_logpdf(double z, double nu);
double std_t_cdf(double z, double nu);

// ---------------------------------------------------------------- minimization
// Brent's method (golden section + parabolic steps) for a unimodal f on [lo, hi]. Returns the
// argmin; *fmin receives the minimum value when given.
double brent_minimize(const std::function<double(double)>& f, double lo, double hi, double tol = 1e-6,
                      double* fmin = nullptr, int max_iter = 200);

// ---------------------------------------------------------------- random numbers
// xoshiro256** seeded through splitmix64: fast, reproducible across platforms.
class Rng {
public:
    explicit Rng(std::uint64_t seed = 1);
    std::uint64_t next();
    double uniform();             // (0, 1), never exactly 0 or 1
    double normal();              // N(0, 1)
    double gamma(double shape);   // Gamma(shape, 1)
    double student_t(double nu);  // standard t(nu); nu = inf gives N(0, 1)

private:
    std::uint64_t s_[4];
    bool has_spare_ = false;
    double spare_ = 0.0;
};

}  // namespace ewmar::math
