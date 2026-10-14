// GARCH(1,1) with standardized Student-t (or Normal) innovations.
//
//   y_t     = mu + eps_t,         eps_t = sqrt(h_t) * z_t,   z_t ~ iid D(0, 1)
//   h_t     = omega + alpha * eps_{t-1}^2 + beta * h_{t-1}
//
// D is either N(0,1) or the unit-variance Student-t with nu > 2 degrees of freedom.
// Constraints: omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1.
//
// The recursion is started with eps_{-1}^2 = h_{-1} = backcast, where backcast is an
// exponentially weighted average (0.94) of the first min(75, T) squared residuals
// (the same convention as the Python `arch` package, which allows cross-checking).
//
// Missing observations (NaN) are supported both in estimation and online filtering:
// the unobserved eps_t^2 is replaced by its conditional expectation h_t.
#pragma once

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace garcht {

enum class Dist { Normal, StudentT };
enum class MeanModel { Zero, Constant };

struct GarchSpec {
    Dist dist = Dist::StudentT;
    MeanModel mean = MeanModel::Zero;
    bool variance_targeting = false;  // omega = (1 - alpha - beta) * sample variance
};

struct GarchParams {
    double mu = 0.0;
    double omega = 0.0;
    double alpha = 0.0;
    double beta = 0.0;
    double nu = std::numeric_limits<double>::infinity();  // +inf => Normal

    double persistence() const { return alpha + beta; }
    double unconditional_variance() const;  // omega / (1 - alpha - beta), +inf if non-stationary
    double half_life() const;               // periods for a variance shock to halve
    bool is_valid(Dist dist) const;
};

// Number of free parameters for a spec (mu?, omega?, alpha, beta, nu?).
int num_free_params(const GarchSpec& spec);

// Backcast used to initialize the variance recursion (finite values only).
double garch_backcast(const double* resid, size_t n);

// Log-likelihood. If grad != nullptr, writes dL/d(mu, omega, alpha, beta, nu) (5 entries,
// natural parameters, omega treated as free). h_out (size n) receives the conditional
// variances h_t if given. Returns -inf for invalid parameters.
double garch_loglik(const double* y, size_t n, Dist dist, const GarchParams& p, double backcast,
                    double* grad = nullptr, double* h_out = nullptr);

// Conditional variance path h_0..h_{n-1}; returns h_n (the one-step-ahead forecast).
double garch_variance_path(const double* y, size_t n, const GarchParams& p, double backcast,
                           double* h_out);

// -------------------------------------------------------------------- estimation

struct FitOptions {
    int max_iter = 400;
    double grad_tol = 1e-7;            // on the mean log-likelihood of standardized data
    double max_persistence = 0.99995;  // alpha + beta upper bound
    double nu_min = 2.1;
    double nu_max = 300.0;
    int min_obs = 100;
    bool compute_std_errors = true;
    int n_starts = 2;  // cold start: BFGS runs from the best `n_starts` grid points
    // Warm start (e.g. yesterday's estimate): BFGS starts there; the grid is still
    // evaluated and searched only if it beats the warm-started optimum.
    const GarchParams* warm_start = nullptr;
};

enum class FitStatus { Ok, NotConverged, InsufficientData, DegenerateData, NumericalError };
const char* to_string(FitStatus s);

struct FitResult {
    FitStatus status = FitStatus::NumericalError;
    GarchParams params;
    GarchParams std_err;         // classic (inverse Hessian); NaN where not estimable; left at the
                                 // GarchParams defaults when the fit stopped early (no data)
    GarchParams robust_std_err;  // Bollerslev-Wooldridge sandwich
    double loglik = -std::numeric_limits<double>::infinity();
    double aic = std::numeric_limits<double>::quiet_NaN();
    double bic = std::numeric_limits<double>::quiet_NaN();
    double backcast = std::numeric_limits<double>::quiet_NaN();
    double last_variance = std::numeric_limits<double>::quiet_NaN();  // h_{T-1} (in-sample)
    double next_variance = std::numeric_limits<double>::quiet_NaN();  // h_T (forecast)
    double grad_inf_norm = std::numeric_limits<double>::quiet_NaN();
    int iterations = 0;
    int evaluations = 0;
    int n_obs = 0;  // finite observations used
    int n_params = 0;
    std::string message;

    bool usable() const { return status == FitStatus::Ok || status == FitStatus::NotConverged; }
};

FitResult fit_garch(const double* y, size_t n, const GarchSpec& spec, const FitOptions& opt = {});
inline FitResult fit_garch(const std::vector<double>& y, const GarchSpec& spec,
                           const FitOptions& opt = {}) {
    return fit_garch(y.data(), y.size(), spec, opt);
}

// -------------------------------------------------------------------- online filter

// O(1) per observation. Holds h_{t+1}, the variance of the *next* observation.
class GarchFilter {
public:
    GarchFilter() = default;
    GarchFilter(const GarchParams& p, double next_variance) { reset(p, next_variance); }

    void reset(const GarchParams& p, double next_variance);

    // Feeds observation y_t; returns the standardized residual z_t = (y_t - mu)/sqrt(h_t).
    // A non-finite y is treated as missing. If z_clip > 0, |z| beyond z_clip is
    // winsorized *in the variance recursion only* (robust filter against bad ticks).
    double update(double y, double z_clip = 0.0);
    void update_missing();

    const GarchParams& params() const { return p_; }
    double variance() const { return h_; }  // h_{t+1}
    double mean() const { return p_.mu; }

    double forecast_variance(int k) const;    // E[h_{t+k}], k >= 1
    double cumulative_variance(int k) const;  // sum_{j=1..k} E[h_{t+j}]

private:
    GarchParams p_;
    double h_ = std::numeric_limits<double>::quiet_NaN();
};

}  // namespace garcht
