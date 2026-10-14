// Parametric VaR / Expected Shortfall for location-scale outcomes X = mu + sigma * Z.
//
// Sign convention: VaR and ES are reported as positive losses.
//   VaR_c = -mu + sigma * q_c            q_c  = -F_Z^{-1}(1 - c)
//   ES_c  = -mu + sigma * e_c            e_c  = -E[Z | Z <= F_Z^{-1}(1 - c)]
//
// Shock distributions:
//   Normal:  q_c = Phi^{-1}(c),          e_c = phi(q_c) / (1 - c)
//   Std-t:   x = t_nu^{-1}(1 - c) < 0,   k = sqrt((nu - 2) / nu)
//            q_c = -k x,                 e_c = k * f_nu(x) / (1 - c) * (nu + x^2) / (nu - 1)
//   t body + GPD left tail (conditional EVT, McNeil & Frey 2000): for losses L = -Z above a
//   threshold u0 (exceeded with probability theta),
//            P(L > u0 + y) = theta * (1 + xi y / beta)^(-1/xi)
//            q_c = u0 + beta/xi * ((p/theta)^(-xi) - 1),   e_c = (q_c + beta - xi u0) / (1 - xi)
//   with p = 1 - c <= theta; above -u0 the t body carries the remaining mass 1 - theta.
#pragma once

#include <limits>
#include <vector>

#include "garcht/garch.hpp"

namespace garcht {

class Rng;

struct TailFactors {
    double confidence = 0.0;
    double var = 0.0;  // q_c
    double es = 0.0;   // e_c (>= q_c)
};

// nu is ignored for Dist::Normal; a non-finite nu with StudentT falls back to Normal.
TailFactors tail_factors(Dist dist, double nu, double confidence);

// Standardized shock distribution: unit-variance Student-t (Normal if nu = +inf), optionally
// with a generalized Pareto left tail.
struct ShockDist {
    double nu = std::numeric_limits<double>::infinity();
    bool gpd = false;
    double u0 = 0.0;     // loss threshold (standardized units, > 0)
    double beta = 1.0;   // GPD scale
    double xi = 0.0;     // GPD shape (heavier tail when larger; < 1 for a finite ES)
    double theta = 0.0;  // P(L > u0)

    static ShockDist student(double nu);
    double cdf(double z) const;
    TailFactors factors(double confidence) const;
    double sample(Rng& rng) const;
    bool operator==(const ShockDist& o) const;
};

// Maximum-likelihood fits on standardized outcomes u (non-finite values are ignored).
// Student: unit-variance t, nu in [nu_min, nu_max].
ShockDist fit_student_shock(const std::vector<double>& u, double nu_min, double nu_max);
// EVT: GPD fitted to the largest `tail_fraction` losses -u, t(nu) body. Falls back to the
// plain t when there are fewer than `min_exceedances` exceedances or the fit fails.
ShockDist fit_evt_shock(const std::vector<double>& u, double nu, double tail_fraction = 0.10,
                        int min_exceedances = 50);

struct RiskMeasure {
    double confidence = 0.0;
    double var = 0.0;
    double es = 0.0;
};

inline RiskMeasure risk_measure(double mu, double sigma, const TailFactors& f) {
    return {f.confidence, -mu + sigma * f.var, -mu + sigma * f.es};
}

// Joint one-step-ahead return forecast: r_i = mu_i + sigma_i * Z_i, Z ~ multivariate
// unit-variance Student-t(nu) (Normal if nu = +inf) with correlation matrix `corr`.
struct JointForecast {
    std::vector<double> mu, sigma;  // per asset, log-return units
    std::vector<double> corr;       // n x n, row-major
    double nu = 0.0;
};

// Cache of tail factors for a fixed list of confidence levels (recomputed only when the
// distribution changes).
class TailTable {
public:
    TailTable() = default;
    TailTable(Dist dist, double nu, const std::vector<double>& confidences) {
        set(dist, nu, confidences);
    }
    void set(Dist dist, double nu, const std::vector<double>& confidences);
    void set(const ShockDist& d, const std::vector<double>& confidences);
    const std::vector<TailFactors>& factors() const { return f_; }
    double nu() const { return dist_.nu; }
    const ShockDist& dist() const { return dist_; }

private:
    std::vector<TailFactors> f_;
    ShockDist dist_;
};

}  // namespace garcht
