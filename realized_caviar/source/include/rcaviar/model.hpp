#pragma once
#include <cstddef>
#include <vector>

namespace rcaviar {
// Decimal log returns. Features are known at forecast time; target is future.
struct Observation { double aggregate_return; double realized_variance; double target_return; };
struct Params {
    double beta0 = -0.001, beta1 = 0.8, beta2 = -0.1, beta3 = -0.1;
    bool valid() const;
};
// Learning restriction: beta0,beta2,beta3 <= 0, 0 <= beta1 <= .995.
// Together with q0 <= 0 this preserves a lower-tail (negative) quantile.
class Filter {
public:
    Filter(Params params, double initial_quantile);
    double update(double aggregate_return, double realized_variance);
    double quantile() const { return q_; }
private:
    Params params_;
    double q_;
};
double pinball(double realized_return, double quantile, double tau);
double long_var(double exposure, double quantile); // positive long exposure only
struct FitOptions { double tau = .01; int max_iterations = 1200; int starts = 6; double tolerance = 1e-7; };
struct FitResult {
    Params params;
    double initial_quantile = 0, last_quantile = 0, mean_loss = 0, constant_loss = 0;
    int evaluations = 0;
    bool converged = false;
};
// Exact, nonsmooth pinball objective; deterministic multi-start Nelder-Mead.
// The caller must supply training observations only. NaN targets are unscored
// but their observed features still advance the recursion. Infinity is invalid.
FitResult fit(const std::vector<Observation>& training, const FitOptions& options = {});
}
