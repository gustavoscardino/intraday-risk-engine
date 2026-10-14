// Exponentially weighted moving average (EWMA) of the covariance of returns:
//
//     S(t) = lambda * S(t-1) + (1 - lambda) * y(t) y(t)'
//
// with one lambda for all assets, so S stays positive semidefinite. Missing returns:
//   * entries between two observed assets are updated as above;
//   * the variance of a missing asset, and its covariances with other missing assets, keep
//     their previous values;
//   * its covariances with observed assets are multiplied by sqrt(lambda).
// The last rule keeps S positive semidefinite: S_new = D S_old D + (1 - lambda) y_o y_o',
// D = diag(sqrt(lambda) on observed assets, 1 on missing ones), a sum of two PSD matrices.
// (Simply keeping the old cross covariances can break positive definiteness.)
#pragma once

#include <cstddef>
#include <vector>

namespace ewmar {

class EwmaCovariance {
public:
    EwmaCovariance() = default;
    explicit EwmaCovariance(size_t n) : n_(n), S_(n * n, 0.0), obs_(n, 0) {}

    void set(const std::vector<double>& S) { S_ = S; }
    void update(const double* y, double lambda);  // y: n returns, NaN = missing

    size_t size() const { return n_; }
    const std::vector<double>& matrix() const { return S_; }
    double operator()(size_t i, size_t j) const { return S_[i * n_ + j]; }
    double quad(const double* a) const;  // a' S a
    void correlation(std::vector<double>& R) const;

private:
    size_t n_ = 0;
    std::vector<double> S_;
    std::vector<char> obs_;
};

// Initial covariance: mean of y y' (zero mean) over the complete rows of [from, to); when there
// are fewer than n + 5 complete rows, a diagonal matrix of each asset's mean square. Variances
// are floored at `floor_var`. rows: row-major [bar][asset].
std::vector<double> initial_covariance(const std::vector<double>& rows, size_t n, size_t from, size_t to,
                                       double floor_var = 1e-14);

// Gaussian predictive quasi-likelihood of the univariate EWMA variances, pooled over assets:
//     sum_i sum_t [ ln v_i(t|t-1) + y_i(t)^2 / v_i(t|t-1) ]      (to be minimized)
// over rows [from, to), each asset starting from v0[i] (non-finite v0 = asset skipped).
double ewma_qlike(const std::vector<double>& rows, size_t n, size_t from, size_t to, const std::vector<double>& v0,
                  double lambda);

// lambda minimizing ewma_qlike over [lo, hi] (Brent).
double estimate_lambda(const std::vector<double>& rows, size_t n, size_t from, size_t to,
                       const std::vector<double>& v0, double lo = 0.80, double hi = 0.999);

}  // namespace ewmar
