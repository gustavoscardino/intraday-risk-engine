#include "ewmar/ewma.hpp"

#include <algorithm>
#include <cmath>

#include "ewmar/math.hpp"

namespace ewmar {

void EwmaCovariance::update(const double* y, double lambda) {
    const size_t n = n_;
    const double w = 1.0 - lambda, sl = std::sqrt(lambda);
    size_t n_obs = 0;
    for (size_t i = 0; i < n; ++i) {
        obs_[i] = std::isfinite(y[i]) ? 1 : 0;
        n_obs += obs_[i];
    }
    if (n_obs == 0) return;  // nothing observed: S unchanged
    if (n_obs == n) {
        for (size_t i = 0; i < n; ++i) {
            double* row = &S_[i * n];
            const double wy = w * y[i];
            for (size_t j = 0; j < n; ++j) row[j] = lambda * row[j] + wy * y[j];
        }
        return;
    }
    for (size_t i = 0; i < n; ++i) {
        double* row = &S_[i * n];
        for (size_t j = 0; j < n; ++j) {
            if (obs_[i] && obs_[j]) row[j] = lambda * row[j] + w * y[i] * y[j];
            else if (obs_[i] != obs_[j]) row[j] *= sl;
            // both missing: unchanged
        }
    }
}

double EwmaCovariance::quad(const double* a) const {
    double s = 0.0;
    for (size_t i = 0; i < n_; ++i) {
        if (a[i] == 0.0) continue;
        const double* row = &S_[i * n_];
        double acc = 0.0;
        for (size_t j = 0; j < n_; ++j) acc += row[j] * a[j];
        s += a[i] * acc;
    }
    return s;
}

void EwmaCovariance::correlation(std::vector<double>& R) const {
    R.assign(n_ * n_, 0.0);
    for (size_t i = 0; i < n_; ++i)
        for (size_t j = 0; j < n_; ++j) {
            const double d = S_[i * n_ + i] * S_[j * n_ + j];
            R[i * n_ + j] = i == j ? 1.0 : (d > 0.0 ? S_[i * n_ + j] / std::sqrt(d) : 0.0);
        }
}

std::vector<double> initial_covariance(const std::vector<double>& rows, size_t n, size_t from, size_t to,
                                       double floor_var) {
    std::vector<double> S(n * n, 0.0);
    size_t complete = 0;
    for (size_t t = from; t < to; ++t) {
        const double* y = &rows[t * n];
        bool ok = true;
        for (size_t i = 0; i < n && ok; ++i) ok = std::isfinite(y[i]);
        if (!ok) continue;
        ++complete;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j) S[i * n + j] += y[i] * y[j];
    }
    if (complete >= n + 5) {
        for (double& v : S) v /= static_cast<double>(complete);
    } else {  // too few complete rows: diagonal of mean squares
        std::fill(S.begin(), S.end(), 0.0);
        for (size_t i = 0; i < n; ++i) {
            double ss = 0.0;
            size_t k = 0;
            for (size_t t = from; t < to; ++t) {
                const double v = rows[t * n + i];
                if (std::isfinite(v)) {
                    ss += v * v;
                    ++k;
                }
            }
            S[i * n + i] = k > 0 ? ss / static_cast<double>(k) : 0.0;
        }
    }
    for (size_t i = 0; i < n; ++i) S[i * n + i] = std::max(S[i * n + i], floor_var);
    return S;
}

double ewma_qlike(const std::vector<double>& rows, size_t n, size_t from, size_t to, const std::vector<double>& v0,
                  double lambda) {
    double acc = 0.0;
    const double w = 1.0 - lambda;
    for (size_t i = 0; i < n; ++i) {
        double v = v0[i];
        if (!(std::isfinite(v) && v > 0.0)) continue;
        for (size_t t = from; t < to; ++t) {
            const double y = rows[t * n + i];
            if (!std::isfinite(y)) continue;
            acc += std::log(v) + y * y / v;
            v = lambda * v + w * y * y;
            if (v < 1e-300) v = 1e-300;
        }
    }
    return acc;
}

double estimate_lambda(const std::vector<double>& rows, size_t n, size_t from, size_t to,
                       const std::vector<double>& v0, double lo, double hi) {
    return math::brent_minimize([&](double lam) { return ewma_qlike(rows, n, from, to, v0, lam); }, lo, hi, 1e-7);
}

}  // namespace ewmar
