#include "riskcommon/riskmetrics.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace riskcommon {

double normal_pdf(double x) { return 0.3989422804014327 * std::exp(-0.5 * x * x); }

// Wichura (1988), algorithm AS 241 (PPND16): about 16 significant digits.
double normal_quantile(double p) {
    if (!(p > 0.0 && p < 1.0)) return p == 0.0 ? -kInf : (p == 1.0 ? kInf : kNaN);
    const double q = p - 0.5;
    if (std::fabs(q) <= 0.425) {
        const double r = 0.180625 - q * q;
        return q *
               (((((((2509.0809287301226727 * r + 33430.575583588128105) * r + 67265.770927008700853) * r +
                    45921.953931549871457) * r + 13731.693765509461125) * r + 1971.5909503065514427) * r +
                 133.14166789178437745) * r + 3.387132872796366608) /
               (((((((5226.495278852545925 * r + 28729.085735721942674) * r + 39307.89580009271061) * r +
                    21213.794301586595867) * r + 5394.1960214247511077) * r + 687.1870074920579083) * r +
                 42.313330701600911252) * r + 1.0);
    }
    double r = q < 0.0 ? p : 1.0 - p;
    r = std::sqrt(-std::log(r));
    double x;
    if (r <= 5.0) {
        r -= 1.6;
        x = (((((((7.7454501427834140764e-4 * r + 0.0227238449892691845833) * r + 0.24178072517745061177) * r +
                 1.27045825245236838258) * r + 3.64784832476320460504) * r + 5.7694972214606914055) * r +
              4.6303378461565452959) * r + 1.42343711074968357734) /
            (((((((1.05075007164441684324e-9 * r + 5.475938084995344946e-4) * r + 0.0151986665636164571966) * r +
                 0.14810397642748007459) * r + 0.68976733498510000455) * r + 1.6763848301838038494) * r +
              2.05319162663775882187) * r + 1.0);
    } else {
        r -= 5.0;
        x = (((((((2.01033439929228813265e-7 * r + 2.71155556874348757815e-5) * r + 0.0012426609473880784386) * r +
                 0.026532189526576123093) * r + 0.29656057182850489123) * r + 1.7848265399172913358) * r +
              5.4637849111641143699) * r + 6.6579046435011037772) /
            (((((((2.04426310338993978564e-15 * r + 1.4215117583164458887e-7) * r + 1.8463183175100546818e-5) * r +
                 7.868691311456132591e-4) * r + 0.0148753612908506148525) * r + 0.13692988092273580531) * r +
              0.59983220655588793769) * r + 1.0);
    }
    return q < 0.0 ? -x : x;
}

void ewma_covariance_update(std::vector<double>& S, size_t n, const double* y, double lambda) {
    const double w = 1.0 - lambda, sl = std::sqrt(lambda);
    for (size_t i = 0; i < n; ++i) {
        const bool oi = std::isfinite(y[i]);
        double* row = &S[i * n];
        for (size_t j = 0; j < n; ++j) {
            const bool oj = std::isfinite(y[j]);
            if (oi && oj) row[j] = lambda * row[j] + w * y[i] * y[j];
            else if (oi != oj) row[j] *= sl;
            // both missing: kept
        }
    }
}

std::vector<double> initial_covariance(const double* rows, size_t n, size_t from, size_t to, double floor_var) {
    std::vector<double> S(n * n, 0.0);
    size_t complete = 0;
    for (size_t t = from; t < to; ++t) {
        const double* r = rows + t * n;
        bool ok = true;
        for (size_t i = 0; i < n && ok; ++i) ok = std::isfinite(r[i]);
        if (!ok) continue;
        ++complete;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j) S[i * n + j] += r[i] * r[j];
    }
    if (complete >= n + 5) {
        for (double& v : S) v /= static_cast<double>(complete);
    } else {  // too few complete rows: each asset's mean square, no covariances
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

RiskMetricsModel::RiskMetricsModel(size_t assets, std::vector<double> confidences, double lambda, ModelInfo info)
    : n_(assets), conf_(std::move(confidences)), lambda_(lambda), info_(std::move(info)), S_(assets * assets, 0.0) {
    if (assets == 0) throw std::invalid_argument("RiskMetricsModel: no assets");
    if (!(lambda > 0.0 && lambda < 1.0)) throw std::invalid_argument("RiskMetricsModel: lambda must be in (0, 1)");
    for (double c : conf_) {
        const double q = normal_quantile(c);
        q_.push_back(q);
        e_.push_back(normal_pdf(q) / (1.0 - c));
    }
}

void RiskMetricsModel::on_bar(int, const double* r) {
    if (ready_) ewma_covariance_update(S_, n_, r, lambda_);
}

void RiskMetricsModel::refit(const History& h, const double*) {
    const size_t W = h.bars;
    if (W < 2 || h.assets != n_) return;
    const size_t burn = std::min(W / 2, std::max(W / 4, n_ + 10));
    S_ = initial_covariance(h.returns, n_, 0, burn);
    for (size_t t = burn; t < W; ++t) ewma_covariance_update(S_, n_, h.row(t), lambda_);
    ready_ = true;
}

void RiskMetricsModel::forecast(int, const double* x, ModelForecast& out) const {
    out.mean = 0.0;
    out.shape = Shape{};
    out.parameters.clear();
    out.measures.resize(conf_.size());
    double var = 0.0;
    for (size_t i = 0; i < n_; ++i) {
        const double xi = std::isfinite(x[i]) ? x[i] : 0.0;
        if (xi == 0.0) continue;
        const double* row = &S_[i * n_];
        double acc = 0.0;
        for (size_t j = 0; j < n_; ++j) acc += row[j] * (std::isfinite(x[j]) ? x[j] : 0.0);
        var += xi * acc;
    }
    out.sigma = ready_ ? std::sqrt(std::max(var, 0.0)) : kNaN;
    for (size_t k = 0; k < conf_.size(); ++k)
        out.measures[k] = {conf_[k], out.sigma * q_[k], out.sigma * e_[k]};
}

}  // namespace riskcommon
