#include "garcht/revaluation.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

#include "garcht/math.hpp"
#include "garcht/random.hpp"

namespace garcht {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

struct D12 {
    double d1, d2, sqrt_t;
};

D12 d12(double s, double k, double t, double vol, double r, double q) {
    const double st = std::sqrt(t);
    const double d1 = (std::log(s / k) + (r - q + 0.5 * vol * vol) * t) / (vol * st);
    return {d1, d1 - vol * st, st};
}

}  // namespace

double bs_price(OptionType type, double s, double k, double t, double vol, double r, double q) {
    // NaN inputs must not silently fall into the intrinsic-value branch below
    if (!(s > 0.0) || !(k > 0.0) || std::isnan(t) || std::isnan(vol) || std::isnan(r) || std::isnan(q))
        return kNaN;
    const double df_q = std::exp(-q * std::max(t, 0.0)), df_r = std::exp(-r * std::max(t, 0.0));
    if (!(t > 0.0) || !(vol > 0.0)) {
        const double fwd_intrinsic = s * df_q - k * df_r;
        return type == OptionType::Call ? std::max(fwd_intrinsic, 0.0) : std::max(-fwd_intrinsic, 0.0);
    }
    const D12 d = d12(s, k, t, vol, r, q);
    if (type == OptionType::Call)
        return s * df_q * math::norm_cdf(d.d1) - k * df_r * math::norm_cdf(d.d2);
    return k * df_r * math::norm_cdf(-d.d2) - s * df_q * math::norm_cdf(-d.d1);
}

double bs_delta(OptionType type, double s, double k, double t, double vol, double r, double q) {
    if (!(s > 0.0) || !(k > 0.0) || std::isnan(t) || std::isnan(vol) || std::isnan(r) || std::isnan(q))
        return kNaN;
    const double df_q = std::exp(-q * std::max(t, 0.0));
    if (!(t > 0.0) || !(vol > 0.0)) {
        const bool itm = s * df_q > k * std::exp(-r * std::max(t, 0.0));
        return type == OptionType::Call ? (itm ? df_q : 0.0) : (itm ? 0.0 : -df_q);
    }
    const D12 d = d12(s, k, t, vol, r, q);
    return type == OptionType::Call ? df_q * math::norm_cdf(d.d1) : df_q * (math::norm_cdf(d.d1) - 1.0);
}

double bs_gamma(double s, double k, double t, double vol, double r, double q) {
    if (std::isnan(s) || std::isnan(k) || std::isnan(t) || std::isnan(vol) || std::isnan(r) || std::isnan(q))
        return kNaN;
    if (!(s > 0.0) || !(k > 0.0) || !(t > 0.0) || !(vol > 0.0)) return 0.0;
    const D12 d = d12(s, k, t, vol, r, q);
    return std::exp(-q * t) * math::norm_pdf(d.d1) / (s * vol * d.sqrt_t);
}

double position_value(const Position& p, double spot, double shift) {
    if (p.kind == Position::Kind::Stock) return p.quantity * spot;
    return p.quantity * p.multiplier * bs_price(p.type, spot, p.strike, p.expiry - shift, p.vol, p.rate);
}

double position_delta_exposure(const Position& p, double spot) {
    if (p.kind == Position::Kind::Stock) return p.quantity * spot;
    return p.quantity * p.multiplier * spot * bs_delta(p.type, spot, p.strike, p.expiry, p.vol, p.rate);
}

bool cholesky_psd(const std::vector<double>& A, int n, std::vector<double>& L) {
    const size_t N = static_cast<size_t>(n);
    L.assign(N * N, 0.0);
    double max_diag = 0.0;
    for (size_t i = 0; i < N; ++i) max_diag = std::max(max_diag, A[i * N + i]);
    const double tol = 1e-12 * std::max(max_diag, 1.0);
    for (size_t j = 0; j < N; ++j) {
        double d = A[j * N + j];
        for (size_t k = 0; k < j; ++k) d -= L[j * N + k] * L[j * N + k];
        if (d < -1e-8 * std::max(max_diag, 1.0)) return false;  // clearly not PSD
        if (d <= tol) {
            // dependent direction: zero column -- valid only if the rest of the column is
            // (numerically) zero too, otherwise A is not PSD (e.g. [[1,1,0],[1,1,.5],[0,.5,1]])
            for (size_t i = j + 1; i < N; ++i) {
                double s = A[i * N + j];
                for (size_t k = 0; k < j; ++k) s -= L[i * N + k] * L[j * N + k];
                if (std::fabs(s) > 1e-7 * std::max(max_diag, 1.0)) return false;
            }
            continue;
        }
        const double ljj = std::sqrt(d);
        L[j * N + j] = ljj;
        for (size_t i = j + 1; i < N; ++i) {
            double s = A[i * N + j];
            for (size_t k = 0; k < j; ++k) s -= L[i * N + k] * L[j * N + k];
            L[i * N + j] = s / ljj;
        }
    }
    return true;
}

MonteCarloRevaluation::MonteCarloRevaluation(int n_scenarios, std::uint64_t seed)
    : m_(std::max(2, n_scenarios + (n_scenarios % 2))), seed_(seed) {}

void MonteCarloRevaluation::simulate(const JointForecast& f) {
    n_ = static_cast<int>(f.sigma.size());
    const size_t n = static_cast<size_t>(n_), M = static_cast<size_t>(m_);
    if (f.mu.size() != n || f.corr.size() != n * n) throw std::invalid_argument("JointForecast: size mismatch");
    std::vector<double> L;
    if (!cholesky_psd(f.corr, n_, L)) throw std::invalid_argument("JointForecast: correlation is not PSD");
    ret_.assign(M * n, 0.0);
    Rng rng(seed_ + 0x9E3779B97F4A7C15ULL * (++draws_));
    const bool t_dist = std::isfinite(f.nu) && f.nu > 2.0;
    std::vector<double> g(n), y(n);
    for (size_t s = 0; s < M; s += 2) {  // antithetic pairs (g, W) and (-g, W)
        for (double& v : g) v = rng.normal();
        double scale = 1.0;
        if (t_dist) scale = std::sqrt((f.nu - 2.0) / (2.0 * rng.gamma(0.5 * f.nu)));
        for (size_t i = 0; i < n; ++i) {
            double acc = 0.0;
            for (size_t k = 0; k <= i; ++k) acc += L[i * n + k] * g[k];
            y[i] = acc * scale;
        }
        for (size_t i = 0; i < n; ++i) {
            ret_[s * n + i] = f.mu[i] + f.sigma[i] * y[i];
            ret_[(s + 1) * n + i] = f.mu[i] - f.sigma[i] * y[i];
        }
    }
}

McRiskResult MonteCarloRevaluation::evaluate(const std::vector<Position>& positions,
                                             const std::vector<double>& spot, double horizon,
                                             const std::vector<double>& confidences) const {
    McRiskResult r;
    const size_t n = static_cast<size_t>(n_), M = static_cast<size_t>(m_);
    if (ret_.empty()) throw std::logic_error("MonteCarloRevaluation: simulate() first");
    std::vector<double> v0(positions.size());
    for (size_t p = 0; p < positions.size(); ++p) {
        const Position& pos = positions[p];
        if (pos.asset < 0 || static_cast<size_t>(pos.asset) >= n) throw std::out_of_range("position asset");
        v0[p] = position_value(pos, spot[static_cast<size_t>(pos.asset)]);
    }
    std::vector<double> loss(M);
    double sum = 0.0, sum2 = 0.0;
    for (size_t s = 0; s < M; ++s) {
        double pnl = 0.0;
        for (size_t p = 0; p < positions.size(); ++p) {
            const Position& pos = positions[p];
            const size_t a = static_cast<size_t>(pos.asset);
            const double s1 = spot[a] * std::exp(ret_[s * n + a]);
            pnl += position_value(pos, s1, horizon) - v0[p];
        }
        loss[s] = -pnl;
        sum += pnl;
        sum2 += pnl * pnl;
    }
    r.scenarios = m_;
    r.mean = sum / static_cast<double>(M);
    r.sigma = std::sqrt(std::max(0.0, sum2 / static_cast<double>(M) - r.mean * r.mean));
    std::sort(loss.begin(), loss.end(), std::greater<double>());  // largest losses first
    for (double c : confidences) {
        double tail = (1.0 - c) * static_cast<double>(M);  // expected tail count
        // (1 - 0.95) * 10000 = 500.00000000000045: snap rounding noise to the integer so the
        // quantile is the 500th largest loss, not the 501st
        if (std::fabs(tail - std::round(tail)) < 1e-9 * std::max(1.0, tail)) tail = std::round(tail);
        const size_t k = static_cast<size_t>(std::floor(tail));
        // empirical quantile: the ceil(tail)-th largest loss
        const size_t var_idx = std::min(M - 1, std::max<size_t>(1, static_cast<size_t>(std::ceil(tail))) - 1);
        double es_sum = 0.0;
        for (size_t j = 0; j < k && j < M; ++j) es_sum += loss[j];
        if (k < M) es_sum += (tail - static_cast<double>(k)) * loss[k];
        r.measures.push_back({c, loss[var_idx], tail > 0.0 ? es_sum / tail : loss[0]});
    }
    return r;
}

}  // namespace garcht
