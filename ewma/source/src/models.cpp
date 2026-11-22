#include "ewmar/models.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "riskcommon/riskmetrics.hpp"

namespace ewmar {

// ============================================================== EWMA-N

EwmaNormalModel::EwmaNormalModel(size_t assets, const ModelConfig& cfg)
    : n_(assets), cfg_(cfg), info_{"EWMA-N", "n_", riskcommon::Family::Normal, {"lambda"}, false},
      lambda_(cfg.lambda), diurnal_(cfg.bars_per_day, assets, cfg.diurnal), Sy_(assets), a_(assets, 0.0),
      Sa_(assets, 0.0) {
    if (!(cfg.lambda > 0.0 && cfg.lambda < 1.0)) throw std::invalid_argument("EWMA-N: lambda must be in (0, 1)");
    for (double c : cfg_.confidences) fac_.push_back(normal_factors(c));
}

void EwmaNormalModel::on_bar(int slot, const double* r) {
    if (!ready_) return;
    for (size_t i = 0; i < n_; ++i) a_[i] = r[i] / diurnal_.factor(slot, i);
    Sy_.update(a_.data(), lambda_);
}

void EwmaNormalModel::refit(const riskcommon::History& h, const double* expo) {
    const size_t n = n_, W = h.bars;
    if (W < 4 || h.assets != n) return;
    const std::vector<double> R(h.returns, h.returns + W * n);
    const std::vector<int> slots(h.slots, h.slots + W);

    // 1. diurnal profile and deseasonalized returns
    diurnal_.estimate(R, slots);
    std::vector<double> Y(W * n);
    for (size_t t = 0; t < W; ++t)
        for (size_t i = 0; i < n; ++i) Y[t * n + i] = R[t * n + i] / diurnal_.factor(slots[t], i);

    // 2. burn-in (first quarter of the window): starting covariance, not scored
    const size_t burn = std::min(W / 2, std::max(W / 4, n + 10));
    std::vector<double> v0(n, kNaN);
    for (size_t i = 0; i < n; ++i) {
        double ss = 0.0;
        size_t k = 0;
        for (size_t t = 0; t < burn; ++t)
            if (std::isfinite(Y[t * n + i])) {
                ss += Y[t * n + i] * Y[t * n + i];
                ++k;
            }
        if (k >= 10 && ss > 0.0) v0[i] = ss / static_cast<double>(k);
    }

    // 3. lambda by predictive quasi-likelihood
    lambda_ = cfg_.estimate_lambda ? estimate_lambda(Y, n, burn, W, v0, cfg_.lambda_lo, cfg_.lambda_hi) : cfg_.lambda;

    // 4. filter over the window; standardized one-step-ahead outcomes of the current portfolio
    std::vector<double> x(expo, expo + n);
    bool any = false;
    for (double v : x) any = any || (std::isfinite(v) && v != 0.0);
    if (!any) std::fill(x.begin(), x.end(), 1.0);  // empty book: equal weights
    for (double& v : x)
        if (!std::isfinite(v)) v = 0.0;
    Sy_.set(initial_covariance(Y, n, 0, burn));
    u_.clear();
    u_.reserve(W - burn);
    std::vector<double> a(n);
    for (size_t t = burn; t < W; ++t) {
        const double* r = &R[t * n];
        bool complete = true;
        double pnl = 0.0;
        for (size_t i = 0; i < n; ++i) {
            if (!std::isfinite(r[i])) {
                if (x[i] != 0.0) complete = false;
                continue;
            }
            pnl += x[i] * r[i];
        }
        if (complete) {
            for (size_t i = 0; i < n; ++i) a[i] = x[i] * diurnal_.factor(slots[t], i);
            const double s2 = Sy_.quad(a.data());
            if (s2 > 0.0) u_.push_back(pnl / std::sqrt(s2));
        }
        Sy_.update(&Y[t * n], lambda_);
    }
    ready_ = true;
}

double EwmaNormalModel::sigma(int slot, const double* x, std::vector<double>* standalone, std::vector<double>* contrib,
                              std::vector<double>* asset_sigma) const {
    const size_t n = n_;
    const std::vector<double>& S = Sy_.matrix();
    for (size_t i = 0; i < n; ++i) a_[i] = (std::isfinite(x[i]) ? x[i] : 0.0) * diurnal_.factor(slot, i);
    double var = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double acc = 0.0;
        for (size_t j = 0; j < n; ++j) acc += S[i * n + j] * a_[j];
        Sa_[i] = acc;
        var += a_[i] * acc;
    }
    const double sig = std::sqrt(std::max(var, 0.0));
    if (standalone && contrib && asset_sigma) {
        standalone->assign(n, kNaN);
        contrib->assign(n, kNaN);
        asset_sigma->assign(n, kNaN);
        for (size_t i = 0; i < n; ++i) {
            const double sd = std::sqrt(std::max(S[i * n + i], 0.0));
            (*asset_sigma)[i] = sd * diurnal_.factor(slot, i);
            (*standalone)[i] = std::fabs(a_[i]) * sd;
            (*contrib)[i] = sig > 0.0 ? a_[i] * Sa_[i] / sig : 0.0;
        }
    }
    return sig;
}

void EwmaNormalModel::forecast(int slot, const double* x, riskcommon::ModelForecast& out) const {
    const size_t K = cfg_.confidences.size();
    out.mean = 0.0;
    out.shape = riskcommon::Shape{};
    out.parameters.assign(1, lambda_);
    out.measures.resize(K);
    out.sigma = ready_ ? sigma(slot, x) : kNaN;
    for (size_t k = 0; k < K; ++k)
        out.measures[k] = {cfg_.confidences[k], out.sigma * fac_[k].var, out.sigma * fac_[k].es};
}

// ============================================================== EWMA-t

EwmaStudentModel::EwmaStudentModel(const EwmaNormalModel& base, const ModelConfig& cfg)
    : base_(base), cfg_(cfg), info_{"EWMA-t", "t_", riskcommon::Family::StudentT, {}, true}, nu_(cfg.nu) {
    if (!(cfg.nu > 2.0)) throw std::invalid_argument("EWMA-t: nu must be > 2");
    for (double c : cfg_.confidences) fac_.push_back(student_factors(nu_, c));
}

void EwmaStudentModel::refit(const riskcommon::History&, const double*) {
    const std::vector<double>& u = base_.standardized_outcomes();
    if (!cfg_.estimate_nu) nu_ = cfg_.nu;
    else if (u.size() >= 100) nu_ = fit_nu(u, cfg_.nu_lo, cfg_.nu_hi);
    for (size_t k = 0; k < cfg_.confidences.size(); ++k) fac_[k] = student_factors(nu_, cfg_.confidences[k]);
}

void EwmaStudentModel::forecast(int slot, const double* x, riskcommon::ModelForecast& out) const {
    const size_t K = cfg_.confidences.size();
    out.mean = 0.0;
    out.shape = riskcommon::Shape{};
    out.shape.nu = nu_;
    out.parameters.clear();
    out.measures.resize(K);
    out.sigma = base_.ready() ? base_.sigma(slot, x) : kNaN;
    for (size_t k = 0; k < K; ++k)
        out.measures[k] = {cfg_.confidences[k], out.sigma * fac_[k].var, out.sigma * fac_[k].es};
}

// ============================================================== registration

std::vector<std::unique_ptr<riskcommon::PortfolioModel>> make_models(size_t assets, const ModelConfig& cfg) {
    std::vector<std::unique_ptr<riskcommon::PortfolioModel>> m;
    m.push_back(std::make_unique<riskcommon::RiskMetricsModel>(assets, cfg.confidences, cfg.ref_lambda));
    auto normal = std::make_unique<EwmaNormalModel>(assets, cfg);
    auto student = std::make_unique<EwmaStudentModel>(*normal, cfg);
    m.push_back(std::move(normal));
    m.push_back(std::move(student));
    return m;
}

}  // namespace ewmar
