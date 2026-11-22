#include "ewmar/engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace ewmar {

EwmaEngine::EwmaEngine(size_t assets, const EngineConfig& cfg)
    : n_(assets),
      cfg_(cfg),
      history_(static_cast<size_t>(std::max(cfg.window_bars, 1)), assets),
      on_cap_(static_cast<size_t>(std::max(cfg.overnight_window, 10))),
      Sgap_(assets) {
    if (assets == 0) throw std::invalid_argument("EwmaEngine: no assets");
    if (cfg.bars_per_day < 1) throw std::invalid_argument("EwmaEngine: bars_per_day < 1");
    if (cfg.window_bars < 4 * cfg.bars_per_day) throw std::invalid_argument("EwmaEngine: window shorter than 4 sessions");
    if (!(cfg.ref_lambda > 0.0 && cfg.ref_lambda < 1.0) || !(cfg.lambda > 0.0 && cfg.lambda < 1.0))
        throw std::invalid_argument("EwmaEngine: lambda must be in (0, 1)");
    if (!(cfg.nu > 2.0)) throw std::invalid_argument("EwmaEngine: nu must be > 2");
    if (cfg.confidences.empty()) throw std::invalid_argument("EwmaEngine: no confidence levels");
    models_ = make_models(assets, cfg);
    for (size_t k = 0; k < models_.size(); ++k) {
        info_.push_back(models_[k]->info());
        if (info_.back().main) main_ = k;
        if (!vol_) vol_ = dynamic_cast<const EwmaNormalModel*>(models_[k].get());
        if (!tail_) tail_ = dynamic_cast<const EwmaStudentModel*>(models_[k].get());
    }
    if (!vol_) throw std::logic_error("EwmaEngine: the EWMA-N volatility model is not registered");
    gaps_.assign(on_cap_ * n_, kNaN);
    for (double c : cfg_.confidences) {
        fac_n_.push_back(normal_factors(c));
        fac_on_t_.push_back(student_factors(cfg_.nu, c));
    }
}

// ============================================================== streaming

void EwmaEngine::on_bar(int slot, const double* r) {
    if (slot < 0 || slot >= cfg_.bars_per_day) throw std::out_of_range("on_bar: slot out of range");
    history_.push(slot, r);
    for (auto& m : models_) m->on_bar(slot, r);
}

void EwmaEngine::on_overnight(const double* g) {
    if (!cfg_.overnight) return;
    std::copy(g, g + n_, &gaps_[on_head_ * n_]);
    on_head_ = (on_head_ + 1) % on_cap_;
    if (on_count_ < on_cap_) ++on_count_;
    if (on_ready_) Sgap_.update(g, on_lambda_);
}

double EwmaEngine::lambda() const { return vol_->lambda(); }
double EwmaEngine::nu() const { return tail_ ? tail_->nu() : kNaN; }
const DiurnalProfile& EwmaEngine::diurnal() const { return vol_->diurnal(); }

// ============================================================== refit

RefitReport EwmaEngine::refit(const double* expo) {
    const auto t0 = std::chrono::steady_clock::now();
    RefitReport rep;
    if (history_.size() < history_.capacity()) return rep;  // not a full window yet
    const riskcommon::History window = history_.view();
    for (auto& m : models_) m->refit(window, expo);  // in registration order
    ready_ = true;
    rep.ready = true;
    rep.lambda = lambda();
    rep.nu = nu();
    rep.nu_obs = vol_->standardized_outcomes().size();
    std::vector<double> x(expo, expo + n_);
    bool any = false;
    for (double v : x) any = any || (std::isfinite(v) && v != 0.0);
    if (!any) std::fill(x.begin(), x.end(), 1.0);  // empty book: equal weights
    for (double& v : x)
        if (!std::isfinite(v)) v = 0.0;
    refit_overnight(x.data(), rep);
    rep.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return rep;
}

void EwmaEngine::refit_overnight(const double* x, RefitReport& rep) {
    on_ready_ = false;
    const size_t n = n_, G = on_count_;
    if (!cfg_.overnight || G < 10) return;
    std::vector<double> rows(G * n);
    const size_t start = (on_head_ + on_cap_ - G) % on_cap_;
    for (size_t c = 0; c < G; ++c) {
        const size_t s = (start + c) % on_cap_;
        std::copy(&gaps_[s * n], &gaps_[s * n] + n, &rows[c * n]);
    }
    const size_t burn = std::max<size_t>(5, std::min<size_t>(20, G / 2));
    std::vector<double> v0(n, kNaN);
    for (size_t i = 0; i < n; ++i) {
        double ss = 0.0;
        size_t k = 0;
        for (size_t c = 0; c < burn; ++c)
            if (std::isfinite(rows[c * n + i])) {
                ss += rows[c * n + i] * rows[c * n + i];
                ++k;
            }
        if (k >= 3 && ss > 0.0) v0[i] = ss / static_cast<double>(k);
    }
    on_lambda_ = cfg_.estimate_lambda && G >= 40 ? estimate_lambda(rows, n, burn, G, v0, cfg_.lambda_lo, cfg_.lambda_hi)
                                                 : (cfg_.estimate_lambda ? cfg_.ref_lambda : cfg_.lambda);
    Sgap_.set(initial_covariance(rows, n, 0, burn));
    std::vector<double> u;
    for (size_t c = burn; c < G; ++c) {
        const double* g = &rows[c * n];
        bool complete = true;
        double pnl = 0.0;
        for (size_t i = 0; i < n; ++i) {
            if (x[i] == 0.0) continue;
            if (!std::isfinite(g[i])) complete = false;
            else pnl += x[i] * g[i];
        }
        const double s2 = Sgap_.quad(x);
        if (complete && s2 > 0.0) u.push_back(pnl / std::sqrt(s2));
        Sgap_.update(g, on_lambda_);
    }
    rep.nu_on_obs = u.size();
    const double nu_intraday = std::isfinite(nu()) ? nu() : cfg_.nu;
    on_nu_ = cfg_.estimate_nu && u.size() >= 30 ? fit_nu(u, cfg_.nu_lo, cfg_.nu_hi) : nu_intraday;
    for (size_t k = 0; k < cfg_.confidences.size(); ++k) fac_on_t_[k] = student_factors(on_nu_, cfg_.confidences[k]);
    on_ready_ = true;
    rep.lambda_on = on_lambda_;
    rep.nu_on = on_nu_;
}

// ============================================================== forecasts

void EwmaEngine::compute(int slot, const double* x, PortfolioRisk& out) const {
    if (slot < 0 || slot >= cfg_.bars_per_day) throw std::out_of_range("compute: slot out of range");
    out.ready = ready_;
    out.models.resize(models_.size());
    for (size_t k = 0; k < models_.size(); ++k) models_[k]->forecast(slot, x, out.models[k]);
    if (ready_) {
        vol_->sigma(slot, x, &out.standalone, &out.contrib, &out.asset_sigma);
    } else {
        out.asset_sigma.assign(n_, kNaN);
        out.contrib.assign(n_, kNaN);
        out.standalone.assign(n_, kNaN);
    }
}

OvernightRisk EwmaEngine::compute_overnight(const double* x) const {
    OvernightRisk o;
    const size_t K = cfg_.confidences.size();
    if (!ready_ || !on_ready_) return o;
    const size_t n = n_;
    std::vector<double> xx(n), a0(n);
    for (size_t i = 0; i < n; ++i) {
        xx[i] = std::isfinite(x[i]) ? x[i] : 0.0;
        a0[i] = xx[i] * diurnal().factor(0, i);
    }
    const double vg = std::max(Sgap_.quad(xx.data()), 0.0), vb = std::max(vol_->covariance().quad(a0.data()), 0.0);
    o.available = true;
    o.sigma_gap = std::sqrt(vg);
    o.sigma_bar = std::sqrt(vb);
    o.sigma = std::sqrt(vg + vb);
    o.lambda = on_lambda_;
    o.nu = on_nu_;
    for (size_t k = 0; k < K; ++k) {
        const double c = cfg_.confidences[k];
        o.normal.push_back({c, o.sigma * fac_n_[k].var, o.sigma * fac_n_[k].es});
        o.student.push_back({c, o.sigma * fac_on_t_[k].var, o.sigma * fac_on_t_[k].es});
    }
    return o;
}

std::vector<double> EwmaEngine::correlation() const {
    std::vector<double> R;
    vol_->covariance().correlation(R);
    return R;
}

}  // namespace ewmar
