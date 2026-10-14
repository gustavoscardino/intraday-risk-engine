#include "ewmar/engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace ewmar {

EwmaEngine::EwmaEngine(size_t assets, const EngineConfig& cfg)
    : n_(assets),
      cfg_(cfg),
      lambda_(cfg.lambda),
      nu_(cfg.nu),
      diurnal_(cfg.bars_per_day, assets, cfg.diurnal),
      Sy_(assets),
      Sref_(assets),
      cap_(static_cast<size_t>(std::max(cfg.window_bars, 1))),
      on_cap_(static_cast<size_t>(std::max(cfg.overnight_window, 10))),
      Sgap_(assets) {
    if (assets == 0) throw std::invalid_argument("EwmaEngine: no assets");
    if (cfg.bars_per_day < 1) throw std::invalid_argument("EwmaEngine: bars_per_day < 1");
    if (cfg.window_bars < 4 * cfg.bars_per_day) throw std::invalid_argument("EwmaEngine: window shorter than 4 sessions");
    if (!(cfg.ref_lambda > 0.0 && cfg.ref_lambda < 1.0) || !(cfg.lambda > 0.0 && cfg.lambda < 1.0))
        throw std::invalid_argument("EwmaEngine: lambda must be in (0, 1)");
    if (!(cfg.nu > 2.0)) throw std::invalid_argument("EwmaEngine: nu must be > 2");
    if (cfg.confidences.empty()) throw std::invalid_argument("EwmaEngine: no confidence levels");
    hist_.assign(cap_ * n_, kNaN);
    hist_slot_.assign(cap_, 0);
    gaps_.assign(on_cap_ * n_, kNaN);
    a_.assign(n_, 0.0);
    Sa_.assign(n_, 0.0);
    for (double c : cfg_.confidences) {
        fac_n_.push_back(normal_factors(c));
        fac_t_.push_back(student_factors(nu_, c));
        fac_on_t_.push_back(student_factors(nu_, c));
    }
}

// ============================================================== streaming

void EwmaEngine::on_bar(int slot, const double* r) {
    if (slot < 0 || slot >= cfg_.bars_per_day) throw std::out_of_range("on_bar: slot out of range");
    const size_t n = n_;
    // undo point
    undo_count_ = count_;
    undo_slot_ = hist_slot_[head_];
    undo_row_.assign(&hist_[head_ * n], &hist_[head_ * n] + n);
    if (ready_) {
        undo_Sy_ = Sy_.matrix();
        undo_Sref_ = Sref_.matrix();
    }
    can_undo_ = true;

    std::copy(r, r + n, &hist_[head_ * n]);
    hist_slot_[head_] = slot;
    head_ = (head_ + 1) % cap_;
    if (count_ < cap_) ++count_;
    if (!ready_) return;
    for (size_t i = 0; i < n; ++i) a_[i] = r[i] / diurnal_.factor(slot, i);
    Sy_.update(a_.data(), lambda_);
    Sref_.update(r, cfg_.ref_lambda);
}

bool EwmaEngine::undo_last_bar() {
    if (!can_undo_) return false;
    const size_t n = n_;
    head_ = (head_ + cap_ - 1) % cap_;
    std::copy(undo_row_.begin(), undo_row_.end(), &hist_[head_ * n]);
    hist_slot_[head_] = undo_slot_;
    count_ = undo_count_;
    if (ready_) {
        Sy_.set(undo_Sy_);
        Sref_.set(undo_Sref_);
    }
    can_undo_ = false;
    return true;
}

void EwmaEngine::amend_last_bar(size_t asset, double value) {
    if (count_ == 0 || asset >= n_) return;
    hist_[((head_ + cap_ - 1) % cap_) * n_ + asset] = value;
}

void EwmaEngine::on_overnight(const double* g) {
    if (!cfg_.overnight) return;
    std::copy(g, g + n_, &gaps_[on_head_ * n_]);
    on_head_ = (on_head_ + 1) % on_cap_;
    if (on_count_ < on_cap_) ++on_count_;
    if (on_ready_) Sgap_.update(g, on_lambda_);
}

// ============================================================== refit

void EwmaEngine::chronological(std::vector<double>& rows, std::vector<int>& slots) const {
    const size_t n = n_, start = (head_ + cap_ - count_) % cap_;
    rows.resize(count_ * n);
    slots.resize(count_);
    for (size_t c = 0; c < count_; ++c) {
        const size_t s = (start + c) % cap_;
        std::copy(&hist_[s * n], &hist_[s * n] + n, &rows[c * n]);
        slots[c] = hist_slot_[s];
    }
}

RefitReport EwmaEngine::refit(const double* expo) {
    const auto t0 = std::chrono::steady_clock::now();
    RefitReport rep;
    if (count_ < cap_) return rep;  // not a full window yet
    const size_t n = n_;
    std::vector<double> R, Y;
    std::vector<int> slots;
    chronological(R, slots);
    const size_t W = slots.size();

    // 1. diurnal profile and deseasonalized returns
    diurnal_.estimate(R, slots);
    Y.resize(W * n);
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

    // 4. filter over the window; standardized one-step-ahead portfolio outcomes for nu
    std::vector<double> x(expo, expo + n);
    bool any = false;
    for (double v : x) any = any || (std::isfinite(v) && v != 0.0);
    if (!any) std::fill(x.begin(), x.end(), 1.0);  // empty book: equal weights
    for (double& v : x)
        if (!std::isfinite(v)) v = 0.0;
    Sy_.set(initial_covariance(Y, n, 0, burn));
    std::vector<double> u;
    u.reserve(W - burn);
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
            if (s2 > 0.0) u.push_back(pnl / std::sqrt(s2));
        }
        Sy_.update(&Y[t * n], lambda_);
    }
    rep.nu_obs = u.size();
    if (!cfg_.estimate_nu) nu_ = cfg_.nu;
    else if (u.size() >= 100) nu_ = fit_nu(u, cfg_.nu_lo, cfg_.nu_hi);

    // 5. EWMA-N (ref): raw returns, fixed lambda, same burn-in
    Sref_.set(initial_covariance(R, n, 0, burn));
    for (size_t t = burn; t < W; ++t) Sref_.update(&R[t * n], cfg_.ref_lambda);

    for (size_t k = 0; k < cfg_.confidences.size(); ++k) fac_t_[k] = student_factors(nu_, cfg_.confidences[k]);
    ready_ = true;
    can_undo_ = false;
    rep.ready = true;
    rep.lambda = lambda_;
    rep.nu = nu_;
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
    on_nu_ = cfg_.estimate_nu && u.size() >= 30 ? fit_nu(u, cfg_.nu_lo, cfg_.nu_hi) : nu_;
    for (size_t k = 0; k < cfg_.confidences.size(); ++k) fac_on_t_[k] = student_factors(on_nu_, cfg_.confidences[k]);
    on_ready_ = true;
    rep.lambda_on = on_lambda_;
    rep.nu_on = on_nu_;
}

// ============================================================== forecasts

void EwmaEngine::compute(int slot, const double* x, PortfolioRisk& out) const {
    const size_t n = n_, K = cfg_.confidences.size();
    out.ready = ready_;
    out.lambda = lambda_;
    out.nu = nu_;
    out.asset_sigma.assign(n, kNaN);
    out.contrib.assign(n, kNaN);
    out.standalone.assign(n, kNaN);
    for (ModelForecast* m : {&out.ref, &out.normal, &out.student}) {
        m->sigma = kNaN;
        m->m.assign(K, RiskMeasure{});
        for (size_t k = 0; k < K; ++k) m->m[k].confidence = cfg_.confidences[k];
    }
    if (!ready_) return;
    if (slot < 0 || slot >= cfg_.bars_per_day) throw std::out_of_range("compute: slot out of range");
    const std::vector<double>& S = Sy_.matrix();
    for (size_t i = 0; i < n; ++i) a_[i] = (std::isfinite(x[i]) ? x[i] : 0.0) * diurnal_.factor(slot, i);
    double var = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double acc = 0.0;
        for (size_t j = 0; j < n; ++j) acc += S[i * n + j] * a_[j];
        Sa_[i] = acc;
        var += a_[i] * acc;
    }
    const double sigma = std::sqrt(std::max(var, 0.0));
    for (size_t i = 0; i < n; ++i) {
        const double sd = std::sqrt(std::max(S[i * n + i], 0.0));
        out.asset_sigma[i] = sd * diurnal_.factor(slot, i);
        out.standalone[i] = std::fabs(a_[i]) * sd;
        out.contrib[i] = sigma > 0.0 ? a_[i] * Sa_[i] / sigma : 0.0;
    }
    for (size_t i = 0; i < n; ++i) a_[i] = std::isfinite(x[i]) ? x[i] : 0.0;
    const double sigma_ref = std::sqrt(std::max(Sref_.quad(a_.data()), 0.0));
    out.ref.sigma = sigma_ref;
    out.normal.sigma = out.student.sigma = sigma;
    for (size_t k = 0; k < K; ++k) {
        out.ref.m[k].var = sigma_ref * fac_n_[k].var;
        out.ref.m[k].es = sigma_ref * fac_n_[k].es;
        out.normal.m[k].var = sigma * fac_n_[k].var;
        out.normal.m[k].es = sigma * fac_n_[k].es;
        out.student.m[k].var = sigma * fac_t_[k].var;
        out.student.m[k].es = sigma * fac_t_[k].es;
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
        a0[i] = xx[i] * diurnal_.factor(0, i);
    }
    const double vg = std::max(Sgap_.quad(xx.data()), 0.0), vb = std::max(Sy_.quad(a0.data()), 0.0);
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

double EwmaEngine::asset_sigma(size_t i, int slot) const {
    if (!ready_ || i >= n_ || slot < 0 || slot >= cfg_.bars_per_day) return kNaN;
    return std::sqrt(std::max(Sy_(i, i), 0.0)) * diurnal_.factor(slot, i);
}

std::vector<double> EwmaEngine::correlation() const {
    std::vector<double> R;
    Sy_.correlation(R);
    return R;
}

}  // namespace ewmar
