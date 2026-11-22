#include "garcht/portfolio.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "garcht/math.hpp"
#include "garcht/optimize.hpp"

namespace garcht {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

template <class F>
void parallel_for(int n, int threads, F&& f) {
    threads = std::min(threads, n);
    if (threads <= 1) {
        for (int i = 0; i < n; ++i) f(i);
        return;
    }
    std::atomic<int> next{0};
    std::exception_ptr error;  // first exception of any worker, rethrown after the join
    std::mutex error_mtx;
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(threads));
    for (int t = 0; t < threads; ++t)
        pool.emplace_back([&] {
            try {
                for (int i; (i = next.fetch_add(1)) < n;) f(i);
            } catch (...) {
                next = n;  // stop the other workers
                std::lock_guard<std::mutex> g(error_mtx);
                if (!error) error = std::current_exception();
            }
        });
    for (auto& th : pool) th.join();
    if (error) std::rethrow_exception(error);
}

int resolve_threads(int requested) {
    if (requested > 0) return requested;
    const unsigned hw = std::thread::hardware_concurrency();
    return hw > 0 ? static_cast<int>(hw) : 1;
}

double clip(double z, double c) { return c > 0.0 ? std::clamp(z, -c, c) : z; }

// GARCH variance path with the same winsorization rule as GarchFilter::update(y, z_clip):
// eps^2 is capped at z_clip^2 h_t inside the recursion. Returns h_{T} (next variance).
double variance_path_clipped(const std::vector<double>& y, const GarchParams& p, double bc, double zc,
                             double* h_out) {
    double h = p.omega + (p.alpha + p.beta) * bc;
    for (size_t t = 0; t < y.size(); ++t) {
        h_out[t] = h;
        const double e = y[t] - p.mu;
        double e2 = std::isfinite(e) ? e * e : h;
        if (zc > 0.0 && e2 > zc * zc * h) e2 = zc * zc * h;
        h = p.omega + p.alpha * e2 + p.beta * h;
    }
    return h;
}

double logistic(double x) {
    if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
    const double e = std::exp(x);
    return e / (1.0 + e);
}

// Student-t excess kurtosis (infinite for nu <= 4).
double t_excess_kurtosis(double nu) {
    if (!std::isfinite(nu)) return 0.0;
    return nu > 4.0 ? 6.0 / (nu - 4.0) : kInf;
}

}  // namespace

const char* to_string(ModelSource s) {
    switch (s) {
        case ModelSource::Fitted: return "fitted";
        case ModelSource::PreviousFit: return "previous_fit";
        case ModelSource::Fallback: return "fallback";
    }
    return "unknown";
}

// ============================================================== construction / accessors

IntradayRiskEngine::IntradayRiskEngine(std::vector<std::string> tickers, EngineConfig cfg)
    : n_(static_cast<int>(tickers.size())), cfg_(std::move(cfg)) {
    if (n_ == 0) throw std::invalid_argument("IntradayRiskEngine: no assets");
    if (cfg_.bars_per_day <= 0) throw std::invalid_argument("bars_per_day must be > 0");
    if (cfg_.window_bars < 10) throw std::invalid_argument("window_bars too small");
    if (!(cfg_.corr_lambda > 0.0 && cfg_.corr_lambda < 1.0))
        throw std::invalid_argument("corr_lambda must be in (0, 1)");
    if (!(cfg_.ewma_lambda > 0.0 && cfg_.ewma_lambda < 1.0))
        throw std::invalid_argument("ewma_lambda must be in (0, 1)");
    if (!(cfg_.overnight_lambda > 0.0 && cfg_.overnight_lambda < 1.0))
        throw std::invalid_argument("overnight_lambda must be in (0, 1)");
    if (!(cfg_.corr_shrinkage >= 0.0 && cfg_.corr_shrinkage <= 1.0))
        throw std::invalid_argument("corr_shrinkage must be in [0, 1]");
    if (!(cfg_.min_active_share > 0.0 && cfg_.min_active_share <= 1.0))
        throw std::invalid_argument("min_active_share must be in (0, 1]");
    if (!(cfg_.dcc_a >= 0.0 && cfg_.dcc_b >= 0.0 && cfg_.dcc_a + cfg_.dcc_b < 1.0))
        throw std::invalid_argument("dcc_a, dcc_b must be >= 0 with a + b < 1");
    for (double c : cfg_.confidences)
        if (!(c > 0.0 && c < 1.0)) throw std::invalid_argument("confidence must be in (0, 1)");
    cfg_.min_fit_bars = std::clamp(cfg_.min_fit_bars, 2, cfg_.window_bars);
    cfg_.overnight_window = std::max(cfg_.overnight_window, 10);
    cfg_.dcc_max_pairs = std::max(cfg_.dcc_max_pairs, 1);

    const size_t n = static_cast<size_t>(n_);
    cap_ = static_cast<size_t>(cfg_.window_bars);
    ret_.assign(cap_ * n, kNaN);
    slot_.assign(cap_, 0);
    assets_.resize(n);
    for (size_t i = 0; i < n; ++i) assets_[i].ticker = std::move(tickers[i]);
    diurnal_ = DiurnalProfile(cfg_.bars_per_day);
    for (AssetModel& a : assets_) a.diurnal = diurnal_;
    R_.assign(n * n, 0.0);
    for (size_t i = 0; i < n; ++i) R_[i * n + i] = 1.0;
    Q_ = R0_ = Rinit_ = on_corr_ = R_;
    if (cfg_.correlation == CorrelationMode::Ewma) {
        dcc_a_ = 1.0 - cfg_.corr_lambda;
        dcc_b_ = cfg_.corr_lambda;
    } else {
        dcc_a_ = cfg_.dcc_a;
        dcc_b_ = cfg_.dcc_b;
    }
    z_.assign(n, kNaN);
    stale_var_.assign(n, 0.0);
    seen_.assign(n, 0);
    inv_sd_.assign(n, 0.0);
    gap_hist_.assign(static_cast<size_t>(cfg_.overnight_window) * n, kNaN);
    corr_scale_.assign(static_cast<size_t>(cfg_.bars_per_day), 1.0);
    on_var_.assign(n, kNaN);
    on_proxy_.assign(n, 0);
    on_core_.assign(n, 0);
    normal_tail_.set(Dist::Normal, kInf, cfg_.confidences);
    nu_p_ = cfg_.spec.dist == Dist::StudentT ? kNaN : kInf;
    on_lambda_ = cfg_.overnight_lambda;
}

bool IntradayRiskEngine::ready() const {
    std::shared_lock<std::shared_mutex> lk(mtx_);
    return ready_;
}

size_t IntradayRiskEngine::history_size() const {
    std::shared_lock<std::shared_mutex> lk(mtx_);
    return count_;
}

AssetModel IntradayRiskEngine::asset(int i) const {
    std::shared_lock<std::shared_mutex> lk(mtx_);
    return assets_.at(static_cast<size_t>(i));
}

std::vector<double> IntradayRiskEngine::correlation() const {
    std::shared_lock<std::shared_mutex> lk(mtx_);
    return R_;
}

DiurnalProfile IntradayRiskEngine::diurnal() const {
    std::shared_lock<std::shared_mutex> lk(mtx_);
    return diurnal_;
}

DiurnalProfile IntradayRiskEngine::diurnal(int asset) const {
    std::shared_lock<std::shared_mutex> lk(mtx_);
    return assets_.at(static_cast<size_t>(asset)).diurnal;
}

double IntradayRiskEngine::portfolio_nu() const {
    std::shared_lock<std::shared_mutex> lk(mtx_);
    return nu_p_;
}

double IntradayRiskEngine::dcc_a() const {
    std::shared_lock<std::shared_mutex> lk(mtx_);
    return dcc_a_;
}

double IntradayRiskEngine::dcc_b() const {
    std::shared_lock<std::shared_mutex> lk(mtx_);
    return dcc_b_;
}

std::vector<double> IntradayRiskEngine::slot_correlation_scale() const {
    std::shared_lock<std::shared_mutex> lk(mtx_);
    return corr_scale_;
}

double IntradayRiskEngine::overnight_lambda() const {
    std::shared_lock<std::shared_mutex> lk(mtx_);
    return on_lambda_;
}

size_t IntradayRiskEngine::oos_outcomes() const {
    std::shared_lock<std::shared_mutex> lk(mtx_);
    return oos_u_.size();
}

// ============================================================== correlation helpers

size_t IntradayRiskEngine::idx(size_t chrono) const {
    return (head_ + cap_ - count_ + chrono) % cap_;
}

bool IntradayRiskEngine::complete_row(const double* v) const {
    for (size_t i = 0; i < static_cast<size_t>(n_); ++i)
        if (assets_[i].active && !std::isfinite(v[i])) return false;
    return true;
}

bool IntradayRiskEngine::is_auction_slot(int slot) const {
    return cfg_.auction_correlation && cfg_.bars_per_day >= 3 && (slot == 0 || slot == cfg_.bars_per_day - 1);
}

void IntradayRiskEngine::shrink_corr(std::vector<double>& R) const {
    if (!(cfg_.corr_shrinkage > 0.0)) return;
    const size_t n = static_cast<size_t>(n_);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            if (i != j) R[i * n + j] *= 1.0 - cfg_.corr_shrinkage;
}

// E[v v' | observed entries] for a zero-mean vector with covariance C (read as a correlation
// when as_corr), over the active assets: missing entries are imputed by their regression on
// the observed ones plus the conditional covariance C_mm - C_mo C_oo^-1 C_om (the Gaussian
// projection of the EM algorithm). Used in place of v v' on bars where an active asset did not
// trade: the result is PSD, so the recursions stay PSD, the observed pairs keep learning and a
// halted asset's correlations persist in expectation. Inactive entries of `out` are not
// written. Returns false when no active asset is observed (nothing to learn).
bool IntradayRiskEngine::conditional_outer(const std::vector<double>& C, bool as_corr, const double* v,
                                           std::vector<double>& out) const {
    const size_t n = static_cast<size_t>(n_);
    std::vector<size_t> o, m;
    for (size_t i = 0; i < n; ++i)
        if (assets_[i].active) (std::isfinite(v[i]) ? o : m).push_back(i);
    if (o.empty()) return false;
    std::vector<double> sd(n, 1.0);
    if (as_corr)
        for (size_t i = 0; i < n; ++i) sd[i] = C[i * n + i] > 0.0 ? std::sqrt(C[i * n + i]) : 1.0;
    auto cov = [&](size_t i, size_t j) { return (as_corr && i == j) ? 1.0 : C[i * n + j] / (sd[i] * sd[j]); };
    const size_t no = o.size(), nm = m.size(), w = 1 + nm;

    // Cholesky of C_oo (a tiny ridge if it is numerically singular)
    double diag = 0.0;
    for (size_t r = 0; r < no; ++r) diag = std::max(diag, cov(o[r], o[r]));
    std::vector<double> L(no * no, 0.0);
    bool ok = false;
    for (double ridge : {0.0, 1e-10, 1e-6}) {
        ok = diag > 0.0;
        for (size_t r = 0; r < no && ok; ++r)
            for (size_t k = 0; k <= r; ++k) {
                double s = cov(o[r], o[k]) + (r == k ? ridge * diag : 0.0);
                for (size_t l = 0; l < k; ++l) s -= L[r * no + l] * L[k * no + l];
                if (r == k) {
                    if (!(s > 0.0)) {
                        ok = false;
                        break;
                    }
                    L[r * no + r] = std::sqrt(s);
                } else {
                    L[r * no + k] = s / L[k * no + k];
                }
            }
        if (ok) break;
    }
    // X = C_oo^-1 [v_o | C_om]
    std::vector<double> X(no * w, 0.0);
    if (ok) {
        for (size_t r = 0; r < no; ++r) {
            X[r * w] = v[o[r]];
            for (size_t k = 0; k < nm; ++k) X[r * w + 1 + k] = cov(o[r], m[k]);
        }
        for (size_t col = 0; col < w; ++col) {
            for (size_t r = 0; r < no; ++r) {
                double s = X[r * w + col];
                for (size_t l = 0; l < r; ++l) s -= L[r * no + l] * X[l * w + col];
                X[r * w + col] = s / L[r * no + r];
            }
            for (size_t r = no; r-- > 0;) {
                double s = X[r * w + col];
                for (size_t l = r + 1; l < no; ++l) s -= L[l * no + r] * X[l * w + col];
                X[r * w + col] = s / L[r * no + r];
            }
        }
    }
    // without a usable C_oo: missing entries independent of the observed ones (still PSD)
    std::vector<double> vh(nm, 0.0);
    if (ok)
        for (size_t k = 0; k < nm; ++k)
            for (size_t r = 0; r < no; ++r) vh[k] += cov(m[k], o[r]) * X[r * w];
    for (size_t r = 0; r < no; ++r) {
        for (size_t s = 0; s < no; ++s) out[o[r] * n + o[s]] = v[o[r]] * v[o[s]];
        for (size_t k = 0; k < nm; ++k) out[o[r] * n + m[k]] = out[m[k] * n + o[r]] = v[o[r]] * vh[k];
    }
    for (size_t k = 0; k < nm; ++k)
        for (size_t l = 0; l < nm; ++l) {
            double c = cov(m[k], m[l]);
            if (ok)
                for (size_t r = 0; r < no; ++r) c -= cov(m[k], o[r]) * X[r * w + 1 + l];
            out[m[k] * n + m[l]] = vh[k] * vh[l] + c;
        }
    return true;
}

// Q <- w * R0 + a * z z' + b * Q over active pairs (EWMA: w = 0, a = 1 - lambda, b = lambda).
// Missing active entries: z z' replaced by its conditional expectation given the observed z.
void IntradayRiskEngine::corr_update(std::vector<double>& Q, const double* z) const {
    const size_t n = static_cast<size_t>(n_);
    const double a = dcc_a_, b = dcc_b_;
    const double w = cfg_.correlation == CorrelationMode::Ewma ? 0.0 : 1.0 - a - b;
    const double c = cfg_.z_clip;
    if (!complete_row(z)) {
        std::vector<double> zc(n), zz(n * n, 0.0);
        for (size_t i = 0; i < n; ++i) zc[i] = std::isfinite(z[i]) ? clip(z[i], c) : kNaN;
        if (!conditional_outer(Q, true, zc.data(), zz)) return;
        for (size_t i = 0; i < n; ++i) {
            if (!assets_[i].active) continue;
            for (size_t j = 0; j < n; ++j)
                if (assets_[j].active) Q[i * n + j] = w * R0_[i * n + j] + a * zz[i * n + j] + b * Q[i * n + j];
        }
        return;
    }
    for (size_t i = 0; i < n; ++i) {
        if (!assets_[i].active) continue;
        const double zi = a * clip(z[i], c);
        double* row = &Q[i * n];
        const double* target = &R0_[i * n];
        for (size_t j = 0; j < n; ++j)
            if (assets_[j].active) row[j] = w * target[j] + zi * clip(z[j], c) + b * row[j];
    }
}

void IntradayRiskEngine::normalize_corr(const std::vector<double>& Q, std::vector<double>& R) const {
    const size_t n = static_cast<size_t>(n_);
    double* inv_sd = inv_sd_.data();  // R_ij = Q_ij / sqrt(Q_ii Q_jj): n sqrt + n^2 multiplies
    for (size_t i = 0; i < n; ++i) {
        const double d = Q[i * n + i];
        inv_sd[i] = d > 0.0 ? 1.0 / std::sqrt(d) : 0.0;
    }
    for (size_t i = 0; i < n; ++i) {
        const double* q = &Q[i * n];
        double* r = &R[i * n];
        const double di = inv_sd[i];
        for (size_t j = 0; j < n; ++j) r[j] = q[j] * di * inv_sd[j];
        r[i] = 1.0;
    }
}

// ============================================================== streaming updates

void IntradayRiskEngine::on_bar(int slot, const double* returns) {
    if (slot < 0 || slot >= cfg_.bars_per_day) throw std::out_of_range("on_bar: slot out of range");
    std::unique_lock<std::shared_mutex> lk(mtx_);
    const size_t n = static_cast<size_t>(n_);
    std::copy(returns, returns + n, &ret_[head_ * n]);
    slot_[head_] = slot;
    head_ = (head_ + 1) % cap_;
    if (count_ < cap_) ++count_;
    if (slot <= last_slot_) {  // a new session: P&L is referenced to the open again
        std::fill(seen_.begin(), seen_.end(), char(0));
        std::fill(stale_var_.begin(), stale_var_.end(), 0.0);
    }
    last_slot_ = slot;
    if (!ready_) {
        for (size_t i = 0; i < n; ++i)
            if (std::isfinite(returns[i])) seen_[i] = 1;
        return;
    }

    for (size_t i = 0; i < n; ++i) {
        const double s = assets_[i].diurnal[slot];
        if (std::isfinite(returns[i])) {
            stale_var_[i] = 0.0;
            seen_[i] = 1;
        } else if (seen_[i]) {  // missed bar after a print: marked P&L keeps accumulating it
            stale_var_[i] += s * s * assets_[i].filter.variance();
        }
        const double z = assets_[i].filter.update(returns[i] / s, cfg_.z_clip);
        assets_[i].last_z = z;
        z_[i] = z;
    }
    if (cfg_.correlation != CorrelationMode::Constant && !is_auction_slot(slot)) {
        corr_update(Q_, z_.data());  // missing active entries imputed (halts)
        normalize_corr(Q_, R_);
        shrink_corr(R_);
    }
}

void IntradayRiskEngine::observe_outcome(double pnl, const PortfolioRisk& f) {
    if (!std::isfinite(pnl) || !(f.sigma > 0.0) || !std::isfinite(f.mean)) return;
    std::unique_lock<std::shared_mutex> lk(mtx_);
    oos_u_.push_back((pnl - f.mean) / f.sigma);
    while (oos_u_.size() > cap_) oos_u_.pop_front();
}

void IntradayRiskEngine::on_overnight(const double* gaps) {
    std::unique_lock<std::shared_mutex> lk(mtx_);
    const size_t n = static_cast<size_t>(n_);
    const size_t cap = static_cast<size_t>(cfg_.overnight_window);
    std::copy(gaps, gaps + n, &gap_hist_[gap_head_ * n]);
    gap_head_ = (gap_head_ + 1) % cap;
    if (gap_count_ < cap) ++gap_count_;
    if (!on_ready_) {  // first estimate as soon as there is enough history
        update_overnight_model();
        return;
    }
    // O(n) streaming update of the variances; correlation and tail are re-estimated at refit()
    const double lam = on_lambda_;
    for (size_t i = 0; i < n; ++i)
        if (std::isfinite(gaps[i]) && std::isfinite(on_var_[i]) && !on_proxy_[i])
            on_var_[i] = lam * on_var_[i] + (1.0 - lam) * gaps[i] * gaps[i];
}

void IntradayRiskEngine::update_overnight_model() {
    const size_t n = static_cast<size_t>(n_);
    const size_t cap = static_cast<size_t>(cfg_.overnight_window);
    const size_t G = gap_count_;
    on_ready_ = false;
    std::fill(on_core_.begin(), on_core_.end(), char(0));
    if (G < 10) return;
    auto row = [&](size_t chrono) { return &gap_hist_[((gap_head_ + cap - G + chrono) % cap) * n]; };

    // EWMA variance per asset; standardized gaps use the variance known before each gap.
    // Initial variance: mean square of the first (up to) 20 gaps, which are not scored.
    std::vector<double> v0(n, kNaN);
    std::vector<size_t> first_scored(n, G);
    for (size_t i = 0; i < n; ++i) {
        double ss = 0.0;
        int k = 0;
        size_t c = 0;
        for (; c < G && k < 20; ++c)
            if (std::isfinite(row(c)[i])) {
                ss += row(c)[i] * row(c)[i];
                ++k;
            }
        if (k >= 5) {
            v0[i] = std::max(ss / k, 1e-14);
            first_scored[i] = c;
        }
    }
    // lambda by predictive Gaussian quasi-likelihood, pooled over assets (honest: every gap is
    // scored with the variance known before it)
    auto qll = [&](double lam) {
        double acc = 0.0;
        for (size_t i = 0; i < n; ++i) {
            if (!std::isfinite(v0[i])) continue;
            double v = v0[i];
            for (size_t c = 0; c < G; ++c) {
                const double g = row(c)[i];
                if (!std::isfinite(g)) continue;
                if (c >= first_scored[i]) acc += std::log(v) + g * g / v;
                v = lam * v + (1.0 - lam) * g * g;
            }
        }
        return acc;
    };
    if (cfg_.estimate_overnight_lambda && G >= 40)
        on_lambda_ = math::brent_minimize(qll, 0.80, 0.999, 1e-6);
    const double lam = on_lambda_;
    std::vector<double> zs(G * n, kNaN);
    on_gaps_.assign(G * n, kNaN);
    on_sd_.assign(G * n, kNaN);
    on_rows_ = G;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(v0[i])) {
            on_var_[i] = kNaN;
            continue;
        }
        double v = v0[i];
        for (size_t c = 0; c < G; ++c) {
            const double g = row(c)[i];
            if (!std::isfinite(g)) continue;
            if (c >= first_scored[i]) {
                on_gaps_[c * n + i] = g;
                on_sd_[c * n + i] = std::sqrt(v);
            }
            // only gaps after the burn-in are standardized: the first ones built v0
            if (c >= first_scored[i]) zs[c * n + i] = g / std::sqrt(v);
            v = lam * v + (1.0 - lam) * g * g;
        }
        on_var_[i] = v;
    }
    // Assets without enough gap history (new listing, gaps unknown) get a proxy variance:
    // the median ratio of gap variance to intraday bar variance among the other assets,
    // times this asset's own intraday variance -- never zero gap risk.
    std::fill(on_proxy_.begin(), on_proxy_.end(), char(0));
    {
        std::vector<double> ratios, peers;
        for (size_t i = 0; i < n; ++i)
            if (std::isfinite(on_var_[i])) {
                peers.push_back(on_var_[i]);
                // ratios from estimated models only (a Fallback asset's variance is itself a proxy)
                const bool est = assets_[i].has_params && assets_[i].source != ModelSource::Fallback;
                const double hb = est ? assets_[i].params.unconditional_variance() : kNaN;
                if (std::isfinite(hb) && hb > 0.0) ratios.push_back(on_var_[i] / hb);
            }
        auto median = [](std::vector<double> v) {
            std::nth_element(v.begin(), v.begin() + static_cast<long>(v.size() / 2), v.end());
            return v[v.size() / 2];
        };
        for (size_t i = 0; i < n; ++i) {
            if (std::isfinite(on_var_[i]) || peers.empty()) continue;
            const double hb = assets_[i].has_params ? assets_[i].params.unconditional_variance() : kNaN;
            on_var_[i] = (!ratios.empty() && std::isfinite(hb) && hb > 0.0) ? median(ratios) * hb : median(peers);
            on_proxy_[i] = 1;
        }
        if (peers.empty()) return;  // no asset has gap history yet: overnight model not ready
    }

    // Correlation: sample correlation of standardized gaps, Ledoit-Wolf shrunk towards the
    // constant-correlation matrix (average sample correlation; Ledoit & Wolf 2003), on the
    // sessions where every "core" asset has a scored gap. Overnight and intraday correlations
    // differ structurally (news vs flow), so the intraday matrix is not a sound target. An
    // asset with a short gap history (new listing) would cut the common sample for everyone:
    // assets leave the core, fewest scored gaps first, until the common sessions cover half of
    // the longest history. Assets outside the core (proxied variance or short history) load on
    // the equal-weight core factor -- PSD by construction, unlike a constant fill-in.
    std::vector<size_t> scored(n, 0);
    size_t smax = 0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t c = 0; c < G; ++c) scored[i] += std::isfinite(zs[c * n + i]);
        smax = std::max(smax, scored[i]);
    }
    std::vector<char> use(n);
    for (size_t i = 0; i < n; ++i) use[i] = std::isfinite(on_var_[i]) && !on_proxy_[i] && scored[i] >= 20;
    std::vector<size_t> rows;
    const size_t need = std::max<size_t>(20, (smax + 1) / 2);
    for (;;) {
        rows.clear();
        for (size_t c = 0; c < G; ++c) {
            bool ok = true;
            for (size_t i = 0; i < n && ok; ++i) ok = !use[i] || std::isfinite(zs[c * n + i]);
            if (ok) rows.push_back(c);
        }
        const size_t core = static_cast<size_t>(std::count(use.begin(), use.end(), char(1)));
        if (rows.size() >= need || core <= 2) break;
        size_t worst = n;
        for (size_t i = 0; i < n; ++i)
            if (use[i] && (worst == n || scored[i] < scored[worst])) worst = i;
        use[worst] = 0;
    }
    const size_t core = static_cast<size_t>(std::count(use.begin(), use.end(), char(1)));
    on_core_ = use;
    std::vector<double> M(n * n, 0.0);
    for (size_t c : rows)
        for (size_t i = 0; i < n; ++i)
            if (use[i])
                for (size_t j = 0; j < n; ++j)
                    if (use[j]) M[i * n + j] += zs[c * n + i] * zs[c * n + j];
    const double m = static_cast<double>(rows.size());
    on_corr_ = R_;  // used only until there are enough common sessions
    if (rows.size() >= 5 && core >= 2) {
        std::vector<double> C(n * n, 0.0);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j) {
                const double d = std::sqrt(M[i * n + i] * M[j * n + j]);
                C[i * n + j] = (use[i] && use[j] && d > 0.0) ? M[i * n + j] / d : (i == j ? 1.0 : 0.0);
            }
        double rbar = 0.0;
        int pairs = 0;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = i + 1; j < n; ++j)
                if (use[i] && use[j]) {
                    rbar += C[i * n + j];
                    ++pairs;
                }
        rbar = pairs > 0 ? rbar / pairs : 0.0;
        // Ledoit-Wolf intensity: sum of estimation variances / squared distance to the target.
        // Sampling variance of a sample correlation by the delta method (valid without
        // normality): Var(z_i z_j - r/2 (z_i^2 + z_j^2)) / m, = (1 - r^2)^2 / m under normality.
        double var_sum = 0.0, dist = 0.0;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = i + 1; j < n; ++j) {
                if (!use[i] || !use[j]) continue;
                const double r = C[i * n + j];
                double s1 = 0.0, s2 = 0.0;
                for (size_t c : rows) {
                    const double zi = zs[c * n + i], zj = zs[c * n + j];
                    const double e = zi * zj - 0.5 * r * (zi * zi + zj * zj);
                    s1 += e;
                    s2 += e * e;
                }
                var_sum += std::max(0.0, s2 / m - (s1 / m) * (s1 / m)) / m;
                const double diff = C[i * n + j] - rbar;
                dist += diff * diff;
            }
        const double delta = dist > 0.0 ? std::clamp(var_sum / dist, 0.0, 1.0) : 1.0;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                if (use[i] && use[j] && i != j) C[i * n + j] = delta * rbar + (1.0 - delta) * C[i * n + j];
        // Outside the core: corr(p, j) = kappa f_j and corr(p, p') = kappa^2, where f_j is the
        // correlation of core asset j with the equal-weight core average F, i.e. p = kappa F +
        // sqrt(1 - kappa^2) e_p; kappa matches the average core correlation rbar (0 if rbar <= 0).
        double q = 0.0;
        std::vector<double> f(n, 0.0);
        for (size_t i = 0; i < n; ++i)
            if (use[i])
                for (size_t j = 0; j < n; ++j)
                    if (use[j]) f[i] += C[i * n + j];
        for (size_t i = 0; i < n; ++i) q += f[i];
        double fbar = 0.0;
        for (size_t i = 0; i < n; ++i)
            if (use[i]) {
                f[i] = q > 0.0 ? f[i] / std::sqrt(q) : 0.0;  // q = 1'C1 >= 0 (C is PSD)
                fbar += f[i] / static_cast<double>(core);
            }
        const double kappa = (rbar > 0.0 && fbar > 0.0) ? std::min(1.0, rbar / fbar) : 0.0;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j) {
                double v;
                if (i == j) v = 1.0;
                else if (use[i] && use[j]) v = C[i * n + j];
                else if (!std::isfinite(on_var_[i]) || !std::isfinite(on_var_[j])) v = 0.0;
                else if (use[i]) v = kappa * f[i];
                else if (use[j]) v = kappa * f[j];
                else v = kappa * kappa;
                on_corr_[i * n + j] = v;
            }
    }

    // Tail: pooled standardized gaps.
    std::vector<double> pooled;
    for (double v : zs)
        if (std::isfinite(v)) pooled.push_back(v);
    if (cfg_.spec.dist == Dist::Normal) on_nu_ = kInf;
    else if (pooled.size() >= 100) on_nu_ = fit_student_shock(pooled, cfg_.fit.nu_min, cfg_.fit.nu_max).nu;
    else on_nu_ = std::isfinite(nu_p_) ? nu_p_ : 6.0;
    on_ready_ = true;
}

// ============================================================== estimation

RefitReport IntradayRiskEngine::refit(const double* exposures) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    std::unique_lock<std::shared_mutex> lk(mtx_);
    RefitReport rep;
    const size_t n = static_cast<size_t>(n_);
    const size_t T = count_;
    rep.window_bars = static_cast<int>(T);
    if (T < static_cast<size_t>(cfg_.min_fit_bars)) {
        rep.ready = ready_;
        rep.portfolio_nu = nu_p_;
        rep.dcc_a = dcc_a_;
        rep.dcc_b = dcc_b_;
        rep.seconds = std::chrono::duration<double>(clock::now() - t0).count();
        return rep;
    }

    // Basic exception safety (e.g. std::bad_alloc in a fit): until this refit completes the
    // engine is not ready, so a half-built state is never served (compute() returns NaN).
    ready_ = false;

    // ---- chronological copy of the window
    std::vector<int> slots(T);
    win_r_.resize(T * n);
    for (size_t c = 0; c < T; ++c) {
        const size_t k = idx(c);
        slots[c] = slot_[k];
        std::copy(&ret_[k * n], &ret_[k * n] + n, &win_r_[c * n]);
    }
    win_rows_ = T;
    win_slot_ = slots;

    // ---- intraday profiles: pooled, and per asset shrunk towards it
    if (cfg_.diurnal) {
        std::vector<int> all_slots(T * n);
        for (size_t c = 0; c < T; ++c) std::fill_n(&all_slots[c * n], n, slots[c]);
        diurnal_ = DiurnalProfile::estimate(win_r_.data(), all_slots.data(), T * n, cfg_.bars_per_day);
        if (cfg_.diurnal_mode == DiurnalMode::PerAsset) {
            const std::vector<DiurnalProfile> per =
                DiurnalProfile::estimate_per_asset(win_r_.data(), slots.data(), T, n, cfg_.bars_per_day);
            for (size_t i = 0; i < n; ++i) assets_[i].diurnal = per[i];
        } else {
            for (AssetModel& a : assets_) a.diurnal = diurnal_;
        }
    } else {
        diurnal_ = DiurnalProfile(cfg_.bars_per_day);
        for (AssetModel& a : assets_) a.diurnal = diurnal_;
    }

    // ---- per-asset GARCH fits (parallel), filters and in-sample paths
    win_z_.assign(T * n, kNaN);
    win_sd_.assign(T * n, kNaN);
    win_mu_.assign(T * n, 0.0);
    // variance path (same winsorization as the live filter), filter state and window series
    auto finish_asset = [&](size_t i, const std::vector<double>& y) {
        AssetModel& a = assets_[i];
        std::vector<double> resid(T), h(T);
        for (size_t c = 0; c < T; ++c) resid[c] = y[c] - a.params.mu;
        double bc = garch_backcast(resid.data(), T);
        if (!std::isfinite(bc)) bc = a.params.unconditional_variance();
        const double h_next = variance_path_clipped(y, a.params, bc, cfg_.z_clip, h.data());
        a.filter.reset(a.params, h_next);
        for (size_t c = 0; c < T; ++c) {
            const double sk = a.diurnal[slots[c]];
            win_sd_[c * n + i] = sk * std::sqrt(h[c]);
            win_mu_[c * n + i] = sk * a.params.mu;
            win_z_[c * n + i] = std::isfinite(y[c]) ? resid[c] / std::sqrt(h[c]) : kNaN;
        }
        a.last_z = win_z_[(T - 1) * n + i];
    };
    parallel_for(n_, resolve_threads(cfg_.threads), [&](int ii) {
        const size_t i = static_cast<size_t>(ii);
        AssetModel& a = assets_[i];
        std::vector<double> y(T);
        size_t finite = 0, zeros = 0;
        for (size_t c = 0; c < T; ++c) {
            y[c] = win_r_[c * n + i] / a.diurnal[slots[c]];
            if (std::isfinite(y[c])) {
                ++finite;
                zeros += y[c] == 0.0;
            }
        }
        a.zero_share = finite ? static_cast<double>(zeros) / static_cast<double>(finite) : 0.0;

        FitOptions fo = cfg_.fit;
        const GarchParams prev = a.params;
        // a proxy (Fallback) was never estimated: no warm start, and it stays a proxy below
        const bool fitted_before = a.has_params && a.source != ModelSource::Fallback;
        if (fitted_before) fo.warm_start = &prev;
        GarchSpec spec = cfg_.spec;
        FitOptions robust_fo = fo;
        robust_fo.nu_min = std::max(fo.nu_min, 3.0);
        GarchSpec robust_spec = spec;
        robust_spec.variance_targeting = true;
        a.robust_fit = a.zero_share > cfg_.discrete_zero_share;
        try {
            a.fit = a.robust_fit ? fit_garch(y, robust_spec, robust_fo) : fit_garch(y, spec, fo);
            const bool degenerate = a.fit.usable() && (a.fit.params.persistence() > 0.9995 ||
                                                       (spec.dist == Dist::StudentT && a.fit.params.nu < fo.nu_min + 0.05));
            if (!a.robust_fit && degenerate) {
                FitResult r = fit_garch(y, robust_spec, robust_fo);
                if (r.usable()) {
                    a.fit = std::move(r);
                    a.robust_fit = true;
                }
            }
        } catch (const std::exception& e) {  // e.g. std::bad_alloc: keep the engine alive
            a.fit = FitResult{};
            a.fit.status = FitStatus::NumericalError;
            a.fit.message = e.what();
        }
        if (a.fit.usable() && a.fit.params.is_valid(cfg_.spec.dist)) {
            a.params = a.fit.params;
            a.source = ModelSource::Fitted;
        } else if (fitted_before) {
            a.source = ModelSource::PreviousFit;
        } else {
            a.source = ModelSource::Fallback;  // parameters set after all fits (needs the peers)
            return;
        }
        a.has_params = true;

        // Robust filter: with z_clip the estimates must not be driven by a bad tick either.
        // Winsorize the residuals at z_clip conditional std devs and refit from these params.
        if (cfg_.z_clip > 0.0 && a.source == ModelSource::Fitted) {
            std::vector<double> h(T), yw(y);
            std::vector<double> r0(T);
            for (size_t c = 0; c < T; ++c) r0[c] = y[c] - a.params.mu;
            double bc0 = garch_backcast(r0.data(), T);
            if (!std::isfinite(bc0)) bc0 = a.params.unconditional_variance();
            variance_path_clipped(y, a.params, bc0, cfg_.z_clip, h.data());
            bool clipped = false;
            for (size_t c = 0; c < T; ++c) {
                const double lim = cfg_.z_clip * std::sqrt(h[c]);
                if (std::isfinite(y[c]) && std::fabs(y[c] - a.params.mu) > lim) {
                    yw[c] = a.params.mu + (y[c] > a.params.mu ? lim : -lim);
                    clipped = true;
                }
            }
            if (clipped) {
                FitOptions wo = a.robust_fit ? robust_fo : fo;
                const GarchParams start = a.params;
                wo.warm_start = &start;
                try {
                    FitResult r = fit_garch(yw, a.robust_fit ? robust_spec : spec, wo);
                    if (r.usable() && r.params.is_valid(cfg_.spec.dist)) {
                        a.fit = std::move(r);
                        a.params = a.fit.params;
                    }
                } catch (const std::exception&) {
                }
            }
        }
        finish_asset(i, y);
    });

    // Assets without any usable history: a conservative proxy variance (90th percentile of
    // the other assets' unconditional variances, or the asset's own sample variance if
    // larger) -- a dead or brand-new feed must not look riskless.
    {
        std::vector<double> peer;
        for (const AssetModel& a : assets_)
            if (a.has_params && a.source != ModelSource::Fallback && std::isfinite(a.params.unconditional_variance()))
                peer.push_back(a.params.unconditional_variance());
        double p90 = 1e-4;  // no peers at all: 1% per bar
        if (!peer.empty()) {
            std::sort(peer.begin(), peer.end());
            p90 = peer[std::min(peer.size() - 1, static_cast<size_t>(0.9 * static_cast<double>(peer.size())))];
        }
        for (size_t i = 0; i < n; ++i) {
            AssetModel& a = assets_[i];
            if (a.source != ModelSource::Fallback) continue;
            std::vector<double> y(T);
            double ss = 0.0;
            size_t m = 0;
            for (size_t c = 0; c < T; ++c) {
                y[c] = win_r_[c * n + i] / a.diurnal[slots[c]];
                if (std::isfinite(y[c])) {
                    ss += y[c] * y[c];
                    ++m;
                }
            }
            const double var = std::max(m > 1 ? ss / static_cast<double>(m) : 0.0, p90);
            a.params = GarchParams{0.0, 0.05 * var, 0.05, 0.90, cfg_.spec.dist == Dist::StudentT ? 8.0 : kInf};
            a.has_params = true;
            finish_asset(i, y);
        }
    }

    size_t n_active = 0;
    for (size_t i = 0; i < n; ++i) {
        AssetModel& a = assets_[i];
        if (a.source == ModelSource::Fitted) ++rep.n_fitted;
        else if (a.source == ModelSource::PreviousFit) ++rep.n_previous;
        else ++rep.n_fallback;
        size_t seen = 0;
        for (size_t c = 0; c < T; ++c) seen += std::isfinite(win_z_[c * n + i]);
        a.active = static_cast<double>(seen) >= cfg_.min_active_share * static_cast<double>(T);
        n_active += a.active;
    }
    rep.n_inactive = static_cast<int>(n - n_active);

    // ---- window correlation of standardized residuals over active assets, on complete
    // continuous-trading bars (=> PSD; inactive assets get an identity block): DCC target.
    std::vector<double> M(n * n, 0.0);
    size_t m = 0;  // complete bars
    std::vector<size_t> reg_rows;  // complete continuous-trading bars, chronological
    for (size_t c = 0; c < T; ++c) {
        const double* z = &win_z_[c * n];
        if (!complete_row(z)) continue;
        ++m;
        const bool regular = !is_auction_slot(slots[c]);
        if (regular) reg_rows.push_back(c);
        for (size_t i = 0; i < n; ++i) {
            if (!assets_[i].active) continue;
            for (size_t j = 0; j < n; ++j) {
                if (!assets_[j].active) continue;
                if (regular) M[i * n + j] += clip(z[i], cfg_.z_clip) * clip(z[j], cfg_.z_clip);
            }
        }
    }
    rep.complete_rows = static_cast<int>(m);
    std::fill(R0_.begin(), R0_.end(), 0.0);
    if (n_active > 0 && reg_rows.size() >= n_active + 10) {
        normalize_corr(M, R0_);
    } else {
        for (size_t i = 0; i < n; ++i) R0_[i * n + i] = 1.0;
    }

    // ---- starting point of the correlation recursions. EWMA has no target: starting it from
    // the full-window average would leak the window's correlation level into every bar, so it
    // starts from the first quarter of the window only; those rows are then not scored again
    // (estimation, auction scales, in-sample tail) -- honest one-step-ahead forecasts.
    Rinit_ = R0_;
    win_start_ = 0;
    if (cfg_.correlation == CorrelationMode::Ewma) {
        const size_t burn = std::min(reg_rows.size(), std::max<size_t>(reg_rows.size() / 4, n_active + 10));
        std::vector<double> Mb(n * n, 0.0);
        for (size_t r = 0; r < burn; ++r) {
            const double* z = &win_z_[reg_rows[r] * n];
            for (size_t i = 0; i < n; ++i)
                if (assets_[i].active)
                    for (size_t j = 0; j < n; ++j)
                        if (assets_[j].active) Mb[i * n + j] += clip(z[i], cfg_.z_clip) * clip(z[j], cfg_.z_clip);
        }
        if (burn >= n_active + 10 && burn < reg_rows.size()) {
            std::fill(Rinit_.begin(), Rinit_.end(), 0.0);
            normalize_corr(Mb, Rinit_);
            win_start_ = reg_rows[burn - 1] + 1;
        }
    }

    // ---- dynamic correlation: estimate the parameters, then run the recursion to "now".
    // Auction bars do not update it; their cross-products are regressed on the forecast
    // correlation to estimate their own scale.
    estimate_corr_dynamics(rep);
    Q_ = Rinit_;
    const size_t K = static_cast<size_t>(cfg_.bars_per_day);
    std::vector<double> row_a, row_b;
    std::vector<size_t> row_k;
    for (size_t c = win_start_; c < T; ++c) {
        const double* z = &win_z_[c * n];
        if (is_auction_slot(slots[c])) {  // regression over the observed active pairs
            double a = 0.0, b = 0.0;
            for (size_t i = 0; i < n; ++i) {
                if (!assets_[i].active || !std::isfinite(z[i])) continue;
                const double zi = clip(z[i], cfg_.z_clip);
                for (size_t j = i + 1; j < n; ++j) {
                    if (!assets_[j].active || !std::isfinite(z[j])) continue;
                    const double rij = Q_[i * n + j] / std::sqrt(Q_[i * n + i] * Q_[j * n + j]);
                    a += zi * clip(z[j], cfg_.z_clip) * rij;
                    b += rij * rij;
                }
            }
            if (b > 0.0) {
                row_a.push_back(a);
                row_b.push_back(b);
                row_k.push_back(static_cast<size_t>(slots[c]));
            }
            continue;
        }
        if (cfg_.correlation != CorrelationMode::Constant) corr_update(Q_, z);  // imputes halts
    }
    if (cfg_.correlation != CorrelationMode::Constant) normalize_corr(Q_, R_);
    else R_ = R0_;
    shrink_corr(R_);

    // c_k = sum A / sum B per auction slot; sampling variance by the delta method over bars
    // (independent across days). Shrunk towards 1 (= continuous-trading correlation) by the
    // positive-part rule c = 1 + d max(0, 1 - v / d^2), d = c_hat - 1: no evidence, no change.
    std::fill(corr_scale_.begin(), corr_scale_.end(), 1.0);
    for (size_t k = 0; k < K; ++k) {
        if (!is_auction_slot(static_cast<int>(k))) continue;
        double sa = 0.0, sb = 0.0;
        int cnt = 0;
        for (size_t r = 0; r < row_a.size(); ++r)
            if (row_k[r] == k) {
                sa += row_a[r];
                sb += row_b[r];
                ++cnt;
            }
        if (cnt < 10 || !(sb > 0.0)) continue;
        const double ck = sa / sb;
        double v = 0.0;
        for (size_t r = 0; r < row_a.size(); ++r)
            if (row_k[r] == k) {
                const double e = row_a[r] - ck * row_b[r];
                v += e * e;
            }
        v /= sb * sb;
        const double d = ck - 1.0;
        const double w = d * d > 0.0 ? std::max(0.0, 1.0 - v / (d * d)) : 0.0;
        corr_scale_[k] = std::clamp(1.0 + w * d, 0.0, 1.0);
    }
    rep.dcc_a = dcc_a_;
    rep.dcc_b = dcc_b_;

    ready_ = true;
    try {
        recalibrate_tail_locked(exposures);
        update_overnight_model();  // its correlation target changed
    } catch (...) {
        ready_ = false;
        throw;
    }
    rep.ready = true;
    rep.portfolio_nu = nu_p_;
    rep.seconds = std::chrono::duration<double>(clock::now() - t0).count();
    return rep;
}

// Correlation dynamics by composite likelihood over pairs (Engle, Shephard and Sheppard,
// 2008): DCC(1,1) (a, b) or EWMA lambda. Each pair's 2x2 recursion depends only on
// (q_ii, q_jj, q_ij), so the objective and its analytic gradient cost O(window * pairs).
void IntradayRiskEngine::estimate_corr_dynamics(RefitReport& rep) {
    if (cfg_.correlation == CorrelationMode::Constant || !cfg_.estimate_correlation) return;
    const bool ewma = cfg_.correlation == CorrelationMode::Ewma;
    const size_t n = static_cast<size_t>(n_);
    std::vector<size_t> act;
    for (size_t i = 0; i < n; ++i)
        if (assets_[i].active) act.push_back(i);
    const size_t na = act.size();
    if (na < 2) return;
    // Continuous-trading rows after the burn-in that built Rinit_ (EWMA), active assets only,
    // clipped; NaN = not traded. Pairwise-complete composite likelihood: each pair's recursion
    // runs over the bars where both assets traded, so a halted asset does not remove the
    // other pairs' information (as a complete-rows sample would).
    const size_t first = ewma ? win_start_ : 0;
    std::vector<double> Z;
    for (size_t c = first; c < win_rows_; ++c) {
        const double* z = &win_z_[c * n];
        if (is_auction_slot(win_slot_[c])) continue;
        int seen = 0;
        for (size_t k = 0; k < na; ++k) seen += std::isfinite(z[act[k]]);
        if (seen < 2) continue;
        for (size_t k = 0; k < na; ++k) Z.push_back(std::isfinite(z[act[k]]) ? clip(z[act[k]], cfg_.z_clip) : kNaN);
    }
    const size_t m = Z.size() / na;

    struct Pair {
        size_t p, q;
        double qbar, qinit;
    };
    std::vector<Pair> pairs;
    const size_t total = na * (na - 1) / 2;
    const size_t stride = (total + static_cast<size_t>(cfg_.dcc_max_pairs) - 1) / static_cast<size_t>(cfg_.dcc_max_pairs);
    size_t counter = 0, terms = 0;
    for (size_t p = 0; p < na; ++p)
        for (size_t q = p + 1; q < na; ++q, ++counter) {
            if (counter % stride != 0) continue;
            size_t both = 0;
            for (size_t t = 0; t < m; ++t) both += std::isfinite(Z[t * na + p]) && std::isfinite(Z[t * na + q]);
            if (both < 50) continue;  // too short a joint history for this pair
            terms += both;
            pairs.push_back({p, q, R0_[act[p] * n + act[q]], Rinit_[act[p] * n + act[q]]});
        }
    if (pairs.empty()) return;

    const double pmax = 0.999;
    const int threads = std::min(resolve_threads(cfg_.threads), static_cast<int>(pairs.size()));
    std::vector<double> part_ll(static_cast<size_t>(threads)), part_ga(part_ll.size()), part_gb(part_ll.size());

    // EWMA (integrated) mode: lambda in (0.8, 0.9995), a = 1 - lambda, no target.
    const double lam_lo = 0.8, lam_span = 0.1995;
    ObjectiveFn objective = [&](const std::vector<double>& u, std::vector<double>& g) -> double {
        double sp, P, sa, a, b, w;
        if (ewma) {
            sp = logistic(u[0]);
            b = lam_lo + lam_span * sp;
            a = 1.0 - b;
            w = 0.0;
            P = 1.0;
            sa = a;
        } else {
            sp = logistic(u[0]);
            P = pmax * sp;
            sa = logistic(u[1]);
            a = sa * P;
            b = (1.0 - sa) * P;
            w = 1.0 - a - b;
        }
        std::atomic<bool> bad{false};
        parallel_for(threads, threads, [&](int th) {
            double ll = 0.0, ga = 0.0, gb = 0.0;
            for (size_t k = static_cast<size_t>(th); k < pairs.size(); k += static_cast<size_t>(threads)) {
                const Pair& pr = pairs[k];
                double q11 = 1.0, q22 = 1.0, q12 = ewma ? pr.qinit : pr.qbar;
                double d11a = 0, d22a = 0, d12a = 0, d11b = 0, d22b = 0, d12b = 0;
                for (size_t t = 0; t < m; ++t) {
                    const double z1 = Z[t * na + pr.p], z2 = Z[t * na + pr.q];
                    if (!std::isfinite(z1) || !std::isfinite(z2)) continue;
                    const double s = std::sqrt(q11 * q22);
                    const double rho = q12 / s;
                    const double one = 1.0 - rho * rho;
                    if (!(one > 1e-12)) {
                        bad = true;
                        break;
                    }
                    const double A = z1 * z1 + z2 * z2, B = z1 * z2;
                    ll += -0.5 * (std::log(one) + (A - 2.0 * rho * B) / one);
                    const double dl = rho / one + (B * one - rho * (A - 2.0 * rho * B)) / (one * one);
                    ga += dl * (d12a / s - 0.5 * rho * (d11a / q11 + d22a / q22));
                    gb += dl * (d12b / s - 0.5 * rho * (d11b / q11 + d22b / q22));
                    // recursion and its derivatives (old q values on the right-hand side)
                    const double tw = ewma ? 0.0 : 1.0;  // target weight derivative (-1 or 0)
                    d11a = -tw + z1 * z1 + b * d11a;
                    d22a = -tw + z2 * z2 + b * d22a;
                    d12a = -tw * pr.qbar + B + b * d12a;
                    d11b = -tw + q11 + b * d11b;
                    d22b = -tw + q22 + b * d22b;
                    d12b = -tw * pr.qbar + q12 + b * d12b;
                    q11 = w + a * z1 * z1 + b * q11;
                    q22 = w + a * z2 * z2 + b * q22;
                    q12 = w * pr.qbar + a * B + b * q12;
                }
            }
            part_ll[static_cast<size_t>(th)] = ll;
            part_ga[static_cast<size_t>(th)] = ga;
            part_gb[static_cast<size_t>(th)] = gb;
        });
        if (bad) return kInf;
        double ll = 0.0, ga = 0.0, gb = 0.0;
        for (size_t k = 0; k < part_ll.size(); ++k) {
            ll += part_ll[k];
            ga += part_ga[k];
            gb += part_gb[k];
        }
        const double scale = -1.0 / static_cast<double>(terms);
        if (ewma) {  // a = 1 - b, b = lam_lo + span * logistic(u0)
            g[0] = scale * (gb - ga) * lam_span * sp * (1.0 - sp);
            return scale * ll;
        }
        const double dP = pmax * sp * (1.0 - sp);
        g[0] = scale * (ga * sa + gb * (1.0 - sa)) * dP;
        g[1] = scale * (ga - gb) * P * sa * (1.0 - sa);
        return scale * ll;
    };
    auto to_u = [&](double a, double b) {
        const double P = std::clamp(a + b, 1e-6, pmax * (1.0 - 1e-9));
        const double sa = std::clamp(a / P, 1e-6, 1.0 - 1e-6);
        return std::vector<double>{std::log((P / pmax) / (1.0 - P / pmax)), std::log(sa / (1.0 - sa))};
    };
    if (ewma) {
        // One parameter on a bounded interval: Brent. BFGS on the logit crawls (hundreds of
        // O(window x pairs) evaluations) when the optimum sits at the upper bound, as it does on
        // short real samples.
        std::vector<double> g(1);
        auto f = [&](double lam) {
            const double s = std::clamp((lam - lam_lo) / lam_span, 1e-12, 1.0 - 1e-12);
            return objective(std::vector<double>{std::log(s / (1.0 - s))}, g);
        };
        // warm start: a narrow bracket around yesterday's lambda (a few parabolic steps), widened
        // to the whole range when the optimum lands on the bracket's edge
        const double lo_b = lam_lo, hi_b = lam_lo + lam_span, prev = dcc_b_;
        double lam;
        if (dcc_estimated_ && prev > lo_b && prev < hi_b) {
            const double a = std::max(lo_b, prev - 0.005), b = std::min(hi_b, prev + 0.005);
            lam = math::brent_minimize(f, a, b, 1e-7);
            if ((a > lo_b && lam - a < 1e-4) || (b < hi_b && b - lam < 1e-4))
                lam = math::brent_minimize(f, lo_b, hi_b, 1e-7);
        } else {
            lam = math::brent_minimize(f, lo_b, hi_b, 1e-7);
        }
        if (!std::isfinite(f(lam))) return;
        dcc_b_ = lam;
        dcc_a_ = 1.0 - lam;
        dcc_estimated_ = true;
        rep.dcc_estimated = true;
        return;
    }
    BfgsOptions bo;
    bo.max_iter = 200;
    bo.grad_tol = 1e-8;
    BfgsResult best;
    best.f = kInf;
    // warm start from the current parameters; the first estimation also tries a second start
    std::vector<std::pair<double, double>> starts{{dcc_a_, dcc_b_}};
    if (!dcc_estimated_) starts.push_back({0.03, 0.94});
    for (const auto& st : starts) {
        BfgsResult r = minimize_bfgs(objective, to_u(st.first, st.second), bo);
        if (std::isfinite(r.f) && r.f < best.f) best = std::move(r);
    }
    if (!std::isfinite(best.f)) return;
    const double sp = logistic(best.x[0]), P = pmax * sp, sa = logistic(best.x[1]);
    dcc_a_ = sa * P;
    dcc_b_ = (1.0 - sa) * P;
    dcc_estimated_ = true;
    rep.dcc_estimated = true;
}

// Standardized in-sample P&L of the given portfolio, one step ahead (the correlation is the
// one known before each bar, scaled by the bar's c_k). Uses the winsorized residuals
// (sd * clip(z)), so with z_clip a bad tick does not distort the tail either. Rows before
// `start` initialized the correlation recursion and are not scored. Inactive assets have no
// reliable history and are left out.
static void insample_portfolio_u(const std::vector<double>& win_z, const std::vector<double>& win_sd,
                                 const std::vector<int>& win_slot, const std::vector<double>& scale,
                                 size_t start, size_t rows, size_t n, const double* x, double zc, double shrinkage,
                                 const std::vector<AssetModel>& assets, std::vector<double> Q,
                                 const std::function<bool(const double*)>& complete,
                                 const std::function<bool(int)>& auction,
                                 const std::function<void(std::vector<double>&, const double*)>& update,
                                 std::vector<double>& u) {
    std::vector<double> b(n);
    for (size_t c = start; c < rows; ++c) {
        const double* z = &win_z[c * n];
        if (complete(z)) {  // a bar with a missing active asset has no complete P&L to score
            double num = 0.0;
            for (size_t i = 0; i < n; ++i) {
                if (!assets[i].active) {
                    b[i] = 0.0;
                    continue;
                }
                b[i] = x[i] * win_sd[c * n + i] / std::sqrt(Q[i * n + i]);
                num += x[i] * win_sd[c * n + i] * clip(z[i], zc);
            }
            // Q has diagonal Q_ii; b_i = x_i sd_i / sqrt(Q_ii) normalizes it to a correlation.
            // Off-diagonals scaled exactly as in compute(): (1 - shrinkage) * c_k.
            const double ck = (1.0 - shrinkage) * scale[static_cast<size_t>(win_slot[c])];
            double var = 0.0;
            for (size_t i = 0; i < n; ++i) {
                double acc = 0.0;
                for (size_t j = 0; j < n; ++j) acc += Q[i * n + j] * b[j];
                var += b[i] * (ck * acc + (1.0 - ck) * Q[i * n + i] * b[i]);
            }
            if (var > 0.0 && std::isfinite(num)) u.push_back(num / std::sqrt(var));
        }
        if (!auction(win_slot[c])) update(Q, z);  // same recursion (and imputation) as on_bar
    }
}

ShockDist IntradayRiskEngine::calibrate_tail(const double* x) const {
    std::vector<double> u;
    const bool oos_only = cfg_.tail == TailCalibration::OutOfSample &&
                          oos_u_.size() >= static_cast<size_t>(cfg_.tail_min_oos);
    if (!oos_only) {
        u.reserve(win_rows_ + oos_u_.size());
        insample_portfolio_u(
            win_z_, win_sd_, win_slot_, corr_scale_, win_start_, win_rows_, static_cast<size_t>(n_), x, cfg_.z_clip,
            cfg_.corr_shrinkage,
            assets_, Rinit_, [this](const double* z) { return complete_row(z); },
            [this](int k) { return is_auction_slot(k); },
            [this](std::vector<double>& Q, const double* z) {
                if (cfg_.correlation != CorrelationMode::Constant) corr_update(Q, z);
            },
            u);
    }
    if (cfg_.tail == TailCalibration::OutOfSample) u.insert(u.end(), oos_u_.begin(), oos_u_.end());

    if (u.size() < 100) {  // not enough information: median of the asset-level estimates
        std::vector<double> nus;
        for (const AssetModel& a : assets_)
            if (std::isfinite(a.params.nu)) nus.push_back(a.params.nu);
        if (nus.empty()) return ShockDist::student(8.0);
        std::nth_element(nus.begin(), nus.begin() + static_cast<long>(nus.size() / 2), nus.end());
        return ShockDist::student(nus[nus.size() / 2]);
    }
    const ShockDist t = fit_student_shock(u, cfg_.fit.nu_min, cfg_.fit.nu_max);
    if (cfg_.tail_model == TailModel::Evt)
        return fit_evt_shock(u, t.nu, cfg_.evt_tail_fraction, cfg_.evt_min_exceedances);
    return t;
}

double IntradayRiskEngine::recalibrate_tail_locked(const double* exposures) {
    if (cfg_.spec.dist == Dist::StudentT) {
        const ShockDist d = ready_ ? calibrate_tail(exposures) : ShockDist::student(kNaN);
        nu_p_ = d.nu;
        tail_.set(d, cfg_.confidences);
        tail_oos_only_ = ready_ && cfg_.tail == TailCalibration::OutOfSample &&
                         oos_u_.size() >= static_cast<size_t>(cfg_.tail_min_oos);
    } else {
        nu_p_ = kInf;
        tail_.set(Dist::Normal, kInf, cfg_.confidences);
    }
    return nu_p_;
}

double IntradayRiskEngine::recalibrate_tail(const double* exposures) {
    std::unique_lock<std::shared_mutex> lk(mtx_);
    // A tail fitted on out-of-sample outcomes alone does not depend on the exposures: it is
    // kept, and refreshed with the newer outcomes at the next refit().
    if (tail_oos_only_ && std::isfinite(nu_p_)) return nu_p_;
    return recalibrate_tail_locked(exposures);
}

// ============================================================== risk

void IntradayRiskEngine::compute(int next_slot, const double* x, PortfolioRisk& out) const {
    std::shared_lock<std::shared_mutex> lk(mtx_);
    const size_t n = static_cast<size_t>(n_);
    const size_t K = cfg_.confidences.size();
    out.asset_sigma.resize(n);
    out.position_sigma.resize(n);
    out.sigma_contrib.resize(n);
    out.measures.resize(K);
    out.nu = nu_p_;
    out.tail = tail_.dist();
    if (!ready_) {
        out.mean = out.sigma = kNaN;
        std::fill(out.asset_sigma.begin(), out.asset_sigma.end(), kNaN);
        std::fill(out.position_sigma.begin(), out.position_sigma.end(), kNaN);
        std::fill(out.sigma_contrib.begin(), out.sigma_contrib.end(), kNaN);
        for (size_t k = 0; k < K; ++k) out.measures[k] = {cfg_.confidences[k], kNaN, kNaN};
        return;
    }

    const int slot = std::clamp(next_slot, 0, cfg_.bars_per_day - 1);
    const double ck = corr_scale_[static_cast<size_t>(slot)];
    double mean = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const AssetModel& a = assets_[i];
        const double s = a.diurnal[slot];
        const double sd = std::sqrt(s * s * a.filter.variance() + stale_var_[i]);
        out.asset_sigma[i] = sd;
        out.position_sigma[i] = x[i] * sd;
        mean += x[i] * s * a.params.mu;
    }
    double var = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double* row = &R_[i * n];
        double acc = 0.0;
        for (size_t j = 0; j < n; ++j) acc += row[j] * out.position_sigma[j];
        acc = ck * acc + (1.0 - ck) * out.position_sigma[i];  // off-diagonals scaled by c_k
        out.sigma_contrib[i] = out.position_sigma[i] * acc;
        var += out.sigma_contrib[i];
    }
    const double sigma = std::sqrt(std::max(var, 0.0));
    const double inv = sigma > 0.0 ? 1.0 / sigma : 0.0;
    for (size_t i = 0; i < n; ++i) out.sigma_contrib[i] *= inv;
    out.mean = mean;
    out.sigma = sigma;

    for (size_t k = 0; k < K; ++k) out.measures[k] = risk_measure(mean, sigma, tail_.factors()[k]);
}

OvernightRisk IntradayRiskEngine::compute_overnight(const double* x) const {
    std::shared_lock<std::shared_mutex> lk(mtx_);
    OvernightRisk r;
    const size_t n = static_cast<size_t>(n_);
    if (!ready_ || !on_ready_) {
        r.sigma = r.sigma_gap = r.sigma_bar = r.nu = kNaN;
        for (double c : cfg_.confidences) r.measures.push_back({c, kNaN, kNaN});
        return r;
    }
    std::vector<double> ag(n), ab(n);
    double mean = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double s0 = assets_[i].diurnal[0];
        ag[i] = std::isfinite(on_var_[i]) ? x[i] * std::sqrt(on_var_[i]) : 0.0;
        ab[i] = x[i] * s0 * std::sqrt(assets_[i].filter.variance());
        mean += x[i] * s0 * assets_[i].params.mu;
    }
    auto quad = [&](const std::vector<double>& C, const std::vector<double>& v) {
        double acc = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double row = 0.0;
            for (size_t j = 0; j < n; ++j) row += C[i * n + j] * v[j];
            acc += v[i] * row;
        }
        return std::max(acc, 0.0);
    };
    std::vector<double> R1 = R_;  // first bar of the session: slot-0 correlation scaling
    const double c0 = corr_scale_[0];
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            if (i != j) R1[i * n + j] *= c0;
    const double vg = quad(on_corr_, ag), vb = quad(R1, ab);

    // Tail of the *portfolio* gap: idiosyncratic parts diversify and the common shock
    // dominates, so the pooled per-asset tail would be too thin. Unit-variance t fitted on the
    // current portfolio's standardized gap history (one-step-ahead standard deviations).
    double nu_gap = on_nu_;
    if (cfg_.spec.dist == Dist::StudentT) {
        std::vector<double> u, b(n);
        for (size_t c = 0; c < on_rows_; ++c) {
            double num = 0.0;
            bool ok = true;
            for (size_t i = 0; i < n && ok; ++i) {
                if (!on_core_[i] || x[i] == 0.0) {  // core assets: the gap history is complete
                    b[i] = 0.0;
                    continue;
                }
                const double g = on_gaps_[c * n + i], sd = on_sd_[c * n + i];
                ok = std::isfinite(g) && std::isfinite(sd);
                num += x[i] * g;
                b[i] = x[i] * sd;
            }
            if (!ok) continue;
            const double var = quad(on_corr_, b);
            if (var > 0.0) u.push_back(num / std::sqrt(var));
        }
        if (u.size() >= 50) nu_gap = fit_student_shock(u, cfg_.fit.nu_min, cfg_.fit.nu_max).nu;
    }
    r.available = true;
    r.mean = mean;
    r.sigma_gap = std::sqrt(vg);
    r.sigma_bar = std::sqrt(vb);
    r.sigma = std::sqrt(vg + vb);
    // Kurtosis of a sum of independent terms: (k1 v1^2 + k2 v2^2) / (v1 + v2)^2.
    const double k1 = t_excess_kurtosis(nu_gap), k2 = t_excess_kurtosis(nu_p_);
    double nu;
    if (cfg_.spec.dist == Dist::Normal) nu = kInf;
    else if (!std::isfinite(k1) || !std::isfinite(k2)) nu = std::min(nu_gap, nu_p_);
    else {
        const double v = vg + vb;
        const double k = v > 0.0 ? (k1 * vg * vg + k2 * vb * vb) / (v * v) : 0.0;
        nu = k > 0.0 ? std::clamp(4.0 + 6.0 / k, cfg_.fit.nu_min, cfg_.fit.nu_max) : kInf;
    }
    r.nu = nu;
    for (double c : cfg_.confidences)
        r.measures.push_back(
            risk_measure(mean, r.sigma, tail_factors(std::isfinite(nu) ? Dist::StudentT : Dist::Normal, nu, c)));
    return r;
}

JointForecast IntradayRiskEngine::joint_forecast(int next_slot) const {
    std::shared_lock<std::shared_mutex> lk(mtx_);
    JointForecast f;
    const size_t n = static_cast<size_t>(n_);
    const int k = std::clamp(next_slot, 0, cfg_.bars_per_day - 1);
    f.mu.resize(n);
    f.sigma.resize(n);
    for (size_t i = 0; i < n; ++i) {
        const double s = assets_[i].diurnal[k];
        f.mu[i] = s * assets_[i].params.mu;
        f.sigma[i] = std::sqrt(s * s * assets_[i].filter.variance() + stale_var_[i]);
    }
    f.corr = R_;
    const double ck = corr_scale_[static_cast<size_t>(k)];
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            if (i != j) f.corr[i * n + j] *= ck;
    f.nu = nu_p_;
    return f;
}

}  // namespace garcht
