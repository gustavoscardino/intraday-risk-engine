// Intraday multi-asset risk engine built on univariate GARCH(1,1)-t models.
//
// Model (per bar, horizon = 1 bar, e.g. 15 minutes):
//   r_{i,t}  = s_{i,k(t)} * y_{i,t}               diurnal factor (per asset, shrunk to pooled)
//   y_{i,t}  ~ GARCH(1,1)-t per asset             h_{i,t}, z_{i,t} = (y_{i,t} - mu_i) / sqrt(h_{i,t})
//   Corr(z)  = R_t                                EWMA (default, lambda estimated by predictive
//                                                 composite likelihood), DCC(1,1) or constant,
//                                                 estimated on continuous-trading bars; the
//                                                 opening/closing auction bars get their own
//                                                 correlation scale c_k in [0, 1]
//   P&L      ~ x' r  with x = signed exposures (money)
//   sigma_p  = sqrt(a' R_k a),  a_i = x_i * s_{i,k(t+1)} * sqrt(h_{i,t+1})
//   VaR/ES   = t body + GPD left tail (EVT, default) or plain t, fitted by maximum likelihood
//              on the portfolio's standardized P&L: out-of-sample outcomes alone once
//              `tail_min_oos` are available, in-sample filtered residuals plus the
//              out-of-sample outcomes so far before that.
//
// Overnight: one gap per session and asset; EWMA variance (per session, lambda estimated by
// predictive likelihood), Ledoit-Wolf shrunk
// correlation (constant-correlation target) and a tail calibrated on the current portfolio's
// standardized gap history. Risk "close -> first bar of the next session" combines the gap
// and the first bar (independent).
//
// The RiskMetrics baseline is a separate model (common/include/riskcommon/riskmetrics.hpp); the
// replay runs both behind the common model interface (garcht/models.hpp).
//
// Thread safety: every public member function may be called concurrently from several
// threads. compute()/compute_overnight()/accessors take a shared lock and run in parallel
// with each other; on_bar(), on_overnight(), observe_outcome(), refit() and
// recalibrate_tail() take an exclusive lock (refit holds it for the whole re-estimation).
//
// Costs: on_bar + compute are O(n^2) with no allocation; refit is O(window * n^2) plus n
// independent GARCH fits and the DCC estimation, run in parallel threads.
#pragma once

#include <deque>
#include <limits>
#include <shared_mutex>
#include <string>
#include <vector>

#include "garcht/garch.hpp"
#include "garcht/risk.hpp"
#include "garcht/seasonality.hpp"

namespace garcht {

enum class CorrelationMode { Constant, Ewma, Dcc };
enum class TailCalibration { InSample, OutOfSample };
enum class DiurnalMode { Pooled, PerAsset };
enum class TailModel { StudentT, Evt };

struct EngineConfig {
    int bars_per_day = 28;          // 10:00-17:00 in 15-minute bars
    int window_bars = 60 * 28;      // rolling estimation window
    int min_fit_bars = 500;         // below this the engine stays "not ready"
    GarchSpec spec{};               // default: Student-t innovations, zero mean
    FitOptions fit{};
    bool diurnal = true;
    DiurnalMode diurnal_mode = DiurnalMode::PerAsset;  // empirical-Bayes shrinkage to pooled
    // Tick-discrete assets (share of zero returns in the window above this) are fitted with
    // variance targeting and nu >= 3; so is any fit that degenerates (a + b > 0.9995 or nu at
    // its lower bound), which otherwise happens when most returns are exactly zero.
    double discrete_zero_share = 0.25;

    // Correlation dynamics. Default: EWMA of standardized residuals with lambda estimated at
    // each refit by *predictive* composite likelihood (first quarter of the window only
    // initializes the recursion). DCC(1,1) mean-reverts to the window average, which on real
    // B3 data kept correlations too close to the window level (see README).
    CorrelationMode correlation = CorrelationMode::Ewma;
    double corr_lambda = 0.99;            // Ewma: decay per bar (starting value if estimated)
    double dcc_a = 0.01, dcc_b = 0.98;    // Dcc: starting values
    bool estimate_correlation = true;     // Ewma: lambda; Dcc: (a, b) -- composite likelihood
    int dcc_max_pairs = 300;              // pairs used by the composite likelihood
    double corr_shrinkage = 0.0;          // R <- (1 - s) R + s I (all correlation modes)
    // Auction bars: the first and last bar of the session (B3 opening and closing calls) are
    // priced by each stock's own order book and are much less correlated than continuous
    // trading. With this flag the correlation dynamics are estimated on continuous-trading
    // bars only, and each auction bar gets a scale c_k in [0, 1] for the off-diagonals,
    // estimated by least squares of the realized cross-products on the forecast correlation
    // and shrunk towards 1 according to its sampling variance (no evidence -> c_k = 1).
    bool auction_correlation = true;
    double min_active_share = 0.5;        // share of the window an asset must be observed

    double z_clip = 0.0;                  // robust filter: winsorize |z| > z_clip (0 = off)
    TailCalibration tail = TailCalibration::OutOfSample;
    int tail_min_oos = 500;               // out-of-sample outcomes needed to use them alone
    // Portfolio tail: unit-variance t(nu_p), or t body with a GPD fitted to the largest
    // `evt_tail_fraction` standardized losses (conditional EVT, McNeil & Frey 2000).
    TailModel tail_model = TailModel::Evt;
    double evt_tail_fraction = 0.10;
    int evt_min_exceedances = 50;
    std::vector<double> confidences{0.95, 0.975, 0.99};
    double ewma_lambda = 0.94;            // RiskMetrics baseline model run beside this one (replay)

    // Overnight variances: EWMA per session with lambda estimated by predictive (Gaussian
    // quasi-) likelihood over the gap history, pooled across assets; this is the start value.
    double overnight_lambda = 0.94;
    bool estimate_overnight_lambda = true;
    int overnight_window = 250;           // sessions kept for overnight correlation / tail
    int threads = 0;                      // 0 = std::thread::hardware_concurrency()
};

enum class ModelSource { Fitted, PreviousFit, Fallback };
const char* to_string(ModelSource s);

struct AssetModel {
    std::string ticker;
    GarchParams params;
    FitResult fit;  // last estimation attempt (including standard errors)
    ModelSource source = ModelSource::Fallback;
    GarchFilter filter;
    double last_z = 0.0;  // last standardized residual (NaN if the bar was missing)
    bool has_params = false;  // ModelSource::Fallback: no usable history; conservative proxy
                              // variance (90th percentile of the other assets), never ~0
    // Enters the correlation estimates (observed in >= min_active_share of the window). An
    // inactive asset (e.g. a dead feed) is treated as uncorrelated. A bar where an active asset
    // is missing (halt) still updates the correlations: its entries are imputed from the
    // observed ones (conditional expectation), so one halted asset cannot freeze the rest.
    bool active = true;
    double zero_share = 0.0;  // share of exactly-zero returns in the window
    bool robust_fit = false;  // fitted with variance targeting (tick-discrete or degenerate)
    DiurnalProfile diurnal;   // this asset's intraday volatility profile
};

struct PortfolioRisk {
    double mean = 0.0;   // expected P&L over the next bar (money)
    double sigma = 0.0;  // std of the P&L over the next bar (money)
    double nu = 0.0;     // tail parameter used (+inf = Normal)
    ShockDist tail;      // full standardized distribution used for VaR / ES
    std::vector<RiskMeasure> measures;   // one per configured confidence (money, positive = loss)
    std::vector<double> asset_sigma;     // per-asset next-bar return std (fraction); for an asset
                                         // that missed bars since its last print this session,
                                         // the std of its move since that print
    std::vector<double> position_sigma;  // stand-alone P&L std per position, signed (money)
    std::vector<double> sigma_contrib;   // Euler contributions to sigma (money); sum == sigma
};

struct OvernightRisk {
    bool available = false;  // enough overnight history
    double mean = 0.0;       // expected P&L (money)
    double sigma_gap = 0.0;  // P&L std of the overnight gap (money)
    double sigma_bar = 0.0;  // P&L std of the first bar of the next session (money)
    double sigma = 0.0;      // combined
    double nu = 0.0;         // tail of the combined distribution (kurtosis matched)
    std::vector<RiskMeasure> measures;
};

struct RefitReport {
    bool ready = false;
    double seconds = 0.0;
    int window_bars = 0;
    int n_fitted = 0, n_previous = 0, n_fallback = 0;
    int n_inactive = 0;     // assets left out of the correlation estimates
    int complete_rows = 0;  // bars with all active assets observed (used for correlations)
    double portfolio_nu = 0.0;
    double dcc_a = 0.0, dcc_b = 0.0;  // Ewma mode: a = 1 - lambda, b = lambda
    bool dcc_estimated = false;
};

class IntradayRiskEngine {
public:
    IntradayRiskEngine(std::vector<std::string> tickers, EngineConfig cfg);
    IntradayRiskEngine(const IntradayRiskEngine&) = delete;
    IntradayRiskEngine& operator=(const IntradayRiskEngine&) = delete;

    int num_assets() const { return n_; }
    const EngineConfig& config() const { return cfg_; }  // immutable after construction
    bool ready() const;
    size_t history_size() const;

    // Feed one bar of log-returns (num_assets() values; NaN = missing). `slot` is the
    // bar position within the session, in [0, bars_per_day).
    void on_bar(int slot, const double* returns);

    // Overnight log-returns (open / previous close) of the session that is starting; call
    // once per session before its first bar. NaN = unknown.
    void on_overnight(const double* gaps);

    // Realized P&L of the bar a forecast referred to; feeds the out-of-sample tail calibration.
    void observe_outcome(double realized_pnl, const PortfolioRisk& forecast);

    // Re-estimates diurnal profile, GARCH models, correlations and the portfolio tail on the
    // rolling window. `exposures` = signed money exposures.
    RefitReport refit(const double* exposures);

    // Re-calibrates the portfolio tail for new exposures (after a rebalance). With the
    // out-of-sample calibration active (enough outcomes observed) the tail does not depend on
    // the exposures: it is left as is and refreshed with the new outcomes at the next refit().
    double recalibrate_tail(const double* exposures);

    // One-bar-ahead risk; `next_slot` is the slot of the bar being forecast.
    void compute(int next_slot, const double* exposures, PortfolioRisk& out) const;
    PortfolioRisk compute(int next_slot, const double* exposures) const {
        PortfolioRisk r;
        compute(next_slot, exposures, r);
        return r;
    }

    // Risk from the close to the end of the first bar of the next session.
    OvernightRisk compute_overnight(const double* exposures) const;

    // Joint distribution of the next bar's returns (for Monte Carlo full revaluation).
    JointForecast joint_forecast(int next_slot) const;

    // Snapshots (copies, safe while other threads update the engine).
    AssetModel asset(int i) const;
    std::vector<double> correlation() const;  // n x n, row-major
    DiurnalProfile diurnal() const;            // pooled profile
    DiurnalProfile diurnal(int asset) const;   // profile used for one asset
    double portfolio_nu() const;
    double dcc_a() const;  // correlation dynamics: DCC (a, b); Ewma: (1 - lambda, lambda)
    double dcc_b() const;
    std::vector<double> slot_correlation_scale() const;  // c_k per slot (1 = continuous trading)
    size_t oos_outcomes() const;
    double overnight_lambda() const;

private:
    size_t idx(size_t chrono) const;  // ring-buffer row of the chrono-th oldest bar
    bool complete_row(const double* v) const;  // all *active* entries finite
    bool is_auction_slot(int slot) const;
    void shrink_corr(std::vector<double>& R) const;  // corr_shrinkage on the off-diagonals
    // E[v v' | observed active entries of v] under covariance C (a correlation if as_corr)
    bool conditional_outer(const std::vector<double>& C, bool as_corr, const double* v,
                           std::vector<double>& out) const;
    void corr_update(std::vector<double>& Q, const double* z) const;  // NaN entries imputed
    void normalize_corr(const std::vector<double>& Q, std::vector<double>& R) const;
    void estimate_corr_dynamics(RefitReport& rep);
    ShockDist calibrate_tail(const double* exposures) const;
    double recalibrate_tail_locked(const double* exposures);
    void update_overnight_model();

    int n_;
    EngineConfig cfg_;
    mutable std::shared_mutex mtx_;
    bool ready_ = false;

    // rolling history (ring buffer, row-major bars x assets)
    std::vector<double> ret_;
    std::vector<int> slot_;
    size_t cap_, head_ = 0, count_ = 0;

    std::vector<AssetModel> assets_;
    DiurnalProfile diurnal_;
    std::vector<double> R_;   // current correlation matrix (n x n)
    std::vector<double> Q_;   // DCC / EWMA quasi-correlation
    std::vector<double> R0_;  // window correlation (continuous trading): DCC target
    std::vector<double> Rinit_;  // starting point of the recursions (EWMA: first quarter only)
    size_t win_start_ = 0;       // first window row scored after that burn-in
    double dcc_a_, dcc_b_;
    bool dcc_estimated_ = false;
    TailTable tail_, normal_tail_;
    double nu_p_ = 0.0;  // == tail_.dist().nu
    bool tail_oos_only_ = false;  // current tail fitted on out-of-sample outcomes alone
    std::deque<double> oos_u_;  // standardized out-of-sample portfolio outcomes

    // in-sample window snapshot from the last refit (for tail re-calibration)
    size_t win_rows_ = 0;
    std::vector<double> win_r_, win_z_, win_sd_;  // row-major win_rows_ x n
    std::vector<double> win_mu_;                  // per-row expected return (s_k * mu_i)
    std::vector<int> win_slot_;                   // slot of each window row
    std::vector<double> corr_scale_;              // c_k per slot

    // overnight model
    std::vector<double> gap_hist_;  // ring buffer, sessions x n
    size_t gap_head_ = 0, gap_count_ = 0;
    std::vector<double> on_var_;    // next-gap variance per asset
    std::vector<double> on_corr_;   // n x n
    std::vector<double> on_gaps_, on_sd_;  // chronological gaps and their one-step-ahead sd (G x n)
    size_t on_rows_ = 0;
    double on_nu_ = 0.0;            // pooled per-asset tail (fallback)
    std::vector<char> on_proxy_;    // asset without enough gaps: proxy variance
    std::vector<char> on_core_;     // asset in the estimated gap correlation (enough common gaps)
    double on_lambda_ = 0.0;
    bool on_ready_ = false;

    std::vector<double> z_;               // on_bar workspace
    // Bars an asset missed since its last print in this session: its P&L is marked from that
    // last trade, so the next-bar forecast adds their expected variance (diurnal included).
    std::vector<double> stale_var_;
    std::vector<char> seen_;              // printed in the current session
    int last_slot_ = std::numeric_limits<int>::max();  // slot <= last_slot_: a new session
    mutable std::vector<double> inv_sd_;  // normalize_corr workspace (callers hold the lock)
};

}  // namespace garcht
