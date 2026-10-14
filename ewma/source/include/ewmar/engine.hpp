// Portfolio risk engine with three EWMA models side by side:
//
//   EWMA-N (ref)  RiskMetrics: lambda = 0.94 fixed, raw returns, Normal distribution
//   EWMA-N        lambda estimated every day, deseasonalized returns y = r / s_k, Normal
//   EWMA-t        the same sigma as EWMA-N, unit-variance Student t with nu estimated every day
//
// Cycle: on_bar() at every bar (O(n^2), microseconds) and refit() at the end of every session,
// which re-estimates the diurnal profile, lambda and nu on the most recent window and re-runs
// the filters over it. compute() gives the one-bar-ahead forecast for exposures x (BRL):
//   a_i = x_i * s_{k(t+1), i},  sigma = sqrt(a' S a)   (EWMA-N / EWMA-t)
//   sigma_ref = sqrt(x' S_ref x)                        (EWMA-N (ref))
// compute_overnight() gives close -> end of the next first bar: an EWMA covariance of the
// opening gaps (its own lambda) plus the first bar of the next day, assumed independent.
#pragma once

#include <cstddef>
#include <vector>

#include "ewmar/ewma.hpp"
#include "ewmar/risk.hpp"
#include "ewmar/seasonality.hpp"

namespace ewmar {

struct EngineConfig {
    int bars_per_day = 28;
    int window_bars = 60 * 28;  // estimation window (bars)
    double ref_lambda = 0.94;   // EWMA-N (ref)
    bool estimate_lambda = true;
    double lambda = 0.94;  // used when estimate_lambda is false (and as the first value)
    double lambda_lo = 0.80, lambda_hi = 0.999;
    bool estimate_nu = true;
    double nu = 8.0;  // used when estimate_nu is false (and while there is too little data)
    double nu_lo = 2.5, nu_hi = 60.0;
    DiurnalMode diurnal = DiurnalMode::Pooled;
    bool overnight = true;
    int overnight_window = 250;  // sessions of opening gaps kept
    std::vector<double> confidences{0.95, 0.975, 0.99};
};

struct ModelForecast {
    double sigma = kNaN;
    std::vector<RiskMeasure> m;  // one per confidence level
};

struct PortfolioRisk {
    bool ready = false;
    ModelForecast ref, normal, student;
    double lambda = kNaN, nu = kNaN;
    std::vector<double> asset_sigma;  // bar volatility of each asset (return units), EWMA-N
    std::vector<double> contrib;      // Euler contribution of each asset to sigma (sums to sigma)
    std::vector<double> standalone;   // sigma of each position alone
};

struct OvernightRisk {
    bool available = false;
    double sigma_gap = kNaN, sigma_bar = kNaN, sigma = kNaN, lambda = kNaN, nu = kNaN;
    std::vector<RiskMeasure> normal, student;
};

struct RefitReport {
    bool ready = false;
    double lambda = kNaN, nu = kNaN, lambda_on = kNaN, nu_on = kNaN;
    size_t nu_obs = 0, nu_on_obs = 0;
    double seconds = 0.0;
};

class EwmaEngine {
public:
    EwmaEngine(size_t assets, const EngineConfig& cfg);

    // Raw returns of one bar (NaN = no valid return). Bars that never printed are fed as all-NaN
    // rows, so the history advances one step per grid slot.
    void on_bar(int slot, const double* returns);
    // Undoes the last on_bar (filters and history), if no refit happened since. Used by the
    // data-quality layer to re-feed a bar once a suspicious price is confirmed or released.
    bool undo_last_bar();
    // Replaces one value of the newest history row (seen by the next refit only).
    void amend_last_bar(size_t asset, double value);
    // Opening gaps of a new session (NaN = no gap).
    void on_overnight(const double* gaps);

    // End-of-session re-estimation; exposures (BRL) weight the nu estimate. Needs a full window.
    RefitReport refit(const double* exposures);

    void compute(int next_slot, const double* exposures, PortfolioRisk& out) const;
    OvernightRisk compute_overnight(const double* exposures) const;

    // Bar volatility of one asset at a slot, in return units (EWMA-N), for the data-quality layer.
    double asset_sigma(size_t asset, int slot) const;

    bool ready() const { return ready_; }
    size_t history_size() const { return count_; }
    size_t assets() const { return n_; }
    double lambda() const { return lambda_; }
    double nu() const { return nu_; }
    const DiurnalProfile& diurnal() const { return diurnal_; }
    std::vector<double> correlation() const;
    const EngineConfig& config() const { return cfg_; }

private:
    void refit_overnight(const double* exposures, RefitReport& rep);
    void chronological(std::vector<double>& rows, std::vector<int>& slots) const;

    size_t n_;
    EngineConfig cfg_;
    bool ready_ = false;
    double lambda_, nu_;
    DiurnalProfile diurnal_;
    EwmaCovariance Sy_, Sref_;
    std::vector<TailFactors> fac_n_, fac_t_, fac_on_t_;

    // history ring of raw returns
    size_t cap_, head_ = 0, count_ = 0;
    std::vector<double> hist_;
    std::vector<int> hist_slot_;

    // one-step undo
    bool can_undo_ = false;
    std::vector<double> undo_Sy_, undo_Sref_, undo_row_;
    int undo_slot_ = 0;
    size_t undo_count_ = 0;

    // overnight
    size_t on_cap_, on_head_ = 0, on_count_ = 0;
    std::vector<double> gaps_;
    bool on_ready_ = false;
    double on_lambda_ = 0.94, on_nu_ = 8.0;
    EwmaCovariance Sgap_;

    mutable std::vector<double> a_, Sa_;  // scratch
};

}  // namespace ewmar
