// Portfolio risk engine with the EWMA models of ewmar/models.hpp side by side (EWMA-N (ref),
// EWMA-N, EWMA-t), driven through the common model interface.
//
// Cycle: on_bar() at every bar (O(n^2), microseconds) and refit() at the end of every session,
// which re-estimates every model on the most recent window. compute() gives the one-bar-ahead
// forecast of every model for exposures x (money), plus the risk decomposition of the EWMA-N
// volatility. compute_overnight() gives close -> end of the next first bar: an EWMA covariance of
// the opening gaps (its own lambda) plus the first bar of the next day (EWMA-N), assumed
// independent.
#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "ewmar/ewma.hpp"
#include "ewmar/models.hpp"
#include "ewmar/risk.hpp"
#include "ewmar/seasonality.hpp"
#include "riskcommon/model.hpp"

namespace ewmar {

struct EngineConfig : ModelConfig {
    int window_bars = 40 * 28;  // estimation window (bars)
    bool overnight = true;
    int overnight_window = 250;  // sessions of opening gaps kept
};

struct PortfolioRisk {
    bool ready = false;
    std::vector<riskcommon::ModelForecast> models;  // one per EwmaEngine::models() entry
    std::vector<double> asset_sigma;  // bar volatility of each asset (return units), EWMA-N
    std::vector<double> contrib;      // Euler contribution of each asset to sigma (sums to sigma)
    std::vector<double> standalone;   // sigma of each position alone
};

struct OvernightRisk {
    bool available = false;
    double sigma_gap = kNaN, sigma_bar = kNaN, sigma = kNaN, lambda = kNaN, nu = kNaN;
    std::vector<RiskMeasure> normal, student;  // the two tails on the same overnight sigma
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
    // Opening gaps of a new session (NaN = no gap).
    void on_overnight(const double* gaps);

    // End-of-session re-estimation; exposures (money) weight the nu estimates. Needs a full window.
    RefitReport refit(const double* exposures);

    void compute(int next_slot, const double* exposures, PortfolioRisk& out) const;
    OvernightRisk compute_overnight(const double* exposures) const;

    const std::vector<riskcommon::ModelInfo>& models() const { return info_; }
    size_t main_model() const { return main_; }
    bool ready() const { return ready_; }
    size_t history_size() const { return history_.size(); }
    size_t assets() const { return n_; }
    double lambda() const;  // of the EWMA-N volatility
    double nu() const;      // of the EWMA-t tail (NaN if not registered)
    const DiurnalProfile& diurnal() const;
    std::vector<double> correlation() const;
    const EngineConfig& config() const { return cfg_; }

private:
    void refit_overnight(const double* exposures, RefitReport& rep);

    size_t n_;
    EngineConfig cfg_;
    bool ready_ = false;
    std::vector<std::unique_ptr<riskcommon::PortfolioModel>> models_;
    std::vector<riskcommon::ModelInfo> info_;
    size_t main_ = 0;
    const EwmaNormalModel* vol_ = nullptr;    // the EWMA-N volatility (decomposition, overnight)
    const EwmaStudentModel* tail_ = nullptr;  // the EWMA-t tail (overnight fallback nu)
    riskcommon::HistoryRing history_;
    std::vector<TailFactors> fac_n_, fac_on_t_;

    // overnight
    size_t on_cap_, on_head_ = 0, on_count_ = 0;
    std::vector<double> gaps_;
    bool on_ready_ = false;
    double on_lambda_ = 0.94, on_nu_ = 8.0;
    EwmaCovariance Sgap_;
};

}  // namespace ewmar
