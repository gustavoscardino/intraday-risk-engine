// Replay of a bar history through the risk engine, emulating the 15-minute cycle, and the
// backtest of its forecasts. Shared by the intraday_risk and monte_carlo programs.
//
// Cycle at each bar close t:
//   1. realized P&L of the bar that just closed, with the positions held at the previous
//      forecast (first bar of a session: from the open, overnight gap excluded) -> backtest
//      and out-of-sample tail calibration;
//   2. at a session start, the overnight gaps feed the overnight model; the close -> first
//      bar P&L closes the overnight forecast issued at the previous close;
//   3. client fills up to the bar time update the positions;
//   4. engine update (filters, correlations, baseline); end of session: full re-estimation;
//   5. one-bar-ahead sigma / VaR / ES; with an option book, Monte Carlo full revaluation and
//      the delta-linear approximation; at the close, the overnight forecast.
#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "garcht/backtest.hpp"
#include "garcht/market_data.hpp"
#include "garcht/portfolio.hpp"
#include "garcht/synthetic.hpp"

namespace garcht {

struct ReplayConfig {
    EngineConfig engine;          // bars_per_day / window_bars are derived from the session
    int window_days = 60;
    bool detect_session = true;   // infer the session grid from the data
    SessionSpec session;          // used when detect_session is false
    bool overnight = true;        // forecast and backtest close -> end of next first bar
    int mc_scenarios = 10000;     // option book: Monte Carlo scenarios per bar
    std::uint64_t mc_seed = 7;
    bool record_params = true;    // keep per-asset parameters of every refit
};

struct ReplayInput {
    std::vector<std::string> tickers;
    BarPanel bars;
    std::vector<double> initial_quantity;  // per ticker
    std::vector<Fill> fills;               // any order
    std::vector<OptionContract> options;   // optional book of European options
};

constexpr double kMissing = std::numeric_limits<double>::quiet_NaN();

struct ForecastRecord {
    std::string forecast_time, target_time;
    int target_slot = 0;
    double net = 0.0, gross = 0.0;  // stock exposures at forecast time
    double mean = 0.0, sigma = 0.0, nu = 0.0, ewma_sigma = 0.0;
    ShockDist tail;  // standardized distribution behind the GARCH VaR / ES
    std::vector<RiskMeasure> garch, ewma;
    double pnl = kMissing;  // realized P&L of the stock positions over the target bar
    int models_fitted = 0;
    double update_us = 0.0;
    // option book (stocks + options); empty measures if there is no book
    std::vector<RiskMeasure> book_mc, book_delta;
    double book_mc_sigma = 0.0, book_delta_sigma = 0.0;
    double book_pnl = kMissing;
    double mc_us = 0.0;
};

struct OvernightRecord {
    std::string forecast_time, target_time;
    double net = 0.0;
    OvernightRisk risk;
    double pnl = kMissing;
};

struct ParamRecord {
    std::string date, ticker;
    ModelSource source = ModelSource::Fallback;
    FitStatus status = FitStatus::NumericalError;
    bool active = true;
    GarchParams params, robust_se;
    double loglik = 0.0;
    int iterations = 0;
};

struct DailyRecord {
    std::string date;
    double nu_p = 0.0, dcc_a = 0.0, dcc_b = 0.0, refit_ms = 0.0;
    int n_fitted = 0, n_inactive = 0;
    bool dcc_estimated = false;
};

struct ReplayResult {
    SessionSpec session;
    bool student_t = true;
    std::string model_name;  // "GARCH-t/EVT", "GARCH-t" or "GARCH-N"
    std::vector<ForecastRecord> forecasts;  // realized ones (the last forecast has no outcome)
    std::vector<OvernightRecord> overnight;
    std::vector<ParamRecord> params;
    std::vector<DailyRecord> daily;
    std::vector<double> latency_us, refit_ms;
    size_t bars = 0, sessions = 0, off_grid = 0, fills_applied = 0, fills_unknown = 0;
    size_t unscored = 0;            // forecasts whose target bar never printed (halts, gaps)
    size_t unscored_overnight = 0;  // overnight forecasts followed by a late opening
    double engine_seconds = 0.0, wall_seconds = 0.0;
    // state after the last bar
    std::string last_time;
    std::vector<double> quantity, last_price, exposure;
    PortfolioRisk last_risk;
    std::vector<double> correlation, diurnal, corr_scale;
    std::vector<AssetModel> models;
};

// Throws std::runtime_error if the data has no more sessions than the window.
ReplayResult run_replay(const ReplayInput& in, const ReplayConfig& cfg);

struct ModelBacktest {
    std::string model;  // e.g. "GARCH-t", "EWMA-N (ref)", "overnight", "book MC", "book delta"
    double confidence = 0.0;
    CoverageTest coverage;
    CoverageTest by_time[3];  // open (first 3 bars) / midday / close (last 3 bars)
    bool has_by_time = false;
    double es_ratio = kMissing;  // mean realized loss / mean forecast ES over exceptions
    EsTest es;
    bool has_es = false;
};

struct BacktestReport {
    std::vector<ModelBacktest> rows;
    BerkowitzTest berkowitz_garch, berkowitz_ewma, berkowitz_overnight;
};

BacktestReport evaluate_backtest(const ReplayResult& r, const std::vector<double>& confidences,
                                 int es_sims = 2000);

}  // namespace garcht
