// Replay of a price history through the data-quality layer and the EWMA engine, emulating the
// 15-minute cycle, and the backtest of its forecasts.
//
// At each bar t:
//   1. the outcome of bar t-1 is scored: P&L of the positions held at the forecast, from the
//      previous close of the session (first bar: from the open, overnight gap excluded), with
//      the prices received (after the arrival rules and the STALE mask); at a session's first bar
//      the overnight forecast is scored as well;
//   2. fills up to the bar time update the positions; the jump filter shared by the three
//      engines decides the bar's prints (SUSPECT / SPIKE / SHOCK_KEPT) and gives the models' view
//      of the prices; returns of that view; dq_score and mode; update of every model (FREEZE: no
//      update); end of session: re-estimation;
//   3. one-bar-ahead sigma / VaR / ES of every model (FREEZE: the last valid forecast is kept);
//      with options, the delta-linear book; at the close, the overnight forecast.
// Exposures are marked at the models' view (a held print keeps the last validated close). The
// windows, the first forecast bar, the conventions and the scoring are those of the GARCH-t engine.
#pragma once

#include <deque>
#include <string>
#include <vector>

#include "ewmar/backtest.hpp"
#include "ewmar/data_quality.hpp"
#include "ewmar/engine.hpp"
#include "ewmar/market_data.hpp"

namespace ewmar {

struct ReplayConfig {
    EngineConfig engine;  // bars_per_day and window_bars are derived from the session
    int window_days = 60;
    bool detect_session = true;
    SessionSpec session;  // used when detect_session is false
    bool dq = true;
    DqConfig dq_cfg;
};

struct ReplayInput {
    std::vector<std::string> tickers;
    FeedFile feed;
    std::vector<double> initial_quantity;
    std::vector<Fill> fills;
    std::vector<OptionContract> options;
};

struct ForecastRecord {
    std::string forecast_time, target_time;
    int target_slot = 0;
    double net = 0.0, gross = 0.0;
    std::vector<riskcommon::ModelForecast> models;  // one per ReplayResult::var_models entry
    double pnl = kNaN;
    double dq_score = 1.0;
    DqMode mode = DqMode::Normal;
    bool frozen = false;  // FREEZE: the last valid forecast was kept
    double update_us = 0.0;
    double book_sigma = kNaN, book_pnl = kNaN;  // delta-linear option book (main model)
    std::vector<RiskMeasure> book;
};

struct OvernightRecord {
    std::string forecast_time, target_time;
    double net = 0.0;
    OvernightRisk risk;
    double pnl = kNaN;
};

struct DailyRecord {
    std::string date;
    double lambda = kNaN, nu = kNaN, lambda_on = kNaN, nu_on = kNaN;
    size_t nu_obs = 0;
    double refit_ms = 0.0;
};

struct ReplayResult {
    SessionSpec session;
    size_t sessions = 0, bars = 0, unscored = 0, unscored_overnight = 0, fills_applied = 0, fills_unknown = 0;
    PanelStats panel_stats;  // without the DQ layer
    bool dq = false, has_book = false;
    std::vector<riskcommon::ModelInfo> var_models;  // the VaR models, in registration order
    size_t main_model = 0;                          // the one behind the book and the decomposition
    std::vector<riskcommon::JumpEvent> jump_events;  // decisions of the jump filter (DQ layer on)
    IngestResult ingest;  // with the DQ layer (panel moved out)
    std::vector<DqEvent> events;
    std::vector<DqAlert> alerts;
    size_t mode_bars[3] = {0, 0, 0};
    size_t suspects = 0, spikes = 0, shocks_kept = 0, unresolved = 0, market_wide = 0, frozen_forecasts = 0;
    std::vector<ForecastRecord> forecasts;  // scored forecasts
    std::vector<OvernightRecord> overnight;
    std::vector<DailyRecord> daily;
    std::vector<double> latency_us, refit_ms;
    double engine_seconds = 0.0, wall_seconds = 0.0;
    // state after the last bar
    std::string last_time;
    std::vector<double> quantity, last_price, exposure;
    PortfolioRisk last_risk;
    std::vector<double> correlation, diurnal_pooled, diurnal;  // diurnal: [slot][asset]
    double lambda = kNaN, nu = kNaN;
};

// Throws std::runtime_error if the data has no more sessions than the window.
ReplayResult run_replay(const ReplayInput& in, const ReplayConfig& cfg);

struct ModelBacktest {
    std::string model;
    double confidence = 0.0;
    CoverageTest coverage;
    CoverageTest by_time[3];  // open (first 3 bars) / midday / close (last 3 bars)
    bool has_by_time = false;
    double es_ratio = kNaN;  // mean loss over exceptions / mean forecast ES over exceptions
    EsTest es;
    bool has_es = false;
};

struct NamedBerkowitz {
    std::string model;
    BerkowitzTest test;
};

struct BacktestReport {
    std::vector<ModelBacktest> rows;
    std::vector<NamedBerkowitz> berkowitz;
};

BacktestReport evaluate_backtest(const ReplayResult& r, const std::vector<double>& confidences, int es_sims = 2000);

}  // namespace ewmar
