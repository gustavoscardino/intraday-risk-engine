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
#include <functional>
#include <limits>
#include <memory>
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

// The cycle above, one bar at a time: run_replay drives it over a whole panel, run_live over bars
// assembled from a feed as they arrive. A step only reads the panel up to its own bar.
class ReplaySession {
public:
    // in.bars is not read (the bars come with each step); session: the grid of the bars.
    ReplaySession(const ReplayInput& in, const ReplayConfig& cfg, const SessionSpec& session);
    ~ReplaySession();
    ReplaySession(const ReplaySession&) = delete;
    ReplaySession& operator=(const ReplaySession&) = delete;

    // Bar t of `panel` has closed. ret / slot / gaps: bar_returns and overnight_returns of the panel
    // (both causal: their row t depends on rows <= t only). eod: t is the last on-grid bar of its
    // session, so the daily refit runs before the forecast and the overnight forecast after it.
    void step(const BarPanel& panel, const std::vector<double>& ret, const std::vector<int>& slot,
              const std::vector<double>& gaps, size_t t, bool eod);
    // A feed learns that a session ended early (halt, truncated feed) only when the next one
    // starts: this runs the end-of-session work (daily refit, overnight forecast) for the last bar
    // stepped, if it has not run yet. The forecast already issued at that bar targets a bar that
    // never prints, so the scored forecasts are the same as with eod known in advance.
    void end_session(const BarPanel& panel);
    // True between the first bar of a session and its end-of-session work.
    bool session_open() const;
    // The forecast issued at the last bar stepped (its outcome still unknown); null if none.
    const ForecastRecord* issued() const;
    // The result so far, with the engine's state after the last bar.
    ReplayResult finish();

private:
    struct State;
    std::unique_ptr<State> s_;
};

// Live replay from a market-data feed (data/tools/feed_server.py). The history in `history.bars`
// is stepped first (calibration); then each line of the feed comes from next_line until it
// returns false or the feed sends #end:
//     timestamp,ticker,open,high,low,close,volume   header (columns in any order)
//     <row>                                          one bar of one stock
//     #clock <timestamp>                             every bar closing at or before it is complete
// A bar is assembled from its rows and stepped when the feed's clock passes it; a row of a bar
// already stepped is late and dropped (the feed's problem, counted), a repeated (bar, stock) keeps
// the first row, rows not after the history are counted and ignored. The session grid comes from
// the history (cfg.detect_session) or cfg.session. on_bar(panel, t, issued) runs after each live
// on-grid bar, with the forecast issued at it (null while the engine is not ready).
struct LiveStats {
    size_t rows = 0, bars = 0, late = 0, duplicates = 0, unknown = 0, bad = 0, before_history = 0;
};
ReplayResult run_live(const ReplayInput& history, const ReplayConfig& cfg,
                      const std::function<bool(std::string&)>& next_line, LiveStats& stats,
                      const std::function<void(const BarPanel&, size_t, const ForecastRecord*)>& on_bar = {},
                      int exchange_utc_offset_min = -180);

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
