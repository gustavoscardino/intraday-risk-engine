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
//   4. update of every model (garcht/models.hpp: GARCH-t/EVT, the RiskMetrics baseline); end of
//      session: full re-estimation;
//   5. one-bar-ahead sigma / VaR / ES of every model; with an option book, the delta-linear
//      approximation and the Monte Carlo full revaluation (on a worker thread, after the VaR is
//      published: it is not part of the bar update); at the close, the overnight forecast.
// Exposures are marked at the model's view of the prices (with the jump filter, a held print keeps
// the last validated close); the forecasts are scored on the P&L of the prices as printed.
#pragma once

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "garcht/backtest.hpp"
#include "garcht/data_quality.hpp"
#include "garcht/market_data.hpp"
#include "garcht/models.hpp"
#include "garcht/portfolio.hpp"
#include "garcht/synthetic.hpp"
#include "riskcommon/model.hpp"

namespace garcht {

struct ReplayConfig {
    EngineConfig engine;          // bars_per_day / window_bars are derived from the session
    int window_days = 40;          // estimation window in sessions (the same in every engine)
    bool detect_session = true;   // infer the session grid from the data
    SessionSpec session;          // used when detect_session is false
    bool overnight = true;        // forecast and backtest close -> end of next first bar
    int mc_scenarios = 10000;     // option book: Monte Carlo scenarios per bar
    std::uint64_t mc_seed = 7;
    bool record_params = true;    // keep per-asset parameters of every refit
    bool jump_filter = false;     // hold isolated price jumps until the next bar (data_quality.hpp)
    bool mc_async = true;         // option book: Monte Carlo revaluation off the bar thread (see ReplaySession)
    JumpFilterConfig jump;
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
    std::vector<riskcommon::ModelForecast> models;  // one per ReplayResult::var_models entry
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
    std::vector<riskcommon::ModelInfo> var_models;  // the VaR models, main model first
    std::string model_name;  // name of the main model: "GARCH-t/EVT", "GARCH-t" or "GARCH-N"
    std::vector<ForecastRecord> forecasts;  // realized ones (the last forecast has no outcome)
    std::vector<OvernightRecord> overnight;
    std::vector<ParamRecord> params;
    std::vector<DailyRecord> daily;
    // latency_us: bar update (fills, filters, correlations, VaR/ES) of bars without a refit;
    // rebalance_us: tail re-calibration after a trade (timed apart: it runs on trade bars only).
    std::vector<double> latency_us, refit_ms, rebalance_us;  // latency: jump filter + bar update
    size_t bars = 0, sessions = 0, off_grid = 0, fills_applied = 0, fills_unknown = 0;
    size_t unscored = 0;            // forecasts whose target bar never printed (halts, gaps)
    size_t unscored_overnight = 0;  // overnight forecasts followed by a late opening
    std::vector<JumpEvent> jump_events;  // decisions of the jump filter (if on)
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

    // Bar t of `panel` has closed. ret / slot / gaps: bar_returns and overnight_returns of the
    // model's view (`model_view`, e.g. the jump-filtered copy; null = `panel` itself). The forecasts
    // are scored on the P&L of `panel` (the prices as printed) and the positions marked at its
    // prices; the tail calibration learns from the P&L of the model's view. Both inputs are causal:
    // their row t depends on rows <= t only. eod: t is the last on-grid bar of its
    // session, so the daily refit runs before the forecast and the overnight forecast after it.
    // input_us: time spent preparing this bar's input (the jump filter's decision), counted in the
    // bar's latency, so that the per-bar figure has the boundary of the other engines.
    void step(const BarPanel& panel, const std::vector<double>& ret, const std::vector<int>& slot,
              const std::vector<double>& gaps, size_t t, bool eod, const BarPanel* model_view = nullptr,
              double input_us = 0.0);
    // A feed learns that a session ended early (halt, truncated feed) only when the next one
    // starts: this runs the end-of-session work (daily refit, overnight forecast) for the last bar
    // stepped, if it has not run yet. The forecast already issued at that bar targets a bar that
    // never prints, so the scored forecasts are the same as with eod known in advance.
    void end_session(const BarPanel& panel);
    // True between the first bar of a session and its end-of-session work.
    bool session_open() const;
    // The forecast issued at the last bar stepped (its outcome still unknown); null if none. Waits
    // for its option revaluation, if one is running.
    const ForecastRecord* issued();
    // The result so far, with the engine's state after the last bar.
    ReplayResult finish();
    // Offline replays (run_replay): the option revaluations, which in operation run on a worker thread
    // between two bars 15 minutes apart, are queued with the inputs of their bar and run after the
    // bars, in bar order (the same numbers). A compressed replay would otherwise either run them
    // while the next bars are timed or make every bar wait for them, and time neither situation of
    // operation. Live (run_live) they run on a worker thread right after each bar's VaR.
    void defer_revaluations();

private:
    struct State;
    std::unique_ptr<State> s_;
};

// Live replay from a market-data feed (data/tools/feed_server.py). The history in `history.bars`
// is stepped first (calibration); then each line of the feed comes from next_line until it
// returns false or the feed sends #end:
//     timestamp,ticker,open,high,low,close,volume   header (columns in any order; the CSV
//                                                    conventions of the file reader)
//     <row>                                          one bar of one stock
//     #clock <timestamp>                             every bar closing at or before it is complete
// A bar is assembled from its rows and stepped when the feed's clock passes it; a row of a bar
// already stepped is late and dropped (the feed's problem, counted), a repeated (bar, stock) keeps
// the first row, rows not after the history are counted and ignored. The session grid comes from
// the history (cfg.detect_session) or cfg.session. on_bar(panel, t, issued) runs after each live
// on-grid bar, with the forecast issued at it (null while the engine is not ready).
// One line of a feed split into fields the way the file reader splits a CSV (quoted fields, a
// UTF-8 BOM; ',' ';' or tab). delimiter 0 detects it (pass the header line first); it is then kept.
std::vector<std::string> split_feed_line(const std::string& line, char& delimiter);

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

struct NamedBerkowitz {
    std::string model;
    BerkowitzTest test;
};

struct BacktestReport {
    std::vector<ModelBacktest> rows;
    std::vector<NamedBerkowitz> berkowitz;  // one per VaR model, in ReplayResult::var_models order
    BerkowitzTest berkowitz_overnight;
};

BacktestReport evaluate_backtest(const ReplayResult& r, const std::vector<double>& confidences,
                                 int es_sims = 2000);

}  // namespace garcht
