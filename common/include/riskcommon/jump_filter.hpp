// Causal filter of isolated price jumps (bad ticks), shared by the three engines.
//
// A wrong print -- one close far off, the next bar back to normal -- is the feed problem that hurts
// a volatility model most: the tick and its reversal enter as two extreme returns and inflate the
// variance (or a realized measure) for many bars. Clipping the returns only shrinks them. This
// filter removes them:
//
//   * a stock whose return jumps beyond z times the robust scale of its own recent returns, while
//     fewer than max(market_assets, market_share * n) stocks jump at the same bar, is HELD: its close
//     at that bar is treated as not printed (as during a one-bar halt);
//   * at its next bar in the session, if the close undid more than revert_share of the jump, the
//     held print was a BAD_TICK and never enters the model; otherwise the new level is genuine
//     (SHOCK_ACCEPTED) and enters now, one bar late, as the return since the last good close;
//   * when many stocks jump together it is a MARKET_MOVE and every return is kept at once;
//   * a jump held at the last bar of a session is UNDECIDED: it never enters the intraday model,
//     and the next overnight gap starts from the last close kept.
//
// The robust scales are calibrated on the sessions completed so far, like every other parameter of
// the engines: they are re-estimated when a session ends (end_session(), or the first bar of the
// next one) and stay fixed during the session, so a bar costs O(n). The decision on a bar uses that
// bar and earlier ones only, so the filter runs the same way on a file and on a live feed. It
// changes what the models see (the "model view" of the prices), never the backtest: the engines
// score their forecasts on the prices as printed. Its cost is the delay of a genuine move confined
// to one stock by one bar (measured in atam/).
#pragma once

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace riskcommon {

// The scale of a return is 1.4826 x median |return| of the stock over its last lookback_sessions
// sessions, times the intraday factor of its slot: the median of |return| / stock median at that
// slot over all stocks' recent bars (the opening bars move more than midday ones). Both use the
// returns kept by the filter only.
struct JumpFilterConfig {
    double z = 10.0;             // jump: |return| > z x scale
    int lookback_sessions = 10;  // recent on-grid returns per stock for the scale
    int slot_lookback = 300;     // recent normalized returns per slot (all stocks) for the intraday factor
    int min_slot_returns = 30;   // intraday factor 1 until a slot has this many
    int min_sessions = 2;        // no decision on a stock with fewer sessions of returns
    int market_assets = 3;       // at least max(market_assets, market_share x n) stocks jumping at the
    double market_share = 0.2;   //   same bar: a market move, kept
    double revert_share = 0.5;   // a held jump is a bad tick if the next close undoes more than this share
};

struct JumpEvent {
    std::string time, ticker;
    std::string action;  // HELD, BAD_TICK, SHOCK_ACCEPTED, MARKET_MOVE, UNDECIDED
    double ret = 0.0;    // log-return of the jump (since the last good close)
    double z = 0.0;      // |ret| / robust scale
    size_t bar = 0, asset = 0;  // row passed to apply() and stock index
};

class JumpFilter {
public:
    JumpFilter(std::vector<std::string> tickers, int bars_per_day, JumpFilterConfig cfg = {});

    // Decides on one on-grid bar; bars come in time order. open / close: the n stocks' open and
    // close at this bar (NaN or <= 0 = not printed). A held print is set to NaN in `close`. `bar` is
    // the caller's row index of this bar (kept in the events and the accepted moves).
    void apply(size_t bar, const std::string& stamp, int date, int slot, const double* open, double* close);
    // The session is over: re-estimates the scales now (with the engines' end-of-day work) rather
    // than at the next session's first bar. Calling it is optional; the result is the same.
    void end_session();

    const std::vector<JumpEvent>& events() const { return events_; }
    // Genuine moves let in one bar late, decided at the bar `bar` for stock `asset`: `ret` is the
    // whole log-return since the last good close. A return computed across the held (missing)
    // close would be spread over two bars; the move happened in one, so engines put it back.
    struct Accepted {
        size_t bar, asset;
        double ret;
    };
    const std::vector<Accepted>& accepted() const { return accepted_; }
    // For engines that keep a bars x n matrix of returns of the filtered prices: puts every accepted
    // move back at full size (rows past the end of `ret` are ignored).
    void restore_accepted(std::vector<double>& ret) const;
    size_t assets() const { return n_; }
    const JumpFilterConfig& config() const { return cfg_; }

    // Counts of the decisions so far.
    struct Counts {
        size_t held = 0, bad_ticks = 0, accepted = 0, undecided = 0, market_moves = 0;
    };
    Counts counts() const;

private:
    // The last `cap` values in arrival order (a ring); the median is taken at calibration only.
    class Window {
    public:
        void push(double v, size_t cap);
        size_t size() const { return ring_.size(); }
        double median(std::vector<double>& scratch) const;  // upper median (element size / 2 in order)

    private:
        std::vector<double> ring_;
        size_t head_ = 0;  // oldest value once the ring is full
    };
    struct Asset {
        double last_close = 0.0;  // last close kept (not held), and its session date
        int last_date = -1;
        Window abs_ret;           // recent |returns| kept
        double scale_median = std::numeric_limits<double>::quiet_NaN();  // at the last calibration
        bool pending = false;     // held at the previous on-grid bar of the session
        double held_close = 0.0, held_ret = 0.0;
    };
    struct Slot {
        Window norm;          // |return| / scale median of its stock
        double factor = 1.0;  // at the last calibration
    };

    size_t n_;
    std::vector<std::string> tickers_;
    JumpFilterConfig cfg_;
    int lookback_, min_returns_;
    std::vector<Asset> assets_;
    std::vector<Slot> slots_;
    int date_ = -1;       // session of the last bar applied
    bool dirty_ = false;  // returns kept since the last calibration
    std::vector<JumpEvent> events_;
    std::vector<Accepted> accepted_;
    std::vector<double> ret_, zs_, scratch_;  // workspace
    std::vector<char> decided_;
    std::vector<size_t> jumps_;
};

}  // namespace riskcommon
