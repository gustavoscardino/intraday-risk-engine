// Causal filter of isolated price jumps (bad ticks) on a bar panel.
//
// A wrong print -- one close far off, the next bar back to normal -- is the feed problem that hurts
// a GARCH most: the tick and its reversal enter as two extreme shocks and inflate the variance for
// many bars. Clipping the shocks (EngineConfig::z_clip) only shrinks them. This filter removes them:
//
//   * a stock whose return jumps beyond z times the robust scale of its own recent returns, while
//     fewer than max(market_assets, market_share * n) stocks jump at the same bar, is HELD: its close
//     at that bar is treated as not printed (as during a one-bar halt);
//   * at its next bar in the session, if the close undid more than revert_share of the jump, the
//     held print was a BAD_TICK and never enters the model; otherwise the new level is genuine
//     (SHOCK_ACCEPTED) and enters now, one bar late, as the return since the last good close;
//   * when many stocks jump together it is a MARKET_MOVE and every return is kept at once.
//
// The decision on bar t uses bars <= t only, so the filter runs the same way on a file and on a
// live feed. Its cost is the delay of a genuine move confined to one stock (measured in atam/).
#pragma once

#include <cstddef>
#include <deque>
#include <string>
#include <vector>

#include "garcht/market_data.hpp"

namespace garcht {

// The scale of a return is 1.4826 x median |recent return| of the stock, times the intraday factor of
// its slot: the median of |return| / stock median at that slot over all stocks' recent bars (the
// opening bars move more than midday ones). Both use returns kept by the filter only.
struct JumpFilterConfig {
    double z = 10.0;            // jump: |return| > z x scale
    int lookback = 280;         // recent on-grid returns per stock for the scale (10 sessions of 28 bars)
    int slot_lookback = 300;    // recent normalized returns per slot (all stocks) for the intraday factor
    int min_slot_returns = 30;  // intraday factor 1 until a slot has this many
    int min_returns = 56;       // no decision on a stock with fewer returns (2 sessions)
    int market_assets = 3;      // at least max(market_assets, market_share x n) stocks jumping at the
    double market_share = 0.2;  //   same bar: a market move, kept
    double revert_share = 0.5;  // a held jump is a bad tick if the next close undoes more than this share
};

struct JumpEvent {
    std::string time, ticker;
    std::string action;  // HELD, BAD_TICK, SHOCK_ACCEPTED, MARKET_MOVE
    double ret = 0.0;    // log-return of the jump (since the last good close)
    double z = 0.0;      // |ret| / robust scale
};

class JumpFilter {
public:
    JumpFilter(size_t n_assets, const SessionSpec& session, JumpFilterConfig cfg = {});
    // Decides on bar t (bars before t already decided, in order). A held jump sets the stock's close
    // at bar t to NaN in `panel`. Reads rows <= t only; changes row t only.
    void apply(BarPanel& panel, size_t t);
    const std::vector<JumpEvent>& events() const { return events_; }

private:
    struct Asset {
        double last_close = 0.0;  // last close kept (not held), and its session date
        int last_date = -1;
        std::deque<double> abs_ret;  // recent |returns| kept
        bool pending = false;        // held at the previous on-grid bar of the session
        double held_close = 0.0, held_ret = 0.0;
    };
    double median_abs(const Asset& a) const;  // NaN until min_returns
    double slot_factor(int slot) const;

    size_t n_;
    SessionSpec session_;
    JumpFilterConfig cfg_;
    std::vector<Asset> assets_;
    std::vector<std::deque<double>> slot_norm_;  // per slot: |return| / median |return| of its stock
    std::vector<JumpEvent> events_;
};

}  // namespace garcht
