// The jump filter on a bar panel: the shared causal filter of isolated price jumps
// (common/include/riskcommon/jump_filter.hpp, used by the three engines) applied to the rows of a
// BarPanel. See that header for the rules (HELD, BAD_TICK, SHOCK_ACCEPTED, MARKET_MOVE, UNDECIDED).
#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "garcht/market_data.hpp"
#include "riskcommon/jump_filter.hpp"

namespace garcht {

using JumpFilterConfig = riskcommon::JumpFilterConfig;
using JumpEvent = riskcommon::JumpEvent;

class JumpFilter {
public:
    JumpFilter(size_t n_assets, const SessionSpec& session, JumpFilterConfig cfg = {});
    // Decides on bar t (bars before t already decided, in order). A held jump sets the stock's close
    // at bar t to NaN in `panel`. Reads rows <= t only; changes row t only.
    void apply(BarPanel& panel, size_t t);
    // The session of the last bar applied is over: re-estimates the robust scales (optional).
    void end_session();
    const std::vector<JumpEvent>& events() const;
    // bar_returns spreads a return across a missing close over the slots it spans (scaled by
    // 1/sqrt(span)); an accepted jump happened in one bar, so this puts the whole move since the
    // last good close back into `ret` (bar_returns of the filtered panel) at full size.
    void restore_accepted(std::vector<double>& ret) const;
    riskcommon::JumpFilter::Counts counts() const;
    // Moves let in one bar late, in decision order (see riskcommon::JumpFilter::accepted).
    const std::vector<riskcommon::JumpFilter::Accepted>& accepted() const;

private:
    size_t n_;
    SessionSpec session_;
    JumpFilterConfig cfg_;
    std::unique_ptr<riskcommon::JumpFilter> core_;  // built at the first bar (the panel names the stocks)
};

}  // namespace garcht
