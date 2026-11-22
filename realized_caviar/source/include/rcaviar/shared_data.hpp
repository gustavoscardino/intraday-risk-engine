#pragma once
#include "rcaviar/data.hpp"
#include "garcht/market_data.hpp"
#include "garcht/backtest.hpp"
#include "garcht/data_quality.hpp"
namespace rcaviar {
// Four consecutive intraday returns form the information window; the next
// bar is the target. Missing features and overnight windows are excluded.
// A missing target is NaN; its features still produce a forecast.
// `panel` is the models' view of the prices. With `printed` (the prices as received, same rows),
// each window also carries the target as printed (scored_return); `returns` replaces the bar
// returns of `panel` (the jump filter puts a move let in one bar late back at full size).
std::vector<Window> make_bar_windows(const garcht::BarPanel& panel, std::size_t asset,
                                     const garcht::BarPanel* printed = nullptr,
                                     const std::vector<double>* returns = nullptr);
struct SharedData {
    std::vector<Window> windows;
    std::int64_t cutoff;
    std::size_t sessions, training_sessions, bad_rows, duplicates;
    // Every observed grid bar, including warmup and missing observations (NaN).
    std::vector<std::int64_t> return_times, session_starts;
    std::vector<double> returns;
    std::vector<int> return_slots;  // session slot of each return (0 = first bar)
    int bars_per_day = 0;
    std::string protection;  // data protection applied, for the report
    std::vector<garcht::JumpEvent> jump_events;  // decisions of the jump filter (every stock)
};
// jump_filter: the causal filter shared by the three engines (common/include/riskcommon/
// jump_filter.hpp) runs over every stock of the file, so its market-move rule sees the whole
// market; the models see the filtered prices, the forecasts are scored on the printed ones.
SharedData read_shared_data(const std::string& folder, const std::string& ticker, int window_days = 0,
                            const std::string& prices_path = {}, bool jump_filter = true);
// Pool transition counts only where target intervals are adjacent. This avoids
// creating artificial exception transitions across sessions or missing windows.
garcht::CoverageTest coverage_with_boundaries(const std::vector<int>& hits,
                                            const std::vector<bool>& adjacent, double tau);
}
