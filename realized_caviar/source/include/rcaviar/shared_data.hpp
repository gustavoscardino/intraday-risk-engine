#pragma once
#include "rcaviar/data.hpp"
#include "garcht/market_data.hpp"
#include "garcht/backtest.hpp"
namespace rcaviar {
// Four consecutive intraday returns form the information window; the next
// bar is the target. Missing features and overnight windows are excluded.
// A missing target is NaN; its features still produce a forecast.
std::vector<Window> make_bar_windows(const garcht::BarPanel& panel, std::size_t asset);
struct SharedData {
    std::vector<Window> windows;
    std::int64_t cutoff;
    std::size_t sessions, training_sessions, bad_rows, duplicates;
};
SharedData read_shared_data(const std::string& folder, const std::string& ticker, int window_days = 0);
// Pool transition counts only where target intervals are adjacent. This avoids
// creating artificial exception transitions across sessions or missing windows.
garcht::CoverageTest coverage_with_boundaries(const std::vector<int>& hits,
                                            const std::vector<bool>& adjacent, double tau);
}
