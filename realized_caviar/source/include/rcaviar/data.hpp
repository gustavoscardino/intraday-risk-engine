#pragma once
#include "rcaviar/model.hpp"
#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <optional>
namespace rcaviar {
struct Price { std::int64_t timestamp; double price; };
// observation: what the models see (features, and the target they are fitted on). scored_return:
// the target as printed, when it differs (shared data with the jump filter); the forecasts are
// scored on it, as the other engines score on the prices as printed. emit=false: a catch-up window
// (the bar of a held print that the next bar confirmed): the recursion takes its update, at the
// next bar, before that bar's own, and no forecast is issued at it.
struct Window { std::int64_t forecast_time, target_time; Observation observation; std::optional<double> scored_return{}; bool emit=true; };
// Strict timestamp,price CSV. Integer seconds; contiguous 5-second samples.
std::vector<Price> read_prices(const std::string& path);
std::vector<Price> synthetic_prices(std::size_t samples = 120000, unsigned seed = 42);
// Bounded, causal feature builder. No future price is needed to emit a forecast.
// begin_session clears only the feature history; the caller owns quantile state.
class StreamingWindows {
public:
    std::optional<Window> push(Price price);
    void begin_session();
private:
    std::array<double,60> returns_{};
    std::optional<Price> previous_;
    std::size_t count_=0, cursor_=0;
    double aggregate_=0, rv_=0;
};
// Includes pending forecasts (NaN target). Gaps reject by default; explicit reset
// starts a new feature window and prevents targets from crossing the gap.
std::vector<Window> make_windows(const std::vector<Price>& prices, bool reset_on_gap=false);
}
