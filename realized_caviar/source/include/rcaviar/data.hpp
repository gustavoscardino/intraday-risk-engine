#pragma once
#include "rcaviar/model.hpp"
#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <optional>
namespace rcaviar {
struct Price { std::int64_t timestamp; double price; };
struct Window { std::int64_t forecast_time, target_time; Observation observation; };
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
