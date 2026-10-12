// Synthetic intraday market: 15 fictitious B3-style stocks, 15-minute bars, a client
// portfolio with fills and a small book of listed options.
//
// Data-generating process (per bar k of day d):
//   r_i = s_{i,k} * (b_i * f + c_i * g_sector(i) + e_i)
// f (market), g (5 sectors) and e (15 idiosyncratic terms) follow their own GARCH-type
// processes with standardized Student-t shocks. With `realistic` (default) the process is
// deliberately *not* the model being tested:
//   * leverage effect (GJR: negative shocks raise volatility more),
//   * idiosyncratic volatility regimes (two-state Markov chain),
//   * opening / closing auction jumps,
//   * asset-specific intraday volatility patterns,
//   * low-priced stocks where the R$ 0.01 tick produces many zero returns,
//   * a market crash with a circuit breaker (trading halted for 30 minutes),
// plus an elevated-volatility regime, an earnings jump and overnight gaps.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "garcht/market_data.hpp"
#include "garcht/revaluation.hpp"

namespace garcht {

struct SyntheticConfig {
    int days = 150;
    std::uint64_t seed = 20260302ULL;
    bool realistic = true;
    bool degraded_copy = true;  // also produce a feed with typical data problems
};

struct AssetInfo {
    std::string ticker, name, sector;
    double market_beta = 0.0;
};

struct Fill {
    std::string id;
    TimePoint time;
    std::string ticker;
    double quantity = 0.0;  // signed: buy > 0
    double price = 0.0;
};

struct OptionContract {
    std::string id, underlying;
    OptionType type = OptionType::Call;
    double strike = 0.0;
    int expiry_ymd = 0;  // expires at the close of this date
    double implied_vol = 0.0;
    double rate = 0.0;
    double multiplier = 100.0;
    double quantity = 0.0;  // contracts, signed
};

struct ScenarioEvent {
    int day = 0;
    std::string timestamp, text;
};

struct SyntheticMarket {
    SessionSpec session;
    std::vector<AssetInfo> assets;
    std::vector<double> initial_quantity;  // per asset
    std::vector<Fill> fills;
    std::vector<OptionContract> options;
    std::vector<BarRow> rows;           // clean feed
    std::vector<BarRow> degraded_rows;  // same market with feed problems (optional)
    std::vector<ScenarioEvent> events;
    std::vector<std::string> truth_lines;  // CSV lines (with header) of the true parameters

    std::vector<std::string> tickers() const;
};

SyntheticMarket generate_market(const SyntheticConfig& cfg);

// Writes assets.csv, portfolio_initial.csv, fills.csv, options.csv, prices_15m.csv,
// prices_15m_degraded.csv, scenario.csv and simulation_truth.csv. Creates the directory.
void write_market(const SyntheticMarket& m, const std::string& dir);

// CSV readers for the portfolio files (shared by the apps).
std::vector<Fill> load_fills_csv(const std::string& path, int exchange_utc_offset_min = -180);
std::vector<OptionContract> load_options_csv(const std::string& path);

}  // namespace garcht
