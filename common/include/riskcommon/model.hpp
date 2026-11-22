// The common model interface of the three engines.
//
// Every VaR model, in every engine, is described by a ModelInfo and forecasts the distribution of
// the P&L of the next target interval as a ModelForecast: P&L = mean + sigma * Z, with Z a
// standardized distribution (Shape) and the VaR / ES it implies at each confidence level. The
// engines keep their models in a list and every consumer -- replay, backtest, output files,
// reports -- iterates over that list: adding a model means writing one class and registering it,
// and nothing downstream names a specific model.
//
// The portfolio engines (ewma, garch_t) implement PortfolioModel; the single-instrument engine
// (realized_caviar) forecasts quantiles of one return series and uses the descriptors only (its
// models are univariate filters with a different life cycle).
#pragma once

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace riskcommon {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

// Losses as positive numbers (money).
struct RiskMeasure {
    double confidence = 0.0;
    double var = kNaN;
    double es = kNaN;
};

// Standardized distribution of the forecast: a unit-variance Student t (Normal when nu is
// infinite), optionally with a generalized Pareto left tail: P(L > u0 + y) = theta (1 + xi y /
// beta)^(-1/xi) for the standardized loss L = -Z beyond the threshold u0.
struct Shape {
    double nu = kInf;
    bool gpd = false;
    double u0 = 0.0, beta = 1.0, xi = 0.0, theta = 0.0;
};

enum class Family { Normal, StudentT, TBodyGpd, Quantile };

struct ModelInfo {
    std::string name;    // label in reports and backtests, e.g. "EWMA-N (ref)"
    std::string prefix;  // column prefix in the output files, e.g. "ref_" (empty for the main model)
    Family family = Family::Normal;
    std::vector<std::string> parameters;  // names of the scalar parameters each forecast reports
    bool main = false;                    // the engine's main model (risk decomposition, robustness)
};

struct ModelForecast {
    double mean = 0.0;   // expected P&L (money)
    double sigma = kNaN;  // P&L scale (money); NaN until the model is ready
    Shape shape;
    std::vector<RiskMeasure> measures;  // one per confidence level of the engine
    std::vector<double> parameters;     // values of ModelInfo::parameters
};

// Rolling history of bar returns kept by an engine, oldest bar first.
struct History {
    const double* returns = nullptr;  // bars x assets, row-major; NaN = missing
    const int* slots = nullptr;       // session slot of each bar
    size_t bars = 0, assets = 0;
    const double* row(size_t t) const { return returns + t * assets; }
};

// Ring buffer of the last `capacity` bars, for engines that hand a History to their models.
class HistoryRing {
public:
    HistoryRing(size_t capacity, size_t assets);
    void push(int slot, const double* returns);
    size_t size() const { return count_; }
    size_t capacity() const { return cap_; }
    History view();  // chronological copy, valid until the next push or view

private:
    size_t cap_, n_, head_ = 0, count_ = 0;
    std::vector<double> rows_, chrono_;
    std::vector<int> slots_, chrono_slots_;
};

// One VaR model of a portfolio engine: sees every bar's log-returns, is re-estimated at the end
// of each session on the rolling window and forecasts the P&L of the next bar for signed money
// exposures. The engine calls every model in registration order; a model may read the state of
// a model registered before it (e.g. a different tail on the same volatility).
class PortfolioModel {
public:
    virtual ~PortfolioModel() = default;
    virtual const ModelInfo& info() const = 0;
    virtual bool ready() const = 0;
    // Log-returns of one bar (NaN = missing); slot = position of the bar in the session.
    virtual void on_bar(int slot, const double* returns) = 0;
    // End-of-session re-estimation on the window; exposures weight portfolio-level estimates.
    virtual void refit(const History& window, const double* exposures) = 0;
    // Forecast for the bar at next_slot.
    virtual void forecast(int next_slot, const double* exposures, ModelForecast& out) const = 0;
    // Realized P&L of the target of a forecast (models that learn from their outcomes).
    virtual void observe_outcome(double /*pnl*/, const ModelForecast& /*forecast*/) {}
    // Positions changed between two refits (models with exposure-dependent estimates).
    virtual void rebalance(const double* /*exposures*/) {}
};

// Output columns of one model: {prefix}sigma, {prefix}nu (t families), {prefix}var_<c>,
// {prefix}es_<c>, {prefix}<parameter>. tag(c) formats a confidence level ("99", "97.5").
std::vector<std::string> forecast_columns(const ModelInfo& m, const std::vector<std::string>& confidence_tags);
std::string format_forecast(const ModelInfo& m, const ModelForecast& f);  // values, comma-led

}  // namespace riskcommon
