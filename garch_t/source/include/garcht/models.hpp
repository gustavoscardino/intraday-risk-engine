// The VaR models of the garch_t engine behind the common model interface
// (common/include/riskcommon/model.hpp):
//
//   GARCH-t/EVT   GarchPortfolioModel: the multivariate GARCH(1,1)-t engine of portfolio.hpp
//                 (main model: risk decomposition, option book, overnight forecast)
//   EWMA-N (ref)  riskcommon::RiskMetricsModel: the RiskMetrics baseline shared with ewma
//
// The replay keeps them in a list (make_models) and every consumer -- the replay, the backtest,
// the output files -- iterates over it. A new model is a class implementing
// riskcommon::PortfolioModel, added to make_models.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "garcht/portfolio.hpp"
#include "riskcommon/model.hpp"
#include "riskcommon/riskmetrics.hpp"

namespace garcht {

class GarchPortfolioModel : public riskcommon::PortfolioModel {
public:
    GarchPortfolioModel(const std::vector<std::string>& tickers, const EngineConfig& cfg);

    const riskcommon::ModelInfo& info() const override { return info_; }
    bool ready() const override { return engine_.ready(); }
    void on_bar(int slot, const double* returns) override { engine_.on_bar(slot, returns); }
    // The engine keeps its own rolling window (it also holds the standardized residuals).
    void refit(const riskcommon::History& window, const double* exposures) override;
    void forecast(int next_slot, const double* exposures, riskcommon::ModelForecast& out) const override;
    void observe_outcome(double pnl, const riskcommon::ModelForecast& f) override;
    void rebalance(const double* exposures) override { engine_.recalibrate_tail(exposures); }

    IntradayRiskEngine& engine() { return engine_; }
    const IntradayRiskEngine& engine() const { return engine_; }
    const RefitReport& last_refit() const { return last_refit_; }
    // The engine's full output of the last forecast (per-asset sigmas, Euler contributions).
    const PortfolioRisk& last_risk() const { return risk_; }

private:
    IntradayRiskEngine engine_;
    riskcommon::ModelInfo info_;
    RefitReport last_refit_;
    mutable PortfolioRisk risk_;
};

// Fills a ModelForecast from the engine's PortfolioRisk.
void to_model_forecast(const PortfolioRisk& r, riskcommon::ModelForecast& out);

// The engine's models, the main one first.
std::vector<std::unique_ptr<riskcommon::PortfolioModel>> make_models(const std::vector<std::string>& tickers,
                                                                     const EngineConfig& cfg);

}  // namespace garcht
