#include "garcht/models.hpp"

namespace garcht {

namespace {
riskcommon::ModelInfo garch_info(const EngineConfig& cfg) {
    riskcommon::ModelInfo m;
    const bool t = cfg.spec.dist == Dist::StudentT;
    m.name = !t ? "GARCH-N" : (cfg.tail_model == TailModel::Evt ? "GARCH-t/EVT" : "GARCH-t");
    m.prefix = "";  // main model: sigma, nu, var_<c>, es_<c>
    m.family = !t ? riskcommon::Family::Normal
                  : (cfg.tail_model == TailModel::Evt ? riskcommon::Family::TBodyGpd : riskcommon::Family::StudentT);
    m.main = true;
    return m;
}
}  // namespace

GarchPortfolioModel::GarchPortfolioModel(const std::vector<std::string>& tickers, const EngineConfig& cfg)
    : engine_(tickers, cfg), info_(garch_info(cfg)) {}

void GarchPortfolioModel::refit(const riskcommon::History&, const double* exposures) {
    last_refit_ = engine_.refit(exposures);
}

void to_model_forecast(const PortfolioRisk& r, riskcommon::ModelForecast& out) {
    out.mean = r.mean;
    out.sigma = r.sigma;
    out.shape = r.tail;
    out.measures = r.measures;
    out.parameters.clear();
}

void GarchPortfolioModel::forecast(int next_slot, const double* exposures, riskcommon::ModelForecast& out) const {
    engine_.compute(next_slot, exposures, risk_);
    to_model_forecast(risk_, out);
}

void GarchPortfolioModel::observe_outcome(double pnl, const riskcommon::ModelForecast& f) {
    PortfolioRisk r;
    r.mean = f.mean;
    r.sigma = f.sigma;
    engine_.observe_outcome(pnl, r);
}

std::vector<std::unique_ptr<riskcommon::PortfolioModel>> make_models(const std::vector<std::string>& tickers,
                                                                     const EngineConfig& cfg) {
    std::vector<std::unique_ptr<riskcommon::PortfolioModel>> m;
    m.push_back(std::make_unique<GarchPortfolioModel>(tickers, cfg));
    m.push_back(std::make_unique<riskcommon::RiskMetricsModel>(tickers.size(), cfg.confidences, cfg.ewma_lambda));
    return m;
}

}  // namespace garcht
