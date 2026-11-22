// The VaR models of the ewma engine behind the common model interface
// (common/include/riskcommon/model.hpp), in registration order:
//
//   EWMA-N (ref)  riskcommon::RiskMetricsModel: lambda = 0.94 fixed, raw returns, Normal (the
//                 baseline shared with garch_t)
//   EWMA-N        EwmaNormalModel: lambda estimated every day, deseasonalized returns y = r / s_k,
//                 Normal
//   EWMA-t        EwmaStudentModel: the sigma of EWMA-N, unit-variance Student t with nu estimated
//                 every day (main model: risk decomposition, option book, robustness report)
//
// Forecast for exposures x (money): a_i = x_i s_{k(t+1), i}, sigma = sqrt(a' S a). A new model is a
// class implementing riskcommon::PortfolioModel, added to make_models.
#pragma once

#include <memory>
#include <vector>

#include "ewmar/ewma.hpp"
#include "ewmar/risk.hpp"
#include "ewmar/seasonality.hpp"
#include "riskcommon/model.hpp"

namespace ewmar {

struct ModelConfig {
    int bars_per_day = 28;
    double ref_lambda = 0.94;  // EWMA-N (ref)
    bool estimate_lambda = true;
    double lambda = 0.94;  // used when estimate_lambda is false (and as the first value)
    double lambda_lo = 0.80, lambda_hi = 0.999;
    bool estimate_nu = true;
    double nu = 8.0;  // used when estimate_nu is false (and while there is too little data)
    double nu_lo = 2.5, nu_hi = 60.0;
    DiurnalMode diurnal = DiurnalMode::Pooled;
    std::vector<double> confidences{0.95, 0.975, 0.99};
};

class EwmaNormalModel : public riskcommon::PortfolioModel {
public:
    EwmaNormalModel(size_t assets, const ModelConfig& cfg);

    const riskcommon::ModelInfo& info() const override { return info_; }
    bool ready() const override { return ready_; }
    void on_bar(int slot, const double* returns) override;
    // Diurnal profile, lambda by predictive quasi-likelihood and the filter over the window (its
    // first quarter starts the covariance and is not scored).
    void refit(const riskcommon::History& window, const double* exposures) override;
    void forecast(int next_slot, const double* exposures, riskcommon::ModelForecast& out) const override;

    // sigma of the next bar's P&L at next_slot; with `contrib`, also each position's stand-alone
    // sigma, its Euler contribution (sums to sigma) and each stock's bar volatility.
    double sigma(int next_slot, const double* exposures, std::vector<double>* standalone = nullptr,
                 std::vector<double>* contrib = nullptr, std::vector<double>* asset_sigma = nullptr) const;
    // Standardized one-step-ahead portfolio outcomes of the last refit window (for a tail model).
    const std::vector<double>& standardized_outcomes() const { return u_; }
    double lambda() const { return lambda_; }
    const DiurnalProfile& diurnal() const { return diurnal_; }
    const EwmaCovariance& covariance() const { return Sy_; }

private:
    size_t n_;
    ModelConfig cfg_;
    riskcommon::ModelInfo info_;
    bool ready_ = false;
    double lambda_;
    DiurnalProfile diurnal_;
    EwmaCovariance Sy_;
    std::vector<TailFactors> fac_;
    std::vector<double> u_;
    mutable std::vector<double> a_, Sa_;  // scratch
};

class EwmaStudentModel : public riskcommon::PortfolioModel {
public:
    // The volatility comes from `base`, which must be registered (and refitted) before this one.
    EwmaStudentModel(const EwmaNormalModel& base, const ModelConfig& cfg);

    const riskcommon::ModelInfo& info() const override { return info_; }
    bool ready() const override { return base_.ready(); }
    void on_bar(int, const double*) override {}
    // nu by maximum likelihood on the base model's standardized outcomes (at least 100).
    void refit(const riskcommon::History& window, const double* exposures) override;
    void forecast(int next_slot, const double* exposures, riskcommon::ModelForecast& out) const override;
    double nu() const { return nu_; }

private:
    const EwmaNormalModel& base_;
    ModelConfig cfg_;
    riskcommon::ModelInfo info_;
    double nu_;
    std::vector<TailFactors> fac_;
};

// The engine's models, in registration order (see the top of this file).
std::vector<std::unique_ptr<riskcommon::PortfolioModel>> make_models(size_t assets, const ModelConfig& cfg);

}  // namespace ewmar
