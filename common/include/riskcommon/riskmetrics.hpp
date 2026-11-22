// RiskMetrics baseline shared by the portfolio engines (the method described in the project text):
//
//     S(t) = lambda S(t-1) + (1 - lambda) r(t) r(t)',   lambda = 0.94,   raw log-returns
//     P&L ~ Normal(0, x' S x),   VaR_c = sigma q_c,  ES_c = sigma phi(q_c) / (1 - c)
//
// One implementation in both engines, so that the baseline -- the yardstick of the comparison --
// differs between them only if their inputs do (atam/, scenario Cp1).
//
// Missing returns: entries between two observed assets are updated as above; the variance of a
// missing asset, and its covariances with other missing assets, keep their values; its covariances
// with observed assets are multiplied by sqrt(lambda). That is S_new = D S_old D + (1 - lambda) y_o
// y_o', D = diag(sqrt(lambda) on observed assets, 1 on missing ones): a sum of two positive
// semidefinite matrices, so S stays a covariance (keeping the old cross terms would not).
//
// Life cycle: not ready until the first refit, which starts S from the mean outer product of the
// first quarter of the window and runs the recursion over the rest; after that every bar updates S.
#pragma once

#include <vector>

#include "riskcommon/model.hpp"

namespace riskcommon {

double normal_quantile(double p);  // Phi^-1(p), p in (0, 1)
double normal_pdf(double x);

// Covariance update with the missing-return rule above (y: n values, NaN = missing).
void ewma_covariance_update(std::vector<double>& S, size_t n, const double* y, double lambda);
// Mean of y y' over the complete rows [from, to) of `rows`; a diagonal of each asset's mean
// square when there are fewer than n + 5 complete rows. Variances floored at floor_var.
std::vector<double> initial_covariance(const double* rows, size_t n, size_t from, size_t to, double floor_var = 1e-14);

class RiskMetricsModel : public PortfolioModel {
public:
    RiskMetricsModel(size_t assets, std::vector<double> confidences, double lambda = 0.94,
                     ModelInfo info = {"EWMA-N (ref)", "ref_", Family::Normal, {}, false});

    const ModelInfo& info() const override { return info_; }
    bool ready() const override { return ready_; }
    void on_bar(int slot, const double* returns) override;
    void refit(const History& window, const double* exposures) override;
    void forecast(int next_slot, const double* exposures, ModelForecast& out) const override;

    const std::vector<double>& covariance() const { return S_; }
    double lambda() const { return lambda_; }

private:
    size_t n_;
    std::vector<double> conf_, q_, e_;  // Normal VaR and ES factors per confidence
    double lambda_;
    ModelInfo info_;
    bool ready_ = false;
    std::vector<double> S_;
};

}  // namespace riskcommon
