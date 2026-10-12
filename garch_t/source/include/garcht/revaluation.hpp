// Full revaluation risk by Monte Carlo, for portfolios with non-linear instruments (options).
//
// The engine's one-bar-ahead forecast defines a joint return distribution
//   r_i = mu_i + sigma_i * Z_i,   Z ~ multivariate Student-t(nu) with correlation R
// (unit-variance marginals), nu = the t body of the portfolio tail (the GPD tail of the EVT
// model is a portfolio-level object a joint simulation does not reproduce). Every linear
// combination of Z is a unit-variance t(nu), so for a stock-only portfolio the simulated
// VaR/ES converge to the analytic t(nu) values (the replay's "book delta-linear" uses the same
// t(nu)); with options, each scenario reprices the positions (Black-Scholes-Merton), capturing
// gamma.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "garcht/risk.hpp"

namespace garcht {

enum class OptionType { Call, Put };

// Black-Scholes-Merton, European, continuous dividend yield q. t <= 0 or vol <= 0 give the
// (discounted forward) intrinsic value; NaN inputs give NaN.
double bs_price(OptionType type, double spot, double strike, double t_years, double vol, double rate,
                double q = 0.0);
double bs_delta(OptionType type, double spot, double strike, double t_years, double vol, double rate,
                double q = 0.0);
double bs_gamma(double spot, double strike, double t_years, double vol, double rate, double q = 0.0);

struct Position {
    enum class Kind { Stock, Option };
    Kind kind = Kind::Stock;
    std::string id;
    int asset = 0;          // index of the stock / underlying in the engine
    double quantity = 0.0;  // shares, or option contracts (signed)
    // options only
    OptionType type = OptionType::Call;
    double strike = 0.0;
    double expiry = 0.0;     // years to expiry at valuation time
    double vol = 0.0;        // implied volatility used for marking (annual)
    double rate = 0.0;       // continuously compounded risk-free rate
    double multiplier = 1.0; // shares per contract
};

double position_value(const Position& p, double spot, double time_shift_years = 0.0);
// Delta-equivalent stock exposure in money (what a linear model sees).
double position_delta_exposure(const Position& p, double spot);

struct McRiskResult {
    std::vector<RiskMeasure> measures;  // per confidence; money, positive = loss
    double mean = 0.0, sigma = 0.0;     // of the simulated P&L
    int scenarios = 0;
};

class MonteCarloRevaluation {
public:
    explicit MonteCarloRevaluation(int n_scenarios = 20000, std::uint64_t seed = 7);

    // Draws antithetic scenarios of the joint one-bar-ahead returns (M x n, row-major).
    void simulate(const JointForecast& f);

    // P&L of the positions over `horizon_years` (time decay included) in every scenario.
    McRiskResult evaluate(const std::vector<Position>& positions, const std::vector<double>& spot,
                          double horizon_years, const std::vector<double>& confidences) const;

    const std::vector<double>& scenarios() const { return ret_; }
    int num_scenarios() const { return m_; }
    int num_assets() const { return n_; }

private:
    int m_;
    int n_ = 0;
    std::uint64_t seed_;
    std::uint64_t draws_ = 0;  // advances the stream between calls
    std::vector<double> ret_;
};

// Lower-triangular L with L L' = A for a positive *semi*-definite A (row-major n x n);
// directions with no variance get a zero column. Returns false if A is clearly not PSD.
bool cholesky_psd(const std::vector<double>& A, int n, std::vector<double>& L);

}  // namespace garcht
