// VaR / ES of a zero-mean P&L X = sigma * Z, with Z Normal or a unit-variance Student t:
//     VaR_c = sigma * q_c,   ES_c = sigma * e_c   (losses as positive numbers)
//   Normal:  q = Phi^-1(c),  e = phi(q) / (1 - c)
//   t(nu):   x = t_nu^-1(1 - c), k = sqrt((nu - 2) / nu),
//            q = -k x,         e = k f_nu(x) / (1 - c) * (nu + x^2) / (nu - 1)
// plus the maximum likelihood estimate of nu and the Black-Scholes delta of the options.
#pragma once

#include <vector>

#include "ewmar/market_data.hpp"
#include "riskcommon/model.hpp"

namespace ewmar {

struct TailFactors {
    double var = 0.0, es = 0.0;  // per unit of sigma
};
TailFactors normal_factors(double confidence);
TailFactors student_factors(double nu, double confidence);  // nu = inf gives the Normal factors

using RiskMeasure = riskcommon::RiskMeasure;  // confidence, var, es (losses positive)

// Maximum likelihood nu of a unit-variance t fitted to u (scale fixed at 1), by Brent on [lo, hi].
// Non-finite values are skipped. *loglik receives the maximized log-likelihood when given.
double fit_nu(const std::vector<double>& u, double lo = 2.5, double hi = 60.0, double* loglik = nullptr);

// ---------------------------------------------------------------- options

double bs_price(bool call, double spot, double strike, double years, double vol, double rate);
double bs_delta(bool call, double spot, double strike, double years, double vol, double rate);

// Time to expiry in trading years (252 sessions): the rest of today's session plus the B3
// business days up to the expiry date (inclusive; the option expires at that day's close).
double years_to_expiry(int ymd, int minute, int expiry_ymd, const SessionSpec& s);

}  // namespace ewmar
