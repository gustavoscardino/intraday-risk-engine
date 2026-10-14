#include "ewmar/risk.hpp"

#include <algorithm>
#include <cmath>

#include "ewmar/math.hpp"

namespace ewmar {

TailFactors normal_factors(double c) {
    const double q = math::norm_ppf(c);
    return {q, math::norm_pdf(q) / (1.0 - c)};
}

TailFactors student_factors(double nu, double c) {
    if (!std::isfinite(nu)) return normal_factors(c);
    const double x = math::t_ppf(1.0 - c, nu);  // negative
    const double k = std::sqrt((nu - 2.0) / nu);
    return {-k * x, k * math::t_pdf(x, nu) / (1.0 - c) * (nu + x * x) / (nu - 1.0)};
}

double fit_nu(const std::vector<double>& u_in, double lo, double hi, double* loglik) {
    std::vector<double> u;
    u.reserve(u_in.size());
    for (double v : u_in)
        if (std::isfinite(v)) u.push_back(v);
    auto nll = [&](double nu) {
        double s = 0.0;
        for (double v : u) s -= math::std_t_logpdf(v, nu);
        return s;
    };
    double f = 0.0;
    const double nu = math::brent_minimize(nll, lo, hi, 1e-6, &f);
    if (loglik) *loglik = -f;
    return nu;
}

double bs_price(bool call, double S, double K, double T, double vol, double r) {
    if (!(T > 0.0) || !(vol > 0.0)) return call ? std::max(S - K, 0.0) : std::max(K - S, 0.0);
    const double sd = vol * std::sqrt(T);
    const double d1 = (std::log(S / K) + (r + 0.5 * vol * vol) * T) / sd, d2 = d1 - sd;
    const double df = std::exp(-r * T);
    return call ? S * math::norm_cdf(d1) - K * df * math::norm_cdf(d2)
                : K * df * math::norm_cdf(-d2) - S * math::norm_cdf(-d1);
}

double bs_delta(bool call, double S, double K, double T, double vol, double r) {
    if (!(T > 0.0) || !(vol > 0.0)) return call ? (S > K ? 1.0 : 0.0) : (S < K ? -1.0 : 0.0);
    const double d1 = (std::log(S / K) + (r + 0.5 * vol * vol) * T) / (vol * std::sqrt(T));
    return call ? math::norm_cdf(d1) : math::norm_cdf(d1) - 1.0;
}

double years_to_expiry(int ymd, int minute, int expiry_ymd, const SessionSpec& s) {
    if (ymd > expiry_ymd) return 0.0;
    const double session = static_cast<double>(s.bars_per_day * s.bar_minutes);
    const int close = s.open_minute + s.bars_per_day * s.bar_minutes;
    double days = std::clamp((close - minute) / session, 0.0, 1.0);
    for (int d = next_day(ymd); d <= expiry_ymd; d = next_day(d))
        if (is_business_day(d)) days += 1.0;
    return days / 252.0;
}

}  // namespace ewmar
