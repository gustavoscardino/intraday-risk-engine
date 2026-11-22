// Tests of the code shared by the three engines. Built and run by each engine's ctest.
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "riskcommon/jump_filter.hpp"
#include "riskcommon/model.hpp"
#include "riskcommon/riskmetrics.hpp"

using namespace riskcommon;

namespace {
int g_checks = 0, g_failures = 0;
#define CHECK(cond)                                                          \
    do {                                                                     \
        ++g_checks;                                                          \
        if (!(cond)) {                                                       \
            ++g_failures;                                                    \
            std::printf("    FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                    \
    } while (0)

bool cholesky_ok(const std::vector<double>& A, size_t n) {  // positive semidefinite (up to rounding)
    std::vector<double> L(n * n, 0.0);
    for (size_t j = 0; j < n; ++j) {
        double d = A[j * n + j];
        for (size_t k = 0; k < j; ++k) d -= L[j * n + k] * L[j * n + k];
        if (d < -1e-15) return false;
        L[j * n + j] = std::sqrt(std::max(d, 0.0));
        for (size_t i = j + 1; i < n; ++i) {
            double s = A[i * n + j];
            for (size_t k = 0; k < j; ++k) s -= L[i * n + k] * L[j * n + k];
            L[i * n + j] = L[j * n + j] > 0.0 ? s / L[j * n + j] : 0.0;
        }
    }
    return true;
}

void normal_quantiles() {
    CHECK(std::fabs(normal_quantile(0.99) - 2.3263478740408408) < 1e-14);
    CHECK(std::fabs(normal_quantile(0.975) - 1.959963984540054) < 1e-14);
    CHECK(std::fabs(normal_quantile(0.5)) < 1e-16);
    CHECK(std::fabs(normal_quantile(1e-10) + 6.361340902404056) < 1e-10);
    CHECK(std::isinf(normal_quantile(1.0)) && std::isnan(normal_quantile(1.5)));
}

void riskmetrics_missing_data_and_forecast() {
    const size_t n = 4;
    std::mt19937_64 rng(3);
    std::normal_distribution<double> z(0.0, 0.01);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<double> rows;
    std::vector<int> slots;
    const size_t W = 400;
    for (size_t t = 0; t < W; ++t) {
        const double common = z(rng);
        for (size_t i = 0; i < n; ++i) rows.push_back(u(rng) < 0.15 ? kNaN : 0.6 * common + 0.8 * z(rng));
        slots.push_back(static_cast<int>(t % 28));
    }
    RiskMetricsModel m(n, {0.95, 0.99});
    std::vector<double> x{1e6, -5e5, 2e5, 0.0};
    ModelForecast f;
    m.forecast(1, x.data(), f);
    CHECK(!m.ready() && std::isnan(f.sigma));  // not ready before the first refit
    History h{rows.data(), slots.data(), W, n};
    m.refit(h, x.data());
    CHECK(m.ready() && cholesky_ok(m.covariance(), n));
    for (size_t t = 0; t < 2000; ++t) {  // bars with missing assets keep S a covariance
        std::vector<double> r(n);
        for (size_t i = 0; i < n; ++i) r[i] = u(rng) < 0.3 ? kNaN : z(rng);
        m.on_bar(0, r.data());
    }
    CHECK(cholesky_ok(m.covariance(), n));
    m.forecast(1, x.data(), f);
    double v = 0.0;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) v += x[i] * m.covariance()[i * n + j] * x[j];
    CHECK(std::fabs(f.sigma - std::sqrt(v)) < 1e-9 * f.sigma);
    CHECK(f.measures.size() == 2 && std::fabs(f.measures[1].var - f.sigma * 2.3263478740408408) < 1e-9 * f.sigma);
    CHECK(f.measures[1].es > f.measures[1].var && f.measures[0].confidence == 0.95);
    // the missing-data rule: a missing asset keeps its variance, its covariances shrink by sqrt(lambda)
    std::vector<double> S = m.covariance();
    const std::vector<double> r{0.01, kNaN, -0.02, 0.005};
    ewma_covariance_update(S, n, r.data(), 0.94);
    CHECK(S[1 * n + 1] == m.covariance()[1 * n + 1]);
    CHECK(std::fabs(S[0 * n + 1] - std::sqrt(0.94) * m.covariance()[0 * n + 1]) < 1e-18);
    CHECK(std::fabs(S[0] - (0.94 * m.covariance()[0] + 0.06 * 1e-4)) < 1e-15);
}

void history_ring_is_chronological() {
    HistoryRing ring(3, 2);
    for (int t = 0; t < 5; ++t) {
        const double r[2] = {double(t), double(10 * t)};
        ring.push(t, r);
    }
    const History h = ring.view();
    CHECK(h.bars == 3 && h.slots[0] == 2 && h.slots[2] == 4 && h.row(0)[0] == 2.0 && h.row(2)[1] == 40.0);
}

void columns_follow_the_model_info() {
    ModelInfo m{"EWMA-t", "t_", Family::StudentT, {"lambda"}, false};
    const std::vector<std::string> c = forecast_columns(m, {"95", "99"});
    CHECK(c.size() == 7 && c[0] == "t_sigma" && c[1] == "t_nu" && c[3] == "t_var_99" && c[5] == "t_es_99" &&
          c[6] == "t_lambda");
    ModelForecast f;
    f.sigma = 2.0;
    f.shape.nu = 7.5;
    f.measures = {{0.95, 3.0, 4.0}, {0.99, 5.0, 6.0}};
    f.parameters = {0.97};
    CHECK(format_forecast(m, f) == ",2.0000,7.5000,3.00,5.00,4.00,6.00,0.97");
}

void jump_filter_decisions_are_causal() {
    // 4 stocks, 28 bars a day, a random walk; day 5: a bad tick on A (reverts), a lasting shift on B,
    // the four together (market move)
    const size_t n = 4, K = 28, days = 6;
    std::mt19937_64 rng(11);
    std::normal_distribution<double> z(0.0, 0.002);
    std::vector<double> open, close;
    std::vector<double> px(n, 50.0);
    for (size_t d = 0; d < days; ++d)
        for (size_t k = 0; k < K; ++k)
            for (size_t i = 0; i < n; ++i) {
                open.push_back(px[i]);
                px[i] *= std::exp(z(rng));
                if (d == 5 && k == 10 && i == 1) px[i] *= 1.06;  // genuine, lasting
                if (d == 5 && k == 20) px[i] *= 0.94;            // the market falls
                close.push_back(px[i]);
                if (d == 5 && k == 5 && i == 0) close.back() *= 1.07;  // wrong print, next bar back
            }
    auto run = [&](std::vector<double> c, JumpFilter& f) {
        for (size_t t = 0; t < days * K; ++t)
            f.apply(t, std::to_string(t), static_cast<int>(t / K), static_cast<int>(t % K), &open[t * n], &c[t * n]);
        return c;
    };
    JumpFilter f({"A", "B", "C", "D"}, static_cast<int>(K));
    const std::vector<double> out = run(close, f);
    const size_t day5 = 5 * K;
    CHECK(std::isnan(out[(day5 + 5) * n + 0]) && out[(day5 + 6) * n + 0] == close[(day5 + 6) * n + 0]);
    CHECK(std::isnan(out[(day5 + 10) * n + 1]) && out[(day5 + 11) * n + 1] == close[(day5 + 11) * n + 1]);
    for (size_t i = 0; i < n; ++i) CHECK(out[(day5 + 20) * n + i] == close[(day5 + 20) * n + i]);
    const JumpFilter::Counts c = f.counts();
    CHECK(c.held == 2 && c.bad_ticks == 1 && c.accepted == 1 && c.market_moves == 4);
    CHECK(f.accepted().size() == 1 && f.accepted()[0].bar == day5 + 11 && f.accepted()[0].asset == 1);
    CHECK(std::fabs(f.accepted()[0].ret - std::log(close[(day5 + 11) * n + 1] / close[(day5 + 9) * n + 1])) < 1e-15);
    // a different future leaves every earlier decision as it was
    std::vector<double> other = close;
    for (size_t k = (day5 + 8) * n; k < other.size(); ++k) other[k] *= 1.2;
    JumpFilter f2({"A", "B", "C", "D"}, static_cast<int>(K));
    const std::vector<double> out2 = run(other, f2);
    bool same = true;
    for (size_t k = 0; k < (day5 + 8) * n; ++k) same = same && ((std::isnan(out[k]) && std::isnan(out2[k])) || out[k] == out2[k]);
    CHECK(same);
}
}  // namespace

int main() {
    normal_quantiles();
    riskmetrics_missing_data_and_forecast();
    history_ring_is_chronological();
    columns_follow_the_model_info();
    jump_filter_decisions_are_causal();
    std::printf("riskcommon: %d checks, %d failed\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
