// Speed of the EWMA engine on simulated returns, for portfolios of 15, 50 and 200 assets:
// per-bar update + one-bar-ahead VaR/ES of the three models, and the end-of-day refit.
//
//   benchmark [--days N] > results/benchmark.txt
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "cli.hpp"
#include "ewmar/ewmar.hpp"
#include "paths.hpp"

using namespace ewmar;

namespace {
double pct(std::vector<double> v, double q) {
    if (v.empty()) return std::nan("");
    std::sort(v.begin(), v.end());
    return v[static_cast<size_t>(q * static_cast<double>(v.size() - 1))];
}
}  // namespace

int main(int argc, char** argv) {
    const paths::Utf8Console console;
    int days = 120;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--days" && i + 1 < argc) {
            if (!cli::int_arg(a, argv[++i], days, 70, 100000)) return 2;
        } else {
            std::fprintf(stderr, "usage: benchmark [--days N]\n");
            return 2;
        }
    }
    const int K = 28, window_days = 40;  // the estimation window of every engine
    std::printf("EWMA engine benchmark: %d sessions of %d bars, window %d sessions, 3 models (ref, N, t)\n", days, K,
                window_days);
    std::printf("%8s %14s %14s %14s %14s %14s\n", "assets", "bar p50 (us)", "bar p95 (us)", "bar p99 (us)",
                "refit p50 (ms)", "refit max (ms)");
    for (const size_t n : {size_t(15), size_t(50), size_t(200)}) {
        EngineConfig cfg;
        cfg.bars_per_day = K;
        cfg.window_bars = window_days * K;
        EwmaEngine eng(n, cfg);
        math::Rng rng(42 + n);
        std::vector<double> r(n), x(n), g(n);
        for (size_t i = 0; i < n; ++i) x[i] = 1e5 * (1.0 + static_cast<double>(i % 7));
        std::vector<double> bar_us, refit_ms;
        PortfolioRisk risk;
        double vol = 1.0;
        for (int d = 0; d < days; ++d) {
            for (size_t i = 0; i < n; ++i) g[i] = 0.004 * rng.student_t(5.0);
            eng.on_overnight(g.data());
            for (int k = 0; k < K; ++k) {
                vol = std::sqrt(0.02 + 0.97 * vol * vol + 0.01 * rng.normal() * rng.normal());
                const double u = 1.0 + 0.8 * std::pow((k - (K - 1) / 2.0) / (K / 2.0), 2);
                const double m = rng.normal();
                for (size_t i = 0; i < n; ++i) r[i] = 0.0015 * u * vol * (0.6 * m + 0.8 * rng.student_t(6.0) / std::sqrt(1.5));
                const auto t0 = std::chrono::steady_clock::now();
                eng.on_bar(k, r.data());
                if (k == K - 1 && eng.history_size() >= static_cast<size_t>(cfg.window_bars)) {
                    const RefitReport rep = eng.refit(x.data());
                    refit_ms.push_back(rep.seconds * 1e3);
                }
                eng.compute((k + 1) % K, x.data(), risk);
                const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
                if (eng.ready() && k != K - 1) bar_us.push_back(us);
            }
        }
        std::printf("%8zu %14.2f %14.2f %14.2f %14.2f %14.2f\n", n, pct(bar_us, 0.5), pct(bar_us, 0.95), pct(bar_us, 0.99),
                    pct(refit_ms, 0.5), pct(refit_ms, 1.0));
    }
    return 0;
}
