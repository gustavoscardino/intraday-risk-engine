// Micro-benchmarks of the GARCH-t engine.
//   benchmark [--quick]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "garcht/garcht.hpp"
#include "garcht/random.hpp"

using namespace garcht;
using clk = std::chrono::steady_clock;

namespace {

volatile double g_sink = 0.0;  // defeats dead-code elimination

std::vector<double> simulate(const GarchParams& p, size_t n, std::uint64_t seed) {
    Rng rng(seed);
    std::vector<double> y(n);
    double h = p.unconditional_variance();
    for (size_t t = 0; t < n + 200; ++t) {
        const double e = std::sqrt(h) * rng.std_t(p.nu);
        if (t >= 200) y[t - 200] = e;
        h = p.omega + p.alpha * e * e + p.beta * h;
    }
    return y;
}

template <class F>
double median_seconds(int reps, F&& f) {
    std::vector<double> v;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = clk::now();
        f();
        v.push_back(std::chrono::duration<double>(clk::now() - t0).count());
    }
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    const bool quick = argc > 1 && std::string(argv[1]) == "--quick";
    const int reps = quick ? 3 : 7;
    const GarchParams truth{0.0, 2e-7, 0.07, 0.91, 5.0};
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    std::printf("GARCH-t engine benchmarks (%u hardware threads)\n\n", hw);

    // ---- log-likelihood kernel
    {
        const size_t n = 100000;
        const std::vector<double> y = simulate(truth, n, 1);
        const double bc = garch_backcast(y.data(), n);
        double g[5];
        const double t_val = median_seconds(reps, [&] { g_sink = garch_loglik(y.data(), n, Dist::StudentT, truth, bc); });
        const double t_grad = median_seconds(reps, [&] { g_sink = garch_loglik(y.data(), n, Dist::StudentT, truth, bc, g); });
        const double n_val = median_seconds(reps, [&] { g_sink = garch_loglik(y.data(), n, Dist::Normal, truth, bc); });
        std::printf("log-likelihood (n = %zu)\n", n);
        std::printf("  Student-t value only      %6.2f ns/obs\n", 1e9 * t_val / n);
        std::printf("  Student-t value+gradient  %6.2f ns/obs\n", 1e9 * t_grad / n);
        std::printf("  Normal value only         %6.2f ns/obs\n\n", 1e9 * n_val / n);
    }

    // ---- estimation
    std::printf("maximum-likelihood fit (BFGS, analytic gradient, std errors included)\n");
    std::printf("  %8s %10s %10s %8s %8s\n", "n", "t (ms)", "Normal(ms)", "iters", "evals");
    for (size_t n : {1000u, 1680u, 5000u, 20000u}) {
        const std::vector<double> y = simulate(truth, n, 2 + n);
        FitResult ft, fn;
        const double tt = median_seconds(reps, [&] { ft = fit_garch(y, GarchSpec{}); });
        const double tn = median_seconds(reps, [&] { fn = fit_garch(y, GarchSpec{Dist::Normal}); });
        std::printf("  %8zu %10.2f %10.2f %8d %8d\n", n, 1e3 * tt, 1e3 * tn, ft.iterations, ft.evaluations);
    }
    {
        const std::vector<double> y = simulate(truth, 1680, 77);
        const FitResult first = fit_garch(y, GarchSpec{});
        FitOptions warm;
        warm.warm_start = &first.params;
        warm.n_starts = 1;
        warm.compute_std_errors = false;
        const double tw = median_seconds(reps, [&] { g_sink = fit_garch(y, GarchSpec{}, warm).loglik; });
        std::printf("  warm-started daily refit, n = 1680, no std errors: %.2f ms\n\n", 1e3 * tw);
    }

    // ---- online filter
    {
        GarchFilter f(truth, truth.unconditional_variance());
        const std::vector<double> y = simulate(truth, 1 << 16, 3);
        const size_t N = quick ? 2000000 : 20000000;
        const auto t0 = clk::now();
        double acc = 0.0;
        for (size_t i = 0; i < N; ++i) acc += f.update(y[i & 0xFFFF]);
        const double s = std::chrono::duration<double>(clk::now() - t0).count();
        g_sink = acc;
        std::printf("online filter update: %.2f ns/update (%.0f M updates/s)\n\n", 1e9 * s / N, N / s / 1e6);
    }

    // ---- t quantile (only needed when nu changes)
    {
        const int N = 20000;
        const auto t0 = clk::now();
        double acc = 0.0;
        for (int i = 0; i < N; ++i) acc += math::t_ppf(0.01, 3.0 + (i % 100) * 0.1);
        g_sink = acc;
        std::printf("Student-t quantile: %.2f us/call\n\n",
                    1e6 * std::chrono::duration<double>(clk::now() - t0).count() / N);
    }

    // ---- portfolio engine
    std::printf("portfolio engine (window 60 days x 28 bars = 1680 bars)\n");
    std::printf("  %6s %14s %14s %16s %12s\n", "assets", "refit 1 thr", "refit all thr", "bar update+VaR", "bars/s");
    for (int n : {15, 50, 200}) {
        EngineConfig cfg;
        cfg.window_bars = 1680;
        std::vector<std::string> tk;
        for (int i = 0; i < n; ++i) tk.push_back("A" + std::to_string(i));
        Rng rng(1000 + n);
        std::vector<std::vector<double>> paths(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) paths[static_cast<size_t>(i)] = simulate(truth, 1680 + 5000, 5000 + i);
        std::vector<double> x(static_cast<size_t>(n));
        for (double& v : x) v = 1e5 * (0.5 + rng.uniform());
        double t1 = 0.0, tall = 0.0;
        for (int threads : {1, 0}) {
            cfg.threads = threads;
            IntradayRiskEngine eng(tk, cfg);
            std::vector<double> row(static_cast<size_t>(n));
            for (size_t t = 0; t < 1680; ++t) {
                for (int i = 0; i < n; ++i) row[static_cast<size_t>(i)] = paths[static_cast<size_t>(i)][t];
                eng.on_bar(static_cast<int>(t % 28), row.data());
            }
            const auto t0 = clk::now();
            eng.refit(x.data());
            (threads == 1 ? t1 : tall) = std::chrono::duration<double>(clk::now() - t0).count();
            if (threads == 0) {
                PortfolioRisk risk;
                const size_t bars = 5000;
                std::vector<double> lat;
                lat.reserve(bars);
                for (size_t t = 1680; t < 1680 + bars; ++t) {
                    for (int i = 0; i < n; ++i) row[static_cast<size_t>(i)] = paths[static_cast<size_t>(i)][t];
                    const auto s0 = clk::now();
                    eng.on_bar(static_cast<int>(t % 28), row.data());
                    eng.compute(static_cast<int>((t + 1) % 28), x.data(), risk);
                    lat.push_back(std::chrono::duration<double>(clk::now() - s0).count());
                    g_sink = risk.garch[2].var;
                }
                std::sort(lat.begin(), lat.end());
                const double med = lat[lat.size() / 2];
                std::printf("  %6d %11.1f ms %11.1f ms %13.2f us %12.0f\n", n, 1e3 * t1, 1e3 * tall, 1e6 * med, 1.0 / med);
            }
        }
    }
    return 0;
}
