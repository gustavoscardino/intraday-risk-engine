// Monte Carlo study: many independent synthetic markets, each replayed through the engine
// and backtested. Aggregating gives tight confidence intervals for the exception rates and
// the rejection frequencies of each test (about 5% at a 5% level when a model is right).
//
//   monte_carlo [--seeds N] [--days D] [--window W] [--threads T] [--simple]
//               [--corr ewma|dcc|constant] [--tail oos|insample] [--tail-model evt|t]
//               [--es-sims S] [--out DIR]
//
// Results (summary CSV + report.txt) go to garch_t/results/monte_carlo_realistic, or
// monte_carlo_simple with --simple, unless --out is given.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "cli.hpp"
#include "garcht/garcht.hpp"
#include "paths.hpp"

using namespace garcht;

namespace {

struct SeedResult {
    std::uint64_t seed = 0;
    bool ok = false;
    std::string error;
    BacktestReport report;
    double mean_nu = 0.0, mean_dcc_a = 0.0, mean_dcc_b = 0.0;
    double seconds = 0.0;
};

void usage() {
    std::fprintf(stderr,
                 "usage: monte_carlo [--seeds N] [--days D] [--window W] [--threads T] [--simple]\n"
                 "                   [--corr ewma|dcc|constant] [--tail oos|insample] [--tail-model evt|t]\n"
                 "                   [--es-sims S] [--out DIR]\n");
}

std::FILE* g_report = nullptr;  // report.txt: copy of the console report

// printf to the console and to report.txt
void say(const char* fmt, ...) {
    va_list a, b;
    va_start(a, fmt);
    va_copy(b, a);
    std::vprintf(fmt, a);
    if (g_report) std::vfprintf(g_report, fmt, b);
    va_end(b);
    va_end(a);
}

}  // namespace

int main(int argc, char** argv) {
    const paths::Utf8Console console;
    int seeds = 60, days = 150, threads = 0, es_sims = 400;
    std::string out;  // default: garch_t/results/monte_carlo_realistic (or _simple)
    SyntheticConfig scfg;
    scfg.degraded_copy = false;
    ReplayConfig rcfg;
    rcfg.engine.threads = 1;  // parallelism is across markets
    rcfg.record_params = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool has = i + 1 < argc;
        int k = 0;
        bool ok = true;
        if (a == "--simple") scfg.realistic = false;
        else if (!has) ok = false;
        else if (a == "--seeds") ok = cli::to_int(a, argv[++i], seeds, 1, 100000);
        else if (a == "--days") ok = cli::to_int(a, argv[++i], days, 10, 100000);
        else if (a == "--window") ok = cli::to_int(a, argv[++i], rcfg.window_days, 5, 10000);
        else if (a == "--threads") ok = cli::to_int(a, argv[++i], threads, 0, 1024);
        else if (a == "--es-sims") ok = cli::to_int(a, argv[++i], es_sims, 0, 10000000);
        else if (a == "--out") out = argv[++i];
        else if (a == "--corr") {
            ok = (k = cli::choice(a, argv[++i], {"ewma", "dcc", "constant"})) >= 0;
            rcfg.engine.correlation = k == 0 ? CorrelationMode::Ewma : k == 1 ? CorrelationMode::Dcc : CorrelationMode::Constant;
        } else if (a == "--tail-model") {
            ok = (k = cli::choice(a, argv[++i], {"evt", "t"})) >= 0;
            rcfg.engine.tail_model = k == 1 ? TailModel::StudentT : TailModel::Evt;
        } else if (a == "--tail") {
            ok = (k = cli::choice(a, argv[++i], {"oos", "insample"})) >= 0;
            rcfg.engine.tail = k == 1 ? TailCalibration::InSample : TailCalibration::OutOfSample;
        } else {
            ok = false;
        }
        if (!ok) {
            usage();
            return 2;
        }
    }
    if (days <= rcfg.window_days + 5) {
        std::fprintf(stderr, "error: --days must exceed --window by more than 5 (out-of-sample days)\n");
        usage();
        return 2;
    }
    if (out.empty()) out = paths::results_dir(scfg.realistic ? "monte_carlo_realistic" : "monte_carlo_simple");
    std::filesystem::create_directories(out);
    g_report = std::fopen((out + "/report.txt").c_str(), "wb");
    scfg.days = days;
    if (threads <= 0) threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    const std::vector<double>& confs = rcfg.engine.confidences;

    say("Monte Carlo study: %d markets x %d days (%s profile), window %d days, %d threads\n", seeds, days,
                scfg.realistic ? "realistic" : "simple", rcfg.window_days, threads);
    std::vector<SeedResult> results(static_cast<size_t>(seeds));
    std::atomic<int> next{0}, done{0};
    std::mutex print_mtx;
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> pool;
    for (int th = 0; th < std::min(threads, seeds); ++th)
        pool.emplace_back([&] {
            for (int k; (k = next.fetch_add(1)) < seeds;) {
                SeedResult& r = results[static_cast<size_t>(k)];
                r.seed = 1000003ULL * static_cast<std::uint64_t>(k + 1);
                const auto s0 = std::chrono::steady_clock::now();
                try {
                    SyntheticConfig c = scfg;
                    c.seed = r.seed;
                    const SyntheticMarket m = generate_market(c);
                    ReplayInput in;
                    in.tickers = m.tickers();
                    in.bars = build_panel(m.rows, in.tickers);
                    in.initial_quantity = m.initial_quantity;
                    in.fills = m.fills;
                    const ReplayResult rr = run_replay(in, rcfg);
                    r.report = evaluate_backtest(rr, confs, es_sims);
                    // overnight coverage without the scripted crash session (a -9 sd market shock
                    // that no 99% VaR should cover; with ~90 nights it alone adds ~1.1 points)
                    std::string crash_date;
                    for (const ScenarioEvent& e : m.events)
                        if (e.text.rfind("market crash", 0) == 0) crash_date = e.timestamp.substr(0, 10);
                    for (size_t ci = 0; ci < confs.size(); ++ci) {
                        std::vector<int> hits;
                        for (const OvernightRecord& o : rr.overnight)
                            if (std::isfinite(o.pnl) && ci < o.risk.measures.size() &&
                                o.target_time.substr(0, 10) != crash_date)
                                hits.push_back(-o.pnl > o.risk.measures[ci].var);
                        ModelBacktest mb;
                        mb.model = "overnight ex-crash";
                        mb.confidence = confs[ci];
                        mb.coverage = coverage_test(hits, 1.0 - confs[ci]);
                        r.report.rows.push_back(mb);
                    }
                    for (const DailyRecord& d : rr.daily) {
                        r.mean_nu += d.nu_p / static_cast<double>(rr.daily.size());
                        r.mean_dcc_a += d.dcc_a / static_cast<double>(rr.daily.size());
                        r.mean_dcc_b += d.dcc_b / static_cast<double>(rr.daily.size());
                    }
                    r.ok = true;
                } catch (const std::exception& e) {
                    r.error = e.what();
                }
                r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count();
                const int d = ++done;
                std::lock_guard<std::mutex> lk(print_mtx);
                std::printf("\r  %d/%d markets done", d, seeds);
                std::fflush(stdout);
            }
        });
    for (auto& t : pool) t.join();
    const double total_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\r%40s\r", "");
    say("  %d markets in %.1f s\n", seeds, total_s);

    // ---- aggregate
    struct Agg {
        long n = 0, exc = 0;
        int runs = 0, rej_uc = 0, rej_cc = 0, rej_es = 0, es_runs = 0;
        double es_ratio_sum = 0.0;
        int es_ratio_n = 0;
        long seg_n[3] = {0, 0, 0}, seg_exc[3] = {0, 0, 0};
    };
    std::map<std::pair<std::string, double>, Agg> agg;
    std::vector<std::string> order;
    int ok_runs = 0, berk_o = 0, berk_o_runs = 0;
    std::vector<std::pair<std::string, int>> berk;  // Berkowitz rejections per model, in model order
    double nu_sum = 0.0, a_sum = 0.0, b_sum = 0.0;
    for (const SeedResult& r : results) {
        if (!r.ok) {
            say("  seed %llu failed: %s\n", static_cast<unsigned long long>(r.seed), r.error.c_str());
            continue;
        }
        ++ok_runs;
        nu_sum += r.mean_nu;
        a_sum += r.mean_dcc_a;
        b_sum += r.mean_dcc_b;
        for (size_t k = 0; k < r.report.berkowitz.size(); ++k) {
            if (berk.size() <= k) berk.emplace_back(r.report.berkowitz[k].model, 0);
            berk[k].second += r.report.berkowitz[k].test.p < 0.05;
        }
        if (r.report.berkowitz_overnight.n > 10) {
            ++berk_o_runs;
            berk_o += r.report.berkowitz_overnight.p < 0.05;
        }
        for (const ModelBacktest& m : r.report.rows) {
            const auto key = std::make_pair(m.model, m.confidence);
            if (!agg.count(key)) order.push_back(m.model);
            Agg& a = agg[key];
            a.n += m.coverage.n;
            a.exc += m.coverage.exceptions;
            ++a.runs;
            a.rej_uc += m.coverage.p_uc < 0.05;
            a.rej_cc += m.coverage.p_cc < 0.05;
            if (m.has_es && std::isfinite(m.es.p2)) {  // p-values need --es-sims > 0
                ++a.es_runs;
                a.rej_es += m.es.p2 < 0.05;
            }
            if (std::isfinite(m.es_ratio)) {
                a.es_ratio_sum += m.es_ratio;
                ++a.es_ratio_n;
            }
            if (m.has_by_time)
                for (int g = 0; g < 3; ++g) {
                    a.seg_n[g] += m.by_time[g].n;
                    a.seg_exc[g] += m.by_time[g].exceptions;
                }
        }
    }
    if (ok_runs == 0) return 1;
    const int failed = seeds - ok_runs;
    order.erase(std::unique(order.begin(), order.end()), order.end());

    say("\nPooled exception rates over %d markets (95%% Wilson intervals) and rejection frequencies at 5%%\n",
                ok_runs);
    say("  %-14s %6s %9s %8s %19s %8s %8s %8s %9s\n", "model", "conf", "forecasts", "rate", "95% CI", "Kupiec",
                "CC", "ES Z2", "ES ratio");
    std::FILE* f = std::fopen((out + "/monte_carlo_summary.csv").c_str(), "wb");
    if (!f) std::fprintf(stderr, "error: cannot write %s/monte_carlo_summary.csv\n", out.c_str());
    if (f)
        std::fprintf(f, "model,confidence,markets,forecasts,exceptions,rate,ci_lo,ci_hi,reject_kupiec,reject_cc,"
                        "reject_es_z2,mean_es_ratio,rate_open,rate_midday,rate_close\n");
    for (const std::string& model : order)
        for (double c : confs) {
            const auto it = agg.find({model, c});
            if (it == agg.end()) continue;
            const Agg& a = it->second;
            double lo, hi;
            wilson_interval(static_cast<int>(a.exc), static_cast<int>(a.n), 0.95, lo, hi);
            const double rate = a.n ? static_cast<double>(a.exc) / static_cast<double>(a.n) : 0.0;
            const double es_ratio = a.es_ratio_n ? a.es_ratio_sum / a.es_ratio_n : std::nan("");
            char es_rej[16] = "     -";
            if (a.es_runs) std::snprintf(es_rej, sizeof es_rej, "%5.1f%%", 100.0 * a.rej_es / a.es_runs);
            say("  %-14s %6.3f %9ld %7.3f%% [%6.3f%%, %6.3f%%] %7.1f%% %7.1f%% %8s %9.3f\n", model.c_str(), c, a.n,
                        100 * rate, 100 * lo, 100 * hi, 100.0 * a.rej_uc / a.runs, 100.0 * a.rej_cc / a.runs, es_rej,
                        es_ratio);
            if (f) {
                auto seg = [&](int g) { return a.seg_n[g] ? static_cast<double>(a.seg_exc[g]) / static_cast<double>(a.seg_n[g]) : std::nan(""); };
                std::fprintf(f, "%s,%.4f,%d,%ld,%ld,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f,%.4f,%.6f,%.6f,%.6f\n", model.c_str(), c,
                             a.runs, a.n, a.exc, rate, lo, hi, static_cast<double>(a.rej_uc) / a.runs,
                             static_cast<double>(a.rej_cc) / a.runs,
                             a.es_runs ? static_cast<double>(a.rej_es) / a.es_runs : std::nan(""), es_ratio, seg(0),
                             seg(1), seg(2));
            }
        }
    if (f) std::fclose(f);

    say("\nPooled exception rate by time of day (open = first 3 bars, close = last 3 bars)\n");
    say("  %-14s %6s %10s %10s %10s\n", "model", "conf", "open", "midday", "close");
    for (const std::string& model : order)
        for (double c : confs) {
            const auto it = agg.find({model, c});
            if (it == agg.end() || it->second.seg_n[0] == 0) continue;
            const Agg& a = it->second;
            say("  %-14s %6.3f", model.c_str(), c);
            for (int g = 0; g < 3; ++g)
                say(" %9.3f%%", 100.0 * static_cast<double>(a.seg_exc[g]) / static_cast<double>(a.seg_n[g]));
            say("\n");
        }
    say("\nBerkowitz test rejections at 5%%:");
    for (const auto& b : berk) say(" %s %.1f%%,", b.first.c_str(), 100.0 * b.second / ok_runs);
    say(" overnight %.1f%% (of %d markets)\n", berk_o_runs ? 100.0 * berk_o / berk_o_runs : 0.0, ok_runs);
    say("Average portfolio nu %.2f; correlation dynamics a %.4f, b %.4f (EWMA: lambda = b)\n", nu_sum / ok_runs,
                a_sum / ok_runs, b_sum / ok_runs);
    if (!f) return 1;
    say("Summary written to %s/monte_carlo_summary.csv (report: report.txt)\n", paths::display(out).c_str());
    if (g_report) std::fclose(g_report);
    if (failed > 0) {  // the pooled numbers above cover the successful markets only
        std::fprintf(stderr, "error: %d of %d markets failed (see above)\n", failed, seeds);
        return 1;
    }
    return 0;
}
