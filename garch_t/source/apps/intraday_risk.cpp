// Intraday risk replay: recomputes portfolio volatility, VaR and ES every bar (15 minutes),
// the overnight risk at each close and, when the dataset has options.csv, the risk of the
// option book by Monte Carlo full revaluation. Ends with a full backtest. The console report
// is also saved as report.txt next to the CSV outputs. See include/garcht/replay.hpp.
//
//   intraday_risk [--data NAME|DIR] [--prices FILE] [--out DIR] [--window-days N]
//                 [--dist t|normal] [--corr ewma|dcc|constant] [--corr-lambda L]
//                 [--tail oos|insample] [--tail-model evt|t] [--diurnal per-asset|pooled] [--no-diurnal] [--z-clip Z] [--threads N]
//                 [--options FILE] [--no-options] [--mc N] [--no-overnight] [--es-sims N]
//                 [--session auto|HH:MM,BAR_MIN,BARS] [--utc-offset MIN] [--daily] [--live HOST:PORT]
//
// With no arguments: dataset data/synthetic, results in garch_t/results/synthetic.
// --live: the bars come from a market-data feed (data/tools/feed_server.py) as they close; the
// dataset's sessions before the first one the feed sends are the calibration history.
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdarg>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "cli.hpp"
#include "garcht/garcht.hpp"
#include "paths.hpp"
#include "tcp_lines.hpp"

using namespace garcht;

namespace {

struct Options {
    std::string data = "synthetic", prices, out, options;  // data: dataset name or folder
    std::string live;                                      // HOST:PORT of a live feed
    bool daily = false, no_options = false, window_given = false, help = false;
    int es_sims = 2000;
    int utc_offset = -180;
    ReplayConfig replay;
};

void usage() {
    std::fprintf(stderr,
                 "usage: intraday_risk [--data NAME|DIR] [--prices FILE] [--out DIR] [--window-days N]\n"
                 "                     [--dist t|normal] [--corr ewma|dcc|constant] [--corr-lambda L]\n"
                 "                     [--tail oos|insample] [--tail-model evt|t] [--diurnal per-asset|pooled] [--no-diurnal] [--z-clip Z] [--threads N]\n"
                 "                     [--options FILE] [--no-options] [--mc N] [--no-overnight] [--es-sims N]\n"
                 "                     [--session auto|HH:MM,BAR_MIN,BARS] [--utc-offset MIN] [--daily] [--live HOST:PORT]\n"
                 "  --data NAME|DIR   dataset: a name searched as data/NAME next to garch_t (default\n"
                 "                    synthetic; also b3_real) or any folder with assets.csv and prices_15m.csv\n"
                 "  --out DIR         output folder (default garch_t/results/NAME)\n"
                 "  --window-days N   estimation window (default: half of the sessions, at most 60)\n"
                 "  --options FILE    option book (default: options.csv of the dataset, if present)\n"
                 "  --corr-lambda L   fixed EWMA correlation decay (default: estimated at each refit)\n"
                 "  --session         HH:MM = session open (local time), bar minutes, bars per session\n"
                 "  --utc-offset MIN  local offset applied to timestamps that carry a zone (Z, +hh:mm);\n"
                 "                    timestamps without a zone are taken as local time already\n"
                 "  --es-sims N       simulations for the Acerbi-Szekely p-values (0 = statistics only)\n"
                 "  --live HOST:PORT  take the bars from a live feed (data/tools/feed_server.py) as they close;\n"
                 "                    the dataset's sessions before the first one it sends are the history\n");
}

bool parse_args(int argc, char** argv, Options& o) {
    EngineConfig& e = o.replay.engine;
    bool fixed_lambda = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool has_val = i + 1 < argc;
        auto val = [&]() { return std::string(argv[++i]); };
        int k = 0;
        if (a == "--help" || a == "-h") {
            o.help = true;
            return false;
        } else if (a == "--daily") o.daily = true;
        else if (a == "--no-options") o.no_options = true;
        else if (a == "--no-diurnal") e.diurnal = false;
        else if (a == "--no-overnight") o.replay.overnight = false;
        else if (!has_val) {
            std::fprintf(stderr, "error: unknown option or missing value: %s\n", a.c_str());
            return false;
        } else if (a == "--data") o.data = val();
        else if (a == "--prices") o.prices = val();
        else if (a == "--live") o.live = val();
        else if (a == "--out") o.out = val();
        else if (a == "--options") o.options = val();
        else if (a == "--window-days") {
            if (!cli::to_int(a, val(), o.replay.window_days, 5, 10000)) return false;
            o.window_given = true;
        } else if (a == "--dist") {
            if ((k = cli::choice(a, val(), {"t", "normal"})) < 0) return false;
            e.spec.dist = k == 0 ? Dist::StudentT : Dist::Normal;
        } else if (a == "--corr") {
            if ((k = cli::choice(a, val(), {"ewma", "dcc", "constant"})) < 0) return false;
            e.correlation = k == 0 ? CorrelationMode::Ewma : k == 1 ? CorrelationMode::Dcc : CorrelationMode::Constant;
        } else if (a == "--corr-lambda") {  // a fixed EWMA decay (no estimation)
            if (!cli::to_double(a, val(), e.corr_lambda, 0.5, 0.9999)) return false;
            e.estimate_correlation = false;
            fixed_lambda = true;
        } else if (a == "--diurnal") {
            if ((k = cli::choice(a, val(), {"per-asset", "pooled"})) < 0) return false;
            e.diurnal_mode = k == 0 ? DiurnalMode::PerAsset : DiurnalMode::Pooled;
        } else if (a == "--tail-model") {
            if ((k = cli::choice(a, val(), {"evt", "t"})) < 0) return false;
            e.tail_model = k == 0 ? TailModel::Evt : TailModel::StudentT;
        } else if (a == "--tail") {
            if ((k = cli::choice(a, val(), {"oos", "insample"})) < 0) return false;
            e.tail = k == 0 ? TailCalibration::OutOfSample : TailCalibration::InSample;
        } else if (a == "--z-clip") {
            if (!cli::to_double(a, val(), e.z_clip, 0.0, 1e6)) return false;
        } else if (a == "--threads") {
            if (!cli::to_int(a, val(), e.threads, 0, 1024)) return false;
        } else if (a == "--mc") {
            if (!cli::to_int(a, val(), o.replay.mc_scenarios, 100, 10000000)) return false;
        } else if (a == "--es-sims") {
            if (!cli::to_int(a, val(), o.es_sims, 0, 10000000)) return false;
        } else if (a == "--utc-offset") {
            if (!cli::to_int(a, val(), o.utc_offset, -16 * 60, 16 * 60)) return false;
        } else if (a == "--session") {
            const std::string v = val();
            if (cli::lower(v) != "auto") {
                int hh = 0, mm = 0, bar = 0, bars = 0;
                char tail = 0;
                if (std::sscanf(v.c_str(), "%d:%d,%d,%d%c", &hh, &mm, &bar, &bars, &tail) != 4 || hh < 0 || hh > 23 ||
                    mm < 0 || mm > 59 || bar <= 0 || bars <= 0 || hh * 60 + mm + bar * bars > 24 * 60) {
                    std::fprintf(stderr, "error: --session expects auto or HH:MM,BAR_MIN,BARS (session open), got '%s'\n",
                                 v.c_str());
                    return false;
                }
                o.replay.detect_session = false;
                o.replay.session = SessionSpec{hh * 60 + mm, bar, bars};
            }
        } else {
            std::fprintf(stderr, "error: unknown option %s\n", a.c_str());
            return false;
        }
    }
    if (fixed_lambda && e.correlation != CorrelationMode::Ewma) {
        std::fprintf(stderr, "error: --corr-lambda applies to --corr ewma only\n");
        return false;
    }
    return true;
}

// Dataset folder, prices file, output folder and option book from the arguments and defaults.
bool resolve_paths(Options& o) {
    namespace fs = std::filesystem;
    const std::string given = o.data;
    o.data = paths::resolve_dataset(o.data);
    std::error_code ec;
    if (!fs::is_directory(o.data, ec)) {
        std::fprintf(stderr,
                     "error: dataset '%s' not found. Keep the 'data' folder next to garch_t\n"
                     "       (data/synthetic, data/b3_real) or pass --data PATH_TO_FOLDER\n",
                     given.c_str());
        return false;
    }
    const bool default_prices = o.prices.empty();
    if (default_prices) o.prices = (fs::path(o.data) / "prices_15m.csv").string();
    if (o.out.empty()) {
        fs::path d(o.data);
        if (!d.has_filename()) d = d.parent_path();
        std::string name = d.filename().string();
        if (name.empty() || name == "." || name == "..") name = "output";
        if (!default_prices) {  // another price file of the same dataset: its own results folder
            std::string stem = fs::path(o.prices).stem().string();
            if (stem.rfind("prices_15m_", 0) == 0) stem = stem.substr(11);
            if (stem != "prices_15m") name += "_" + stem;
        }
        o.out = paths::results_dir(name);
    }
    const std::string book = (fs::path(o.data) / "options.csv").string();
    if (o.options.empty() && !o.no_options && fs::exists(book, ec)) o.options = book;
    if (o.no_options) o.options.clear();
    return true;
}

std::FILE* open_out(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        // Some security suites refuse to *truncate* a file created by another program but
        // allow replacing it: remove and recreate before giving up.
        std::error_code ec;
        std::filesystem::remove(path, ec);
        f = std::fopen(path.c_str(), "wb");
    }
    if (!f) throw std::runtime_error("cannot write " + path + ": " + std::strerror(errno));
    return f;
}

double percentile(std::vector<double> v, double q) {
    if (v.empty()) return std::nan("");
    std::sort(v.begin(), v.end());
    const double pos = q * static_cast<double>(v.size() - 1);
    const size_t lo = static_cast<size_t>(pos);
    const size_t hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (pos - static_cast<double>(lo)) * (v[hi] - v[lo]);
}

std::string tag(double c) {  // 0.975 -> "975", 0.99 -> "99"
    char buf[16];
    std::snprintf(buf, sizeof buf, "%g", c * 100.0);
    std::string s = buf;
    s.erase(std::remove(s.begin(), s.end(), '.'), s.end());
    return s;
}

bool file_exists(const std::string& p) {
    std::error_code ec;
    return std::filesystem::exists(p, ec);
}

// The first k bars of a panel (the history of a live run).
BarPanel head_bars(const BarPanel& p, size_t k) {
    BarPanel h = p;
    const size_t n = p.num_assets();
    h.timestamps.resize(k);
    h.time_key.resize(k);
    h.date.resize(k);
    h.minute.resize(k);
    for (auto* m : {&h.open, &h.high, &h.low, &h.close, &h.volume}) m->resize(k * n);
    return h;
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
    Options opt;
    if (!parse_args(argc, argv, opt)) {
        usage();
        return opt.help ? 0 : 2;
    }
    if (!resolve_paths(opt)) return 2;
    try {
        std::filesystem::create_directories(opt.out);
        g_report = open_out(opt.out + "/report.txt");
        say("dataset: %s\nresults: %s\n", paths::display(opt.data).c_str(), paths::display(opt.out).c_str());
        const std::vector<double>& confs = opt.replay.engine.confidences;

        // ---------------------------------------------------------------- inputs
        ReplayInput in;
        const CsvTable assets_csv = read_csv(opt.data + "/assets.csv");
        const int ct = assets_csv.column("ticker");
        if (ct < 0) throw std::runtime_error(opt.data + "/assets.csv: no ticker column");
        for (const auto& r : assets_csv.rows) in.tickers.push_back(r.at(static_cast<size_t>(ct)));
        const size_t n = in.tickers.size();
        in.initial_quantity.assign(n, 0.0);
        if (file_exists(opt.data + "/portfolio_initial.csv")) {
            const CsvTable pos = read_csv(opt.data + "/portfolio_initial.csv");
            const int pt = pos.column("ticker"), pq = pos.column("quantity");
            if (pt < 0 || pq < 0) throw std::runtime_error("portfolio_initial.csv: needs ticker and quantity columns");
            auto up = [](std::string v) {
                for (char& c : v) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                return v;
            };
            for (size_t r = 0; r < pos.rows.size(); ++r) {
                const auto& row = pos.rows[r];
                if (static_cast<size_t>(std::max(pt, pq)) >= row.size())
                    throw std::runtime_error("portfolio_initial.csv: short row " + std::to_string(r + 2));
                double q = 0.0;
                if (!parse_number(row[static_cast<size_t>(pq)], q) || !std::isfinite(q))
                    throw std::runtime_error("portfolio_initial.csv: invalid quantity on row " + std::to_string(r + 2));
                bool found = false;
                for (size_t i = 0; i < n; ++i)
                    if (up(row[static_cast<size_t>(pt)]) == up(in.tickers[i])) {
                        in.initial_quantity[i] = q;
                        found = true;
                    }
                if (!found)
                    std::fprintf(stderr, "warning: position in unknown ticker %s ignored\n",
                                 row[static_cast<size_t>(pt)].c_str());
            }
        }
        if (file_exists(opt.data + "/fills.csv")) in.fills = load_fills_csv(opt.data + "/fills.csv", opt.utc_offset);
        if (!opt.options.empty()) in.options = load_options_csv(opt.options);

        const auto t_load = std::chrono::steady_clock::now();
        in.bars = load_bars_csv(opt.prices, in.tickers, opt.utc_offset);
        const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count();
        std::unique_ptr<TcpLines> feed;
        std::vector<std::string> received;  // the feed's first lines, read to find where the history ends
        if (!opt.live.empty()) {
            feed = std::make_unique<TcpLines>(opt.live);
            std::string line;
            int first_live_date = 0;
            while (!first_live_date && feed->next(line)) {
                received.push_back(line);
                TimePoint tp;
                if (received.size() > 1 && !line.empty() && line[0] != '#' &&
                    parse_time(line.substr(0, line.find(',')), tp, opt.utc_offset))
                    first_live_date = tp.ymd;
            }
            if (!first_live_date) throw std::runtime_error("the feed at " + opt.live + " sent no bar");
            size_t k = 0;
            while (k < in.bars.num_bars() && in.bars.date[k] < first_live_date) ++k;
            in.bars = head_bars(in.bars, k);
            say("live feed %s: the history is the dataset before %d-%02d-%02d (%zu bars)\n", opt.live.c_str(),
                first_live_date / 10000, first_live_date / 100 % 100, first_live_date % 100, k);
        }
        if (!opt.window_given) {  // half of the sessions, between 20 and 60 days
            std::vector<int> days(in.bars.date.begin(), in.bars.date.end());
            std::sort(days.begin(), days.end());
            const int sessions = static_cast<int>(std::unique(days.begin(), days.end()) - days.begin());
            opt.replay.window_days = std::clamp(sessions / 2, 20, 60);
        }
        say("loaded %zu bars x %zu assets from %s in %.2f s (rows %zu, duplicates %zu, bad %zu, unknown %zu)\n",
                    in.bars.num_bars(), n, paths::display(opt.prices).c_str(), load_s, in.bars.rows_read, in.bars.duplicates,
                    in.bars.bad_rows, in.bars.unknown_tickers);
        size_t missing = 0;
        for (double v : in.bars.close) missing += !(std::isfinite(v) && v > 0);
        if (missing) say("  missing/invalid closes: %zu of %zu cells\n", missing, in.bars.close.size());

        // ---------------------------------------------------------------- replay
        ReplayResult res;
        if (!feed) {
            res = run_replay(in, opt.replay);
        } else {
            size_t k99 = confs.size() - 1;
            for (size_t k = 0; k < confs.size(); ++k)
                if (std::fabs(confs[k] - 0.99) < 1e-12) k99 = k;
            size_t next = 0;
            LiveStats st;
            std::printf("waiting for bars (VaR / ES %g%% of the next bar, money of the stock positions)\n", 100 * confs[k99]);
            res = run_live(in, opt.replay, [&](std::string& l) {
                if (next < received.size()) {
                    l = received[next++];
                    return true;
                }
                return feed->next(l);
            }, st, [&](const BarPanel& p, size_t t, const ForecastRecord* f) {
                if (f)
                    std::printf("  %s  VaR %12.0f  ES %12.0f  sigma %10.0f\n", p.timestamps[t].c_str(), f->garch[k99].var,
                                f->garch[k99].es, f->sigma);
                else
                    std::printf("  %s  (calibrating: the window is not full yet)\n", p.timestamps[t].c_str());
                std::fflush(stdout);
            }, opt.utc_offset);
            say("live feed: %zu rows, %zu bars; late rows dropped %zu, duplicated %zu, unknown ticker %zu, invalid %zu\n",
                st.rows, st.bars, st.late, st.duplicates, st.unknown, st.bad);
        }
        const SessionSpec& ss = res.session;
        say("session: first bar closes %02d:%02d, %d-minute bars, %d bars/day; %zu sessions, window %d days%s\n",
                    (ss.open_minute + ss.bar_minutes) / 60, (ss.open_minute + ss.bar_minutes) % 60, ss.bar_minutes,
                    ss.bars_per_day, res.sessions, opt.replay.window_days,
                    opt.window_given ? "" : " (automatic: half of the sessions, at most 60)");
        if (res.forecasts.empty())
            say("warning: no out-of-sample forecasts: the dataset has too few sessions for the window\n");
        const BacktestReport bt = evaluate_backtest(res, confs, opt.es_sims);

        // ---------------------------------------------------------------- csv outputs
        const bool book = !in.options.empty();
        std::FILE* f = open_out(opt.out + "/risk_15m.csv");
        std::fprintf(f, "forecast_time,target_time,target_slot,net_exposure,gross_exposure,sigma,nu");
        for (double c : confs) std::fprintf(f, ",var_%s", tag(c).c_str());
        for (double c : confs) std::fprintf(f, ",es_%s", tag(c).c_str());
        std::fprintf(f, ",ewma_sigma");
        for (double c : confs) std::fprintf(f, ",ewma_var_%s", tag(c).c_str());
        for (double c : confs) std::fprintf(f, ",ewma_es_%s", tag(c).c_str());
        std::fprintf(f, ",realized_pnl,models_fitted,update_us");
        if (book) {
            for (double c : confs) std::fprintf(f, ",book_mc_var_%s", tag(c).c_str());
            for (double c : confs) std::fprintf(f, ",book_delta_var_%s", tag(c).c_str());
            std::fprintf(f, ",book_realized_pnl,mc_us");
        }
        std::fprintf(f, "\n");
        for (const ForecastRecord& r : res.forecasts) {
            std::fprintf(f, "%s,%s,%d,%.2f,%.2f,%.4f,%.4f", r.forecast_time.c_str(), r.target_time.c_str(),
                         r.target_slot, r.net, r.gross, r.sigma, r.nu);
            for (const auto& m : r.garch) std::fprintf(f, ",%.2f", m.var);
            for (const auto& m : r.garch) std::fprintf(f, ",%.2f", m.es);
            std::fprintf(f, ",%.4f", r.ewma_sigma);
            for (const auto& m : r.ewma) std::fprintf(f, ",%.2f", m.var);
            for (const auto& m : r.ewma) std::fprintf(f, ",%.2f", m.es);
            std::fprintf(f, ",%.2f,%d,%.2f", r.pnl, r.models_fitted, r.update_us);
            if (book) {
                for (const auto& m : r.book_mc) std::fprintf(f, ",%.2f", m.var);
                for (const auto& m : r.book_delta) std::fprintf(f, ",%.2f", m.var);
                std::fprintf(f, ",%.2f,%.1f", r.book_pnl, r.mc_us);
            }
            std::fprintf(f, "\n");
        }
        std::fclose(f);

        if (!res.overnight.empty()) {
            f = open_out(opt.out + "/risk_overnight.csv");
            std::fprintf(f, "forecast_time,target_time,net_exposure,sigma_gap,sigma_first_bar,sigma,nu");
            for (double c : confs) std::fprintf(f, ",var_%s", tag(c).c_str());
            for (double c : confs) std::fprintf(f, ",es_%s", tag(c).c_str());
            std::fprintf(f, ",realized_pnl\n");
            for (const OvernightRecord& o : res.overnight) {
                std::fprintf(f, "%s,%s,%.2f,%.2f,%.2f,%.2f,%.3f", o.forecast_time.c_str(), o.target_time.c_str(), o.net,
                             o.risk.sigma_gap, o.risk.sigma_bar, o.risk.sigma, o.risk.nu);
                for (const auto& m : o.risk.measures) std::fprintf(f, ",%.2f", m.var);
                for (const auto& m : o.risk.measures) std::fprintf(f, ",%.2f", m.es);
                std::fprintf(f, ",%.2f\n", o.pnl);
            }
            std::fclose(f);
        }

        f = open_out(opt.out + "/garch_params_daily.csv");
        std::fprintf(f, "date,ticker,source,status,active,mu,omega,alpha,beta,nu,persistence,half_life_bars,"
                        "uncond_daily_vol,loglik,iterations,rse_alpha,rse_beta,rse_nu\n");
        for (const ParamRecord& p : res.params)
            std::fprintf(f, "%s,%s,%s,%s,%d,%.6g,%.6g,%.6f,%.6f,%.4f,%.6f,%.2f,%.6f,%.4f,%d,%.6f,%.6f,%.4f\n",
                         p.date.c_str(), p.ticker.c_str(), to_string(p.source), to_string(p.status), p.active ? 1 : 0,
                         p.params.mu, p.params.omega, p.params.alpha, p.params.beta, p.params.nu,
                         p.params.persistence(), p.params.half_life(),
                         std::sqrt(p.params.unconditional_variance() * ss.bars_per_day), p.loglik, p.iterations,
                         p.robust_se.alpha, p.robust_se.beta, p.robust_se.nu);
        std::fclose(f);

        f = open_out(opt.out + "/daily_summary.csv");
        // correlation dynamics: EWMA -> a = 1 - lambda, b = lambda; DCC -> (a, b)
        std::fprintf(f, "date,nu_portfolio,corr_a,corr_b,corr_estimated,refit_ms,models_fitted,inactive_assets\n");
        for (const DailyRecord& d : res.daily)
            std::fprintf(f, "%s,%.4f,%.5f,%.5f,%d,%.2f,%d,%d\n", d.date.c_str(), d.nu_p, d.dcc_a, d.dcc_b,
                         d.dcc_estimated ? 1 : 0, d.refit_ms, d.n_fitted, d.n_inactive);
        std::fclose(f);

        f = open_out(opt.out + "/backtest_summary.csv");
        std::fprintf(f, "model,confidence,n,exceptions,rate,rate_ci_lo,rate_ci_hi,expected_rate,kupiec_p,"
                        "christoffersen_ind_p,conditional_coverage_p,traffic_light,es_ratio,es_z2,es_z2_p,"
                        "rate_open,rate_midday,rate_close,kupiec_p_open,kupiec_p_midday,kupiec_p_close\n");
        for (const ModelBacktest& m : bt.rows) {
            const CoverageTest& c = m.coverage;
            std::fprintf(f, "%s,%.4f,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%s,%.4f,%.4f,%.4f", m.model.c_str(),
                         m.confidence, c.n, c.exceptions, c.rate, c.rate_lo, c.rate_hi, c.expected_rate, c.p_uc,
                         c.p_ind, c.p_cc, to_string(c.zone), m.es_ratio, m.has_es ? m.es.z2 : std::nan(""),
                         m.has_es ? m.es.p2 : std::nan(""));
            if (m.has_by_time)
                std::fprintf(f, ",%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", m.by_time[0].rate, m.by_time[1].rate,
                             m.by_time[2].rate, m.by_time[0].p_uc, m.by_time[1].p_uc, m.by_time[2].p_uc);
            else
                std::fprintf(f, ",,,,,,\n");
        }
        std::fprintf(f, "# berkowitz_p: garch %.6f, ewma %.6f, overnight %.6f\n", bt.berkowitz_garch.p,
                     bt.berkowitz_ewma.p, bt.berkowitz_overnight.p);
        std::fclose(f);

        // last snapshot
        {
            const size_t k99 = confs.size() - 1;
            const PortfolioRisk& lr = res.last_risk;
            const double var_factor = lr.sigma > 0 ? lr.garch[k99].var / lr.sigma : 0.0;
            f = open_out(opt.out + "/risk_contributions_last.csv");
            std::fprintf(f, "ticker,quantity,price,exposure,bar_sigma_pct,standalone_var_%s,component_var_%s,share_pct\n",
                         tag(confs[k99]).c_str(), tag(confs[k99]).c_str());
            double total = 0.0;
            for (double v : lr.sigma_contrib) total += v;
            for (size_t i = 0; i < n; ++i)
                std::fprintf(f, "%s,%.0f,%.2f,%.2f,%.4f,%.2f,%.2f,%.2f\n", in.tickers[i].c_str(), res.quantity[i],
                             res.last_price[i], res.exposure[i], 100.0 * lr.asset_sigma[i],
                             std::fabs(lr.position_sigma[i]) * var_factor, lr.sigma_contrib[i] * var_factor,
                             total > 0 ? 100.0 * lr.sigma_contrib[i] / total : 0.0);
            std::fclose(f);
            f = open_out(opt.out + "/diurnal_profile.csv");
            std::fprintf(f, "slot,bar_close,factor_pooled,correlation_scale\n");
            for (size_t k = 0; k < res.diurnal.size(); ++k) {
                const int m = ss.open_minute + static_cast<int>(k + 1) * ss.bar_minutes;
                std::fprintf(f, "%zu,%02d:%02d,%.4f,%.4f\n", k, m / 60, m % 60, res.diurnal[k],
                             k < res.corr_scale.size() ? res.corr_scale[k] : 1.0);
            }
            std::fclose(f);
            f = open_out(opt.out + "/correlation_last.csv");
            std::fprintf(f, "ticker");
            for (const auto& tk : in.tickers) std::fprintf(f, ",%s", tk.c_str());
            std::fprintf(f, "\n");
            for (size_t i = 0; i < n; ++i) {
                std::fprintf(f, "%s", in.tickers[i].c_str());
                for (size_t j = 0; j < n; ++j) std::fprintf(f, ",%.4f", res.correlation[i * n + j]);
                std::fprintf(f, "\n");
            }
            std::fclose(f);
        }

        // ---------------------------------------------------------------- console report
        if (opt.daily)
            for (const DailyRecord& d : res.daily)
                say("%s  nu_p %5.2f  corr a %.4f b %.4f  refit %6.1f ms  fitted %d  inactive %d\n", d.date.c_str(),
                            d.nu_p, d.dcc_a, d.dcc_b, d.refit_ms, d.n_fitted, d.n_inactive);

        const int n_fc = bt.rows.empty() ? 0 : bt.rows.front().coverage.n;
        say("\nBacktest of the one-bar-ahead forecasts (%d bars, overnight gap excluded)\n", n_fc);
        say("  %-14s %6s %5s %7s %17s %8s %9s %8s %8s %7s %8s\n", "model", "conf", "exc", "rate", "95% CI",
                    "expected", "Kupiec p", "CC p", "ES Z2 p", "zone", "ES ratio");
        for (const ModelBacktest& m : bt.rows) {
            if (!m.has_by_time) continue;
            const CoverageTest& c = m.coverage;
            say("  %-14s %6.3f %5d %6.2f%% [%5.2f%%, %5.2f%%] %7.2f%% %9.4f %8.4f %8.4f %7s %8.3f\n",
                        m.model.c_str(), m.confidence, c.exceptions, 100 * c.rate, 100 * c.rate_lo, 100 * c.rate_hi,
                        100 * c.expected_rate, c.p_uc, c.p_cc, m.es.p2, to_string(c.zone), m.es_ratio);
        }
        say("  Berkowitz density test (whole distribution): %s p = %.4f (rho %.3f, sd %.3f); EWMA-N p = %.4f "
                    "(rho %.3f, sd %.3f)\n",
                    res.model_name.c_str(), bt.berkowitz_garch.p, bt.berkowitz_garch.rho,
                    bt.berkowitz_garch.sigma, bt.berkowitz_ewma.p, bt.berkowitz_ewma.rho, bt.berkowitz_ewma.sigma);

        say("\nCoverage by time of day (exception rate, Kupiec p in brackets)\n");
        say("  %-14s %6s %20s %20s %20s\n", "model", "conf", "open (3 bars)", "midday", "close (3 bars)");
        for (const ModelBacktest& m : bt.rows) {
            if (!m.has_by_time) continue;
            say("  %-14s %6.3f", m.model.c_str(), m.confidence);
            for (const CoverageTest& c : m.by_time) say("      %5.2f%% [%6.4f]", 100 * c.rate, c.p_uc);
            say("\n");
        }

        bool header = false;
        for (const ModelBacktest& m : bt.rows) {
            if (m.has_by_time) continue;
            if (!header) {
                say("\nOther horizons / books\n  %-20s %6s %5s %5s %7s %17s %9s %8s %7s %8s\n", "model", "conf",
                            "n", "exc", "rate", "95% CI", "Kupiec p", "CC p", "zone", "ES ratio");
                header = true;
            }
            const CoverageTest& c = m.coverage;
            say("  %-20s %6.3f %5d %5d %6.2f%% [%5.2f%%, %5.2f%%] %9.4f %8.4f %7s %8.3f\n", m.model.c_str(),
                        m.confidence, c.n, c.exceptions, 100 * c.rate, 100 * c.rate_lo, 100 * c.rate_hi, c.p_uc, c.p_cc,
                        to_string(c.zone), m.es_ratio);
        }
        if (!res.overnight.empty())
            say("  overnight Berkowitz p = %.4f (%d sessions)\n", bt.berkowitz_overnight.p, bt.berkowitz_overnight.n);

        const CorrelationMode cm = opt.replay.engine.correlation;
        char corr_desc[96];
        if (res.daily.empty() || cm == CorrelationMode::Constant)
            std::snprintf(corr_desc, sizeof corr_desc, "constant correlation");
        else if (cm == CorrelationMode::Ewma)
            std::snprintf(corr_desc, sizeof corr_desc, "EWMA correlation, lambda = %.4f", res.daily.back().dcc_b);
        else
            std::snprintf(corr_desc, sizeof corr_desc, "DCC correlation, a = %.4f, b = %.4f", res.daily.back().dcc_a,
                          res.daily.back().dcc_b);
        say("\nGARCH(1,1)-%s models at the last refit (deseasonalized returns); %s\n",
                    res.student_t ? "t" : "N", corr_desc);
        say("  %-7s %-12s %6s %8s %8s %8s %7s %9s %10s\n", "ticker", "source", "active", "alpha", "beta", "a+b",
                    "nu", "half-life", "daily vol");
        for (const AssetModel& a : res.models)
            say("  %-7s %-12s %6s %8.4f %8.4f %8.4f %7.2f %8.1fb %9.2f%%\n", a.ticker.c_str(), to_string(a.source),
                        a.active ? "yes" : "no", a.params.alpha, a.params.beta, a.params.persistence(), a.params.nu,
                        a.params.half_life(), 100 * std::sqrt(a.params.unconditional_variance() * ss.bars_per_day));
        const size_t k99 = confs.size() - 1;
        say("\nLast forecast (%s -> next bar): sigma %.0f, VaR%s %.0f, ES%s %.0f, nu_p %.2f\n",
                    res.last_time.c_str(), res.last_risk.sigma, tag(confs[k99]).c_str(), res.last_risk.garch[k99].var,
                    tag(confs[k99]).c_str(), res.last_risk.garch[k99].es, res.last_risk.nu);

        const int thr = opt.replay.engine.threads > 0 ? opt.replay.engine.threads
                                                      : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
        say("\nPerformance (%zu assets, %d threads for refits)\n", n, thr);
        say("  per-bar update (fills + filters + correlations + VaR/ES): p50 %.2f us, p95 %.2f us, p99 %.2f us, "
                    "max %.2f us\n",
                    percentile(res.latency_us, 0.50), percentile(res.latency_us, 0.95), percentile(res.latency_us, 0.99),
                    percentile(res.latency_us, 1.0));
        say("  end-of-day refit (GARCH fits + correlation dynamics + tails + overnight): p50 %.1f ms, p95 %.1f ms, max %.1f ms\n",
                    percentile(res.refit_ms, 0.5), percentile(res.refit_ms, 0.95), percentile(res.refit_ms, 1.0));
        if (book) {
            std::vector<double> mcu;
            for (const ForecastRecord& r : res.forecasts) mcu.push_back(r.mc_us / 1000.0);
            say("  option book Monte Carlo (%d scenarios, full revaluation): p50 %.2f ms per bar\n",
                        opt.replay.mc_scenarios, percentile(mcu, 0.5));
        }
        say("  replay: %zu bars, %zu forecasts scored (%zu unscored: target bar never printed; %zu overnight "
                    "unscored: late opening), %zu fills applied (%zu unknown), %zu off-grid bars, engine %.3f s, "
                    "wall %.3f s\n",
                    res.bars, res.forecasts.size(), res.unscored, res.unscored_overnight, res.fills_applied,
                    res.fills_unknown, res.off_grid, res.engine_seconds, res.wall_seconds);
        say("\noutputs written to %s (report: report.txt)\n", paths::display(opt.out).c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        if (g_report) std::fprintf(g_report, "error: %s\n", e.what());
        if (g_report) std::fclose(g_report);
        return 1;
    }
    if (g_report) std::fclose(g_report);
    return 0;
}
