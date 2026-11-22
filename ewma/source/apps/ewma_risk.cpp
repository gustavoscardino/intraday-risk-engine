// EWMA intraday risk: replays a 15-minute price history through the data-quality layer and the
// EWMA engine, forecasting VaR and ES of the next bar with three models side by side
// (EWMA-N (ref), EWMA-N, EWMA-t), the overnight risk at every close and, with options.csv, the
// delta-linear risk of the option book. Ends with the backtest of every forecast. The console
// report is also saved as report.txt next to the CSV outputs.
//
//   ewma_risk [--data NAME|DIR] [--prices FILE] [--out DIR] [--window-days N]
//             [--lambda L] [--nu N] [--diurnal pooled|per-asset|off] [--no-overnight]
//             [--options FILE] [--no-options] [--no-dq] [--no-jump-filter] [--dq-thresholds A,B] [--truth FILE]
//             [--es-sims N] [--session auto|HH:MM,BAR_MIN,BARS] [--resample] [--utc-offset MIN] [--daily]
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli.hpp"
#include "ewmar/ewmar.hpp"
#include "paths.hpp"

using namespace ewmar;
namespace fs = std::filesystem;

namespace {

// Default estimation window (sessions), the same in every engine of the repository: fixed, so that
// a forecast never depends on how many sessions follow it in the file.
constexpr int kDefaultWindow = 40;

struct Options {
    std::string data = "synthetic", prices, out, options, truth;
    bool no_options = false, daily = false, help = false, window_given = false, no_truth = false, resample = false;
    int es_sims = 2000, utc_offset = -180;
    ReplayConfig replay;
};

void usage() {
    std::fprintf(stderr,
                 "usage: ewma_risk [--data NAME|DIR] [--prices FILE] [--out DIR] [--window-days N]\n"
                 "                 [--lambda L] [--nu N] [--diurnal pooled|per-asset|off] [--no-overnight]\n"
                 "                 [--options FILE] [--no-options] [--no-dq] [--no-jump-filter] [--dq-thresholds A,B]\n"
                 "                 [--truth FILE|none]\n"
                 "                 [--es-sims N] [--session auto|HH:MM,BAR_MIN,BARS] [--resample] [--utc-offset MIN]\n"
                 "                 [--daily]\n"
                 "  --data NAME|DIR      dataset: data/NAME next to ewma (synthetic, b3_real) or any folder\n"
                 "                       with assets.csv and prices_15m.csv (default synthetic)\n"
                 "  --prices FILE        another price file of the dataset (e.g. prices_15m_degraded.csv)\n"
                 "  --out DIR            output folder (default ewma/results/NAME[_degraded][_nodq])\n"
                 "  --window-days N      estimation window in sessions (default 40; half of the sessions if fewer)\n"
                 "  --lambda L           fixes the lambda of EWMA-N / EWMA-t (default: estimated every day)\n"
                 "  --nu N               fixes the nu of EWMA-t (default: estimated every day)\n"
                 "  --diurnal MODE       intraday profile: pooled (default), per-asset or off\n"
                 "  --no-dq              disables the data-quality layer (plain loader, no jump filter)\n"
                 "  --no-jump-filter     keeps the layer but lets every isolated jump into the models\n"
                 "  --dq-thresholds A,B  dq_score limits of NORMAL and FREEZE (default 0.9,0.6)\n"
                 "  --truth FILE|none    clean price file for the detection rates and the robustness\n"
                 "                       comparison (default: prices_15m.csv of the dataset, when --prices\n"
                 "                       is another file)\n"
                 "  --no-overnight       skips the overnight risk\n"
                 "  --options FILE       option book (default: options.csv of the dataset, if present);\n"
                 "                       --no-options ignores it\n"
                 "  --resample           the price file has samples finer than the --session grid (e.g. 5 s):\n"
                 "                       each bar takes the first open, high, low, last close and summed\n"
                 "                       volume of its samples (needs --session)\n"
                 "  --session SPEC       auto (default) or HH:MM,BAR_MIN,BARS: session open (local time),\n"
                 "                       minutes per bar and bars per session\n"
                 "  --utc-offset MIN     offset applied to timestamps that carry a zone (default -180)\n"
                 "  --daily              prints one summary line per day\n"
                 "  --es-sims N          simulations of the ES test p-value (0 = statistic only; default 2000)\n");
}

bool parse_args(int argc, char** argv, Options& o) {
    EngineConfig& e = o.replay.engine;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool has = i + 1 < argc;
        auto val = [&]() { return std::string(argv[++i]); };
        int k = 0;
        if (a == "--help" || a == "-h") {
            o.help = true;
            return false;
        } else if (a == "--daily") o.daily = true;
        else if (a == "--no-options") o.no_options = true;
        else if (a == "--no-overnight") e.overnight = false;
        else if (a == "--no-dq") o.replay.dq = false;
        else if (a == "--no-jump-filter") o.replay.dq_cfg.jump_filter = false;
        else if (a == "--resample") o.resample = true;
        else if (!has) {
            std::fprintf(stderr, "error: unknown option or missing value: %s\n", a.c_str());
            return false;
        } else if (a == "--data") o.data = val();
        else if (a == "--prices") o.prices = val();
        else if (a == "--out") o.out = val();
        else if (a == "--options") o.options = val();
        else if (a == "--truth") {
            o.truth = val();
            if (cli::lower(o.truth) == "none") {
                o.truth.clear();
                o.no_truth = true;
            }
        } else if (a == "--window-days") {
            if (!cli::int_arg(a, val(), o.replay.window_days, 5, 10000)) return false;
            o.window_given = true;
        } else if (a == "--lambda") {
            if (!cli::double_arg(a, val(), e.lambda, 0.5, 0.9999)) return false;
            e.estimate_lambda = false;
        } else if (a == "--nu") {
            if (!cli::double_arg(a, val(), e.nu, 2.05, 1000.0)) return false;
            e.estimate_nu = false;
        } else if (a == "--diurnal") {
            if ((k = cli::choice_arg(a, val(), {"pooled", "per-asset", "off"})) < 0) return false;
            e.diurnal = k == 0 ? DiurnalMode::Pooled : (k == 1 ? DiurnalMode::PerAsset : DiurnalMode::Off);
        } else if (a == "--dq-thresholds") {
            const std::string v = val();
            double hi = 0.0, lo = 0.0;
            char extra = 0;
            if (std::sscanf(v.c_str(), "%lf,%lf%c", &hi, &lo, &extra) != 2 || !(hi > lo) || lo < 0.0 || hi > 1.0) {
                std::fprintf(stderr, "error: --dq-thresholds expects NORMAL,FREEZE with 1 >= NORMAL > FREEZE >= 0, got '%s'\n",
                             v.c_str());
                return false;
            }
            o.replay.dq_cfg.normal_threshold = hi;
            o.replay.dq_cfg.freeze_threshold = lo;
        } else if (a == "--es-sims") {
            if (!cli::int_arg(a, val(), o.es_sims, 0, 10000000)) return false;
        } else if (a == "--utc-offset") {
            if (!cli::int_arg(a, val(), o.utc_offset, -16 * 60, 16 * 60)) return false;
        } else if (a == "--session") {
            const std::string v = val();
            if (cli::lower(v) != "auto") {
                int hh = 0, mm = 0, bar = 0, bars = 0;
                char extra = 0;
                if (std::sscanf(v.c_str(), "%d:%d,%d,%d%c", &hh, &mm, &bar, &bars, &extra) != 4 || hh < 0 || hh > 23 ||
                    mm < 0 || mm > 59 || bar <= 0 || bars <= 0 || hh * 60 + mm + bar * bars > 24 * 60) {
                    std::fprintf(stderr, "error: --session expects auto or HH:MM,BAR_MIN,BARS, got '%s'\n", v.c_str());
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
    return true;
}

bool same_file(const std::string& a, const std::string& b) {
    std::error_code ec;
    return fs::equivalent(a, b, ec) && !ec;
}

bool resolve_paths(Options& o) {
    const std::string given = o.data;
    o.data = paths::resolve_dataset(o.data);
    std::error_code ec;
    if (!fs::is_directory(o.data, ec)) {
        std::fprintf(stderr,
                     "error: dataset '%s' not found. Keep the 'data' folder next to ewma (data/synthetic,\n"
                     "       data/b3_real) or pass --data PATH_TO_FOLDER\n",
                     given.c_str());
        return false;
    }
    const std::string default_prices = (fs::path(o.data) / "prices_15m.csv").string();
    if (o.prices.empty()) o.prices = default_prices;
    if (o.out.empty()) {
        fs::path d(o.data);
        if (!d.has_filename()) d = d.parent_path();
        std::string name = d.filename().string();
        if (name.empty() || name == "." || name == "..") name = "output";
        if (!same_file(o.prices, default_prices)) {
            std::string stem = fs::path(o.prices).stem().string();
            if (stem.rfind("prices_15m_", 0) == 0) stem = stem.substr(11);
            name += "_" + stem;
        }
        if (!o.replay.dq) name += "_nodq";
        o.out = paths::results_dir(name);
    }
    const std::string book = (fs::path(o.data) / "options.csv").string();
    if (o.no_options) o.options.clear();
    else if (o.options.empty() && fs::exists(book, ec)) o.options = book;
    if (o.truth.empty() && !o.no_truth && fs::exists(default_prices, ec)) o.truth = default_prices;
    return true;
}

std::FILE* open_out(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {  // some antivirus suites refuse to truncate a file another program created
        std::error_code ec;
        fs::remove(path, ec);
        f = std::fopen(path.c_str(), "wb");
    }
    if (!f) throw std::runtime_error("cannot write " + path + ": " + std::strerror(errno));
    return f;
}

double percentile(std::vector<double> v, double q) {
    if (v.empty()) return std::nan("");
    std::sort(v.begin(), v.end());
    const double pos = q * static_cast<double>(v.size() - 1);
    const size_t lo = static_cast<size_t>(pos), hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (pos - static_cast<double>(lo)) * (v[hi] - v[lo]);
}

std::string tag(double c) {  // 0.975 -> "975"
    char buf[16];
    std::snprintf(buf, sizeof buf, "%g", c * 100.0);
    std::string s = buf;
    s.erase(std::remove(s.begin(), s.end(), '.'), s.end());
    return s;
}

std::string csv_text(const std::string& s) {  // quotes a field when needed
    if (s.find_first_of(",\"\n") == std::string::npos) return s;
    std::string q = "\"";
    for (char c : s) q += c == '"' ? std::string("\"\"") : std::string(1, c);
    return q + "\"";
}

std::FILE* g_report = nullptr;
void say(const char* fmt, ...) {
    va_list a, b;
    va_start(a, fmt);
    va_copy(b, a);
    std::vprintf(fmt, a);
    if (g_report) std::vfprintf(g_report, fmt, b);
    va_end(b);
    va_end(a);
}

ReplayInput load_inputs(const Options& o, const std::string& prices) {
    ReplayInput in;
    const CsvTable assets = read_csv(o.data + "/assets.csv");
    const int ct = assets.column("ticker");
    if (ct < 0) throw std::runtime_error(o.data + "/assets.csv: no ticker column");
    for (const auto& row : assets.rows) in.tickers.push_back(row.at(static_cast<size_t>(ct)));
    const size_t n = in.tickers.size();
    in.initial_quantity.assign(n, 0.0);
    std::error_code ec;
    if (fs::exists(o.data + "/portfolio_initial.csv", ec)) {
        const CsvTable pos = read_csv(o.data + "/portfolio_initial.csv");
        const int pt = pos.column("ticker"), pq = pos.column("quantity");
        if (pt < 0 || pq < 0) throw std::runtime_error("portfolio_initial.csv: needs ticker and quantity columns");
        for (size_t r = 0; r < pos.rows.size(); ++r) {
            const auto& row = pos.rows[r];
            double q = 0.0;
            if (static_cast<size_t>(std::max(pt, pq)) >= row.size() || !parse_number(row[static_cast<size_t>(pq)], q) ||
                !std::isfinite(q))
                throw std::runtime_error("portfolio_initial.csv: invalid row at line " + std::to_string(pos.line[r]));
            bool found = false;
            for (size_t i = 0; i < n; ++i)
                if (cli::lower(row[static_cast<size_t>(pt)]) == cli::lower(in.tickers[i])) {
                    in.initial_quantity[i] = q;
                    found = true;
                }
            if (!found) std::fprintf(stderr, "warning: position in unknown ticker %s ignored\n", row[static_cast<size_t>(pt)].c_str());
        }
    }
    if (fs::exists(o.data + "/fills.csv", ec)) in.fills = load_fills(o.data + "/fills.csv", o.utc_offset);
    if (!o.options.empty()) in.options = load_options(o.options);
    in.feed = load_feed(prices, in.tickers, o.utc_offset);
    if (o.resample) {
        if (o.replay.detect_session) throw std::runtime_error("--resample needs the bar grid: pass --session HH:MM,BAR_MIN,BARS");
        in.feed = resample_feed(in.feed, o.replay.session);
    }
    return in;
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
        fs::create_directories(opt.out);
        g_report = open_out(opt.out + "/report.txt");
        say("EWMA intraday risk (EWMA-N (ref), EWMA-N, EWMA-t)\n");
        say("dataset: %s\nprices:  %s\nresults: %s\n", paths::display(opt.data).c_str(), paths::display(opt.prices).c_str(),
            paths::display(opt.out).c_str());
        const std::vector<double>& confs = opt.replay.engine.confidences;
        const size_t k99 = confs.size() - 1;

        // ---------------------------------------------------------------- inputs
        ReplayInput in = load_inputs(opt, opt.prices);
        const size_t n = in.tickers.size();
        bool short_data = false;
        if (!opt.window_given) {  // fixed, as in every engine, so forecasts do not depend on the file's length
            opt.replay.window_days = kDefaultWindow;
            std::vector<int> days;
            for (const RawRow& r : in.feed.rows) days.push_back(r.time.ymd);
            std::sort(days.begin(), days.end());
            const int sessions = static_cast<int>(std::unique(days.begin(), days.end()) - days.begin());
            if (sessions <= kDefaultWindow) {  // too short a file: half of it
                opt.replay.window_days = std::max(5, sessions / 2);
                short_data = true;
            }
        }
        say("read %zu rows x %zu assets (unparseable %zu, unknown tickers %zu); %zu fills, %zu options\n", in.feed.lines, n,
            in.feed.bad_rows, in.feed.unknown_rows, in.fills.size(), in.options.size());

        // ---------------------------------------------------------------- replay
        const ReplayResult res = run_replay(in, opt.replay);
        const SessionSpec& ss = res.session;
        const size_t main = res.main_model;  // the model behind the book, the decomposition and the robustness check
        say("session: first bar closes %02d:%02d, %d-minute bars, %d bars/day; %zu sessions, window %d days%s\n",
            ss.close_minute(0) / 60, ss.close_minute(0) % 60, ss.bar_minutes, ss.bars_per_day, res.sessions,
            opt.replay.window_days,
            opt.window_given ? "" : short_data ? " (default 40 does not fit: half of the sessions)" : " (default)");
        const EngineConfig& ec = opt.replay.engine;
        say("models: lambda %s, nu %s, diurnal profile %s, overnight %s, data-quality layer %s\n",
            ec.estimate_lambda ? "estimated daily" : "fixed", ec.estimate_nu ? "estimated daily" : "fixed",
            to_string(ec.diurnal), ec.overnight ? "on" : "off", res.dq ? "ON" : "OFF (--no-dq)");
        if (res.forecasts.empty()) say("warning: no out-of-sample forecasts (too few sessions for the window)\n");
        const BacktestReport bt = evaluate_backtest(res, confs, opt.es_sims);

        // ---------------------------------------------------------------- truth: detection and robustness
        std::vector<DetectionRow> detection;
        bool have_robust = false;
        size_t rob_n = 0, rob_affected = 0;
        double rob_mean = 0.0, rob_max = 0.0;
        std::string rob_max_time;
        BacktestReport bt_clean;
        if (!opt.truth.empty()) {
            const bool self = same_file(opt.truth, opt.prices);
            ReplayInput clean = self ? ReplayInput{} : load_inputs(opt, opt.truth);
            const FeedFile& clean_feed = self ? in.feed : clean.feed;
            // detection needs an independent clean file: a feed compared with itself has no truth
            if (res.dq && !self)
                detection = evaluate_detection(clean_feed, in.feed, ss, res.ingest.cell_flags, opt.replay.dq_cfg.lateness_bars);
            if (!self) {  // the same market through a clean feed (DQ layer on)
                ReplayConfig rc = opt.replay;
                rc.dq = true;
                const ReplayResult cr = run_replay(clean, rc);
                bt_clean = evaluate_backtest(cr, confs, 0);
                std::map<std::string, double> clean_var;
                for (const ForecastRecord& f : cr.forecasts) clean_var[f.forecast_time] = f.models[main].measures[k99].var;
                double sum = 0.0;
                for (const ForecastRecord& f : res.forecasts) {
                    const auto it = clean_var.find(f.forecast_time);
                    if (it == clean_var.end() || !(it->second > 0.0)) continue;
                    const double d = std::fabs(f.models[main].measures[k99].var / it->second - 1.0);
                    sum += d;
                    ++rob_n;
                    if (d > 0.01) ++rob_affected;
                    if (d > rob_max) {
                        rob_max = d;
                        rob_max_time = f.forecast_time;
                    }
                }
                rob_mean = rob_n ? sum / static_cast<double>(rob_n) : 0.0;
                have_robust = rob_n > 0;
            }
        }

        // ---------------------------------------------------------------- CSV outputs
        std::FILE* f = open_out(opt.out + "/risk_15m.csv");
        std::fprintf(f, "forecast_time,target_time,target_slot,net_exposure,gross_exposure");
        std::vector<std::string> conf_tags;
        for (double c : confs) conf_tags.push_back(tag(c));
        for (const riskcommon::ModelInfo& m : res.var_models)  // one block of columns per model
            for (const std::string& col : riskcommon::forecast_columns(m, conf_tags)) std::fprintf(f, ",%s", col.c_str());
        std::fprintf(f, ",realized_pnl,dq_score,mode,frozen,update_us");
        if (res.has_book) {
            std::fprintf(f, ",book_sigma");
            for (double c : confs) std::fprintf(f, ",book_var_%s", tag(c).c_str());
            for (double c : confs) std::fprintf(f, ",book_es_%s", tag(c).c_str());
            std::fprintf(f, ",book_realized_pnl");
        }
        std::fprintf(f, "\n");
        for (const ForecastRecord& r : res.forecasts) {
            std::fprintf(f, "%s,%s,%d,%.2f,%.2f", r.forecast_time.c_str(), r.target_time.c_str(), r.target_slot, r.net,
                         r.gross);
            for (size_t m = 0; m < res.var_models.size(); ++m)
                std::fputs(riskcommon::format_forecast(res.var_models[m], r.models[m]).c_str(), f);
            std::fprintf(f, ",%.2f,%.4f,%s,%d,%.2f", r.pnl, r.dq_score, to_string(r.mode), r.frozen ? 1 : 0, r.update_us);
            if (res.has_book) {
                std::fprintf(f, ",%.4f", r.book_sigma);
                for (const auto& m : r.book) std::fprintf(f, ",%.2f", m.var);
                for (const auto& m : r.book) std::fprintf(f, ",%.2f", m.es);
                std::fprintf(f, ",%.2f", r.book_pnl);
            }
            std::fprintf(f, "\n");
        }
        std::fclose(f);

        if (!res.overnight.empty()) {
            f = open_out(opt.out + "/risk_overnight.csv");
            std::fprintf(f, "forecast_time,target_time,net_exposure,sigma_gap,sigma_first_bar,sigma,lambda_overnight,nu");
            for (double c : confs) std::fprintf(f, ",var_%s", tag(c).c_str());
            for (double c : confs) std::fprintf(f, ",es_%s", tag(c).c_str());
            for (double c : confs) std::fprintf(f, ",n_var_%s", tag(c).c_str());
            for (double c : confs) std::fprintf(f, ",n_es_%s", tag(c).c_str());
            std::fprintf(f, ",realized_pnl\n");
            for (const OvernightRecord& o : res.overnight) {
                std::fprintf(f, "%s,%s,%.2f,%.2f,%.2f,%.2f,%.5f,%.3f", o.forecast_time.c_str(), o.target_time.c_str(), o.net,
                             o.risk.sigma_gap, o.risk.sigma_bar, o.risk.sigma, o.risk.lambda, o.risk.nu);
                for (const auto& m : o.risk.student) std::fprintf(f, ",%.2f", m.var);
                for (const auto& m : o.risk.student) std::fprintf(f, ",%.2f", m.es);
                for (const auto& m : o.risk.normal) std::fprintf(f, ",%.2f", m.var);
                for (const auto& m : o.risk.normal) std::fprintf(f, ",%.2f", m.es);
                std::fprintf(f, ",%.2f\n", o.pnl);
            }
            std::fclose(f);
        }

        f = open_out(opt.out + "/backtest_summary.csv");  // same columns as garch_t
        std::fprintf(f, "model,confidence,n,exceptions,rate,rate_ci_lo,rate_ci_hi,expected_rate,kupiec_p,"
                        "christoffersen_ind_p,conditional_coverage_p,traffic_light,es_ratio,es_z2,es_z2_p,"
                        "rate_open,rate_midday,rate_close,kupiec_p_open,kupiec_p_midday,kupiec_p_close\n");
        for (const ModelBacktest& m : bt.rows) {
            const CoverageTest& c = m.coverage;
            std::fprintf(f, "%s,%.4f,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%s,%.4f,%.4f,%.4f", m.model.c_str(),
                         m.confidence, c.n, c.exceptions, c.rate, c.rate_lo, c.rate_hi, c.expected_rate, c.p_uc, c.p_ind,
                         c.p_cc, to_string(c.zone), m.es_ratio, m.has_es ? m.es.z2 : std::nan(""),
                         m.has_es ? m.es.p2 : std::nan(""));
            if (m.has_by_time)
                std::fprintf(f, ",%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", m.by_time[0].rate, m.by_time[1].rate, m.by_time[2].rate,
                             m.by_time[0].p_uc, m.by_time[1].p_uc, m.by_time[2].p_uc);
            else
                std::fprintf(f, ",,,,,,\n");
        }
        std::fprintf(f, "# berkowitz_p:");
        for (size_t k = 0; k < bt.berkowitz.size(); ++k)
            std::fprintf(f, "%s %s %.6f", k ? "," : "", bt.berkowitz[k].model.c_str(), bt.berkowitz[k].test.p);
        std::fprintf(f, "\n");
        std::fclose(f);

        f = open_out(opt.out + "/daily_summary.csv");
        std::fprintf(f, "date,lambda,nu_portfolio,lambda_overnight,nu_overnight,nu_observations,refit_ms\n");
        for (const DailyRecord& d : res.daily)
            std::fprintf(f, "%s,%.5f,%.4f,%.5f,%.4f,%zu,%.3f\n", d.date.c_str(), d.lambda, d.nu, d.lambda_on, d.nu_on,
                         d.nu_obs, d.refit_ms);
        std::fclose(f);

        f = open_out(opt.out + "/diurnal_profile.csv");
        std::fprintf(f, "slot,bar_close,factor_pooled");
        const bool per_asset = ec.diurnal == DiurnalMode::PerAsset;
        if (per_asset)
            for (const auto& tk : in.tickers) std::fprintf(f, ",%s", tk.c_str());
        std::fprintf(f, "\n");
        for (int k = 0; k < ss.bars_per_day; ++k) {
            const int m = ss.close_minute(k);
            std::fprintf(f, "%d,%02d:%02d,%.4f", k, m / 60, m % 60, res.diurnal_pooled[static_cast<size_t>(k)]);
            if (per_asset)
                for (size_t i = 0; i < n; ++i) std::fprintf(f, ",%.4f", res.diurnal[static_cast<size_t>(k) * n + i]);
            std::fprintf(f, "\n");
        }
        std::fclose(f);

        {
            const PortfolioRisk& lr = res.last_risk;
            const double sig = lr.ready ? lr.models[main].sigma : 0.0;
            const double factor = sig > 0.0 ? lr.models[main].measures[k99].var / sig : 0.0;
            f = open_out(opt.out + "/risk_contributions_last.csv");
            std::fprintf(f, "ticker,quantity,price,exposure,bar_sigma_pct,standalone_var_%s,component_var_%s,share_pct\n",
                         tag(confs[k99]).c_str(), tag(confs[k99]).c_str());
            for (size_t i = 0; i < n; ++i)
                std::fprintf(f, "%s,%.0f,%.2f,%.2f,%.4f,%.2f,%.2f,%.2f\n", in.tickers[i].c_str(), res.quantity[i],
                             res.last_price[i], res.exposure[i], 100.0 * lr.asset_sigma[i], lr.standalone[i] * factor,
                             lr.contrib[i] * factor, sig > 0.0 ? 100.0 * lr.contrib[i] / sig : 0.0);
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

        if (res.dq) {
            f = open_out(opt.out + "/dq_events.csv");
            std::fprintf(f, "time,ticker,flag,detail\n");
            for (const DqEvent& e : res.events)
                std::fprintf(f, "%s,%s,%s,%s\n", e.time.c_str(), e.ticker.c_str(), e.flag.c_str(), csv_text(e.detail).c_str());
            std::fclose(f);
            f = open_out(opt.out + "/alerts.csv");
            std::fprintf(f, "time,severity,type,message\n");
            for (const DqAlert& a : res.alerts)
                std::fprintf(f, "%s,%s,%s,%s\n", a.time.c_str(), a.severity.c_str(), a.type.c_str(), csv_text(a.message).c_str());
            std::fclose(f);
            if (opt.replay.dq_cfg.jump_filter) {  // the same file as garch_t writes
                f = open_out(opt.out + "/jump_events.csv");
                std::fprintf(f, "time,ticker,action,return,z\n");
                for (const riskcommon::JumpEvent& e : res.jump_events)
                    std::fprintf(f, "%s,%s,%s,%.6f,%.2f\n", e.time.c_str(), e.ticker.c_str(), e.action.c_str(), e.ret, e.z);
                std::fclose(f);
            }
        }
        if (!detection.empty()) {
            f = open_out(opt.out + "/dq_detection.csv");
            std::fprintf(f, "flag,injected,detected,recall,flags_on_other_cells\n");
            for (const DetectionRow& d : detection)
                std::fprintf(f, "%s,%zu,%zu,%.4f,%zu\n", d.flag.c_str(), d.injected, d.detected, d.recall, d.other);
            std::fclose(f);
        }

        // ---------------------------------------------------------------- console report
        if (opt.daily)
            for (const DailyRecord& d : res.daily)
                say("%s  lambda %.4f  nu %6.2f  overnight lambda %.4f nu %6.2f  refit %.2f ms\n", d.date.c_str(), d.lambda,
                    d.nu, d.lambda_on, d.nu_on, d.refit_ms);

        const int n_fc = bt.rows.empty() ? 0 : bt.rows.front().coverage.n;
        say("\nBacktest of the one-bar-ahead forecasts (%d bars, overnight gap excluded)\n", n_fc);
        say("  %-14s %6s %5s %7s %17s %8s %9s %8s %8s %7s %8s\n", "model", "conf", "exc", "rate", "95% CI", "expected",
            "Kupiec p", "CC p", "ES Z2 p", "zone", "ES ratio");
        for (const ModelBacktest& m : bt.rows) {
            if (!m.has_by_time) continue;
            const CoverageTest& c = m.coverage;
            say("  %-14s %6.3f %5d %6.2f%% [%5.2f%%, %5.2f%%] %7.2f%% %9.4f %8.4f %8.4f %7s %8.3f\n", m.model.c_str(),
                m.confidence, c.exceptions, 100 * c.rate, 100 * c.rate_lo, 100 * c.rate_hi, 100 * c.expected_rate, c.p_uc,
                c.p_cc, m.es.p2, to_string(c.zone), m.es_ratio);
        }
        say("  Berkowitz density test (whole distribution):");
        for (const NamedBerkowitz& b : bt.berkowitz)
            if (b.model.rfind("overnight", 0) != 0) say(" %s p = %.4f;", b.model.c_str(), b.test.p);
        say("\n");

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
                say("\nOther horizons / books\n  %-20s %6s %5s %5s %7s %17s %9s %8s %7s %8s\n", "model", "conf", "n", "exc",
                    "rate", "95% CI", "Kupiec p", "CC p", "zone", "ES ratio");
                header = true;
            }
            const CoverageTest& c = m.coverage;
            say("  %-20s %6.3f %5d %5d %6.2f%% [%5.2f%%, %5.2f%%] %9.4f %8.4f %7s %8.3f\n", m.model.c_str(), m.confidence,
                c.n, c.exceptions, 100 * c.rate, 100 * c.rate_lo, 100 * c.rate_hi, c.p_uc, c.p_cc, to_string(c.zone),
                m.es_ratio);
        }
        for (const NamedBerkowitz& b : bt.berkowitz)
            if (b.model.rfind("overnight", 0) == 0)
                say("  %s Berkowitz p = %.4f (%d sessions)\n", b.model.c_str(), b.test.p, b.test.n);

        if (!res.daily.empty()) {
            std::vector<double> lam, nus;
            for (const DailyRecord& d : res.daily) {
                lam.push_back(d.lambda);
                nus.push_back(d.nu);
            }
            const DailyRecord& d = res.daily.back();
            say("\nEWMA parameters (%zu daily re-estimations; diurnal profile %s)\n", res.daily.size(), to_string(ec.diurnal));
            say("  lambda (EWMA-N / EWMA-t): last %.4f, range %.4f to %.4f (median %.4f); half-life %.1f bars\n", d.lambda,
                percentile(lam, 0.0), percentile(lam, 1.0), percentile(lam, 0.5), std::log(0.5) / std::log(d.lambda));
            say("  nu (EWMA-t):              last %.2f, range %.2f to %.2f (median %.2f), fitted on %zu standardized outcomes\n",
                d.nu, percentile(nus, 0.0), percentile(nus, 1.0), percentile(nus, 0.5), d.nu_obs);
            if (std::isfinite(d.lambda_on))
                say("  overnight:                lambda %.4f, nu %.2f\n", d.lambda_on, d.nu_on);
            say("  EWMA-N (ref):             lambda %.2f fixed, raw returns, Normal\n", ec.ref_lambda);
        }

        if (res.dq) {
            const IngestResult& g = res.ingest;
            const size_t bars = res.mode_bars[0] + res.mode_bars[1] + res.mode_bars[2];
            say("\nData-quality layer\n");
            say("  feed rows: %zu accepted, %zu duplicates discarded, %zu out of order (accepted), %zu delayed past the bar "
                "close (discarded), %zu off-grid\n",
                g.accepted, g.duplicates, g.out_of_order, g.delayed, g.off_grid);
            say("  cells: %zu GAP, %zu STALE\n", g.gaps, g.stale);
            if (opt.replay.dq_cfg.jump_filter)
                say("  jump filter (z %g, shared with garch_t and realized_caviar): isolated jumps held %zu (bad ticks "
                    "removed %zu, genuine moves accepted one bar late %zu, held at a session's last bar %zu), market moves "
                    "kept %zu (jump_events.csv)\n",
                    opt.replay.dq_cfg.jump.z, res.suspects, res.spikes, res.shocks_kept, res.unresolved, res.market_wide);
            else
                say("  jump filter off: every isolated jump enters the models\n");
            say("  modes: NORMAL %zu bars (%.2f%%), DEGRADED %zu (%.2f%%), FREEZE %zu (%.2f%%); %zu forecasts kept frozen\n",
                res.mode_bars[0], bars ? 100.0 * res.mode_bars[0] / bars : 0.0, res.mode_bars[1],
                bars ? 100.0 * res.mode_bars[1] / bars : 0.0, res.mode_bars[2], bars ? 100.0 * res.mode_bars[2] / bars : 0.0,
                res.frozen_forecasts);
            size_t crit = 0, warn = 0;
            for (const DqAlert& a : res.alerts) {
                crit += a.severity == "critical";
                warn += a.severity == "warning";
            }
            say("  alerts: %zu (%zu critical, %zu warning) -> alerts.csv; events -> dq_events.csv\n", res.alerts.size(), crit,
                warn);
            for (const DqEvent& e : res.events)
                if (e.flag == "SPIKE" || e.flag == "SHOCK_KEPT") say("    %s %-6s %-10s %s\n", e.time.c_str(), e.ticker.c_str(),
                                                                     e.flag.c_str(), e.detail.c_str());
        }
        if (!detection.empty()) {
            say("\nDetection against the clean file (%s)\n", paths::display(opt.truth).c_str());
            say("  %-13s %9s %9s %8s %14s\n", "flag", "injected", "detected", "recall", "other cells");
            for (const DetectionRow& d : detection)
                say("  %-13s %9zu %9zu %8s %14zu\n", d.flag.c_str(), d.injected, d.detected,
                    std::isfinite(d.recall) ? (std::to_string(static_cast<int>(std::lround(100 * d.recall))) + "%").c_str() : "-",
                    d.other);
            say("  (other cells: flags where the clean file has no such fault, e.g. bars that never traded)\n");
        }
        if (have_robust) {
            say("\nRobustness: %s VaR %s of this run vs the same market with a clean feed\n", res.var_models[main].name.c_str(),
                tag(confs[k99]).c_str());
            say("  %zu forecasts compared: mean deviation %.3f%%, max %.2f%% (at %s), %zu bars off by more than 1%%\n", rob_n,
                100 * rob_mean, 100 * rob_max, rob_max_time.c_str(), rob_affected);
            for (const ModelBacktest& m : bt_clean.rows)
                if (m.has_by_time && m.model == res.var_models[main].name)
                    say("  clean-feed backtest %s %.3f: %d exceptions (%.2f%%), zone %s\n", m.model.c_str(), m.confidence,
                        m.coverage.exceptions, 100 * m.coverage.rate, to_string(m.coverage.zone));
        }

        const PortfolioRisk& lr = res.last_risk;
        if (lr.ready) {
            say("\nLast forecast (%s -> next bar), VaR / ES %s:\n", res.last_time.c_str(), tag(confs[k99]).c_str());
            for (size_t m = 0; m < res.var_models.size(); ++m)
                say("  %-14s sigma %10.0f  VaR %10.0f  ES %10.0f\n", res.var_models[m].name.c_str(), lr.models[m].sigma,
                    lr.models[m].measures[k99].var, lr.models[m].measures[k99].es);
        }

        say("\nPerformance (%zu assets)\n", n);
        say("  per-bar update (fills + jump filter + DQ score + models + VaR/ES): p50 %.2f us, p95 %.2f us, p99 %.2f us, "
            "max %.2f us\n",
            percentile(res.latency_us, 0.5), percentile(res.latency_us, 0.95), percentile(res.latency_us, 0.99),
            percentile(res.latency_us, 1.0));
        say("  end-of-day refit (profile + lambda + nu + filters + overnight): p50 %.2f ms, p95 %.2f ms, max %.2f ms\n",
            percentile(res.refit_ms, 0.5), percentile(res.refit_ms, 0.95), percentile(res.refit_ms, 1.0));
        say("  replay: %zu bars, %zu forecasts scored (%zu unscored: target bar never printed; %zu overnight unscored), "
            "%zu fills applied (%zu unknown), engine %.3f s, wall %.3f s\n",
            res.bars, res.forecasts.size(), res.unscored, res.unscored_overnight, res.fills_applied, res.fills_unknown,
            res.engine_seconds, res.wall_seconds);
        say("\noutputs written to %s (report: report.txt)\n", paths::display(opt.out).c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        if (g_report) {
            std::fprintf(g_report, "error: %s\n", e.what());
            std::fclose(g_report);
        }
        return 1;
    }
    if (g_report) std::fclose(g_report);
    return 0;
}
