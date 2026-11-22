#include "garcht/replay.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <stdexcept>
#include <thread>

#include "garcht/revaluation.hpp"

namespace garcht {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

bool valid_price(double v) { return std::isfinite(v) && v > 0.0; }

int weekday_of(int ymd) {  // 0 = Monday
    static const int t[] = {0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4};
    int y = ymd / 10000;
    const int m = (ymd / 100) % 100, d = ymd % 100;
    y -= m < 3;
    return ((y + y / 4 - y / 100 + y / 400 + t[m - 1] + d) % 7 + 6) % 7;
}

int next_ymd(int ymd) {
    static const int dm[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    int y = ymd / 10000, m = (ymd / 100) % 100, d = ymd % 100;
    const bool leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    if (++d > ((m == 2 && leap) ? 29 : dm[m - 1])) {
        d = 1;
        if (++m > 12) {
            m = 1;
            ++y;
        }
    }
    return y * 10000 + m * 100 + d;
}

// Time to expiry in trading years (252 sessions), counting the rest of today's session.
double years_to_expiry(int ymd, int minute, int expiry_ymd, const SessionSpec& s) {
    if (ymd > expiry_ymd) return 0.0;
    const double session_min = static_cast<double>(s.bars_per_day * s.bar_minutes);
    const int close_min = s.open_minute + s.bars_per_day * s.bar_minutes;
    double days = std::clamp((close_min - minute) / session_min, 0.0, 1.0);
    for (int d = next_ymd(ymd); d <= expiry_ymd; d = next_ymd(d))
        if (weekday_of(d) < 5 && !is_b3_holiday(d)) days += 1.0;  // B3 trading days
    return days / 252.0;
}

}  // namespace

// ================================================================== one bar at a time

struct ReplaySession::State {
    struct BookOption {
        OptionContract c;
        int asset;
    };
    // Monte Carlo revaluation of the option book for one forecast, run on a worker thread.
    struct McOut {
        std::vector<RiskMeasure> measures;
        double sigma = 0.0, us = 0.0;
    };

    ReplayConfig cfg;
    SessionSpec session;
    EngineConfig ecfg;
    ReplayResult res;
    size_t n = 0;
    int K = 0;
    std::map<std::string, size_t> tix;
    std::vector<BookOption> book;
    bool has_book = false;
    std::unique_ptr<MonteCarloRevaluation> mc;
    double bar_years = 0.0;
    std::vector<Fill> fills;
    // the VaR models (garcht/models.hpp), main model first, and the window they are refitted on
    std::vector<std::unique_ptr<riskcommon::PortfolioModel>> models;
    GarchPortfolioModel* garch = nullptr;  // models[0]: option book, overnight, decomposition
    IntradayRiskEngine* engine = nullptr;
    riskcommon::HistoryRing history;
    std::vector<riskcommon::ModelForecast> risk;  // forecast of each model at the last bar
    std::vector<double> qty, last_px, expo, xbook;
    std::vector<int> px_date;  // session date of last_px[i]
    std::vector<double> model_px;   // last close of the model's view (jump filter: validated prices)
    std::vector<int> model_px_date;
    std::vector<double> missing_row;
    size_t next_fill = 0;
    PortfolioRisk book_delta_risk;
    int models_fitted = 0;

    // pending forecasts
    bool have_fc = false;
    ForecastRecord fc;
    std::vector<double> fc_qty, fc_spot;
    std::vector<char> fc_live;  // options valued at forecast time
    double fc_option_value = 0.0;
    bool have_on = false;
    int on_date = -1;
    OvernightRecord on;
    std::vector<double> on_qty;

    // Option revaluations (see book_forecast). Each job has an id; mc_record[id] is the index of its
    // forecast in res.forecasts once that forecast is scored (npos: not scored, result discarded).
    // Live: the job in flight runs on a worker thread (mc_job). Offline replay (defer_mc): the jobs
    // are queued and run after the bars.
    static constexpr size_t npos = static_cast<size_t>(-1);
    bool defer_mc = false;
    std::vector<std::function<McOut()>> mc_queue;
    std::vector<size_t> mc_record;
    std::future<McOut> mc_job;
    size_t mc_job_id = npos;  // id of the job in mc_job
    size_t fc_job = npos;     // id of the pending forecast's job

    int cur_date = -1, cur_slot = -1;  // last on-grid bar fed to the engine
    bool eod_done = true;              // end-of-session work of the current session done
    size_t last_t = 0;                 // last on-grid bar stepped
    bool last_timed = false;           // its update time is in latency_us (no refit at that bar)
    double last_net = 0.0;
    std::chrono::steady_clock::time_point t_wall = std::chrono::steady_clock::now();

    static EngineConfig engine_config(const ReplayConfig& cfg, const SessionSpec& s) {
        EngineConfig e = cfg.engine;
        e.bars_per_day = s.bars_per_day;
        e.window_bars = cfg.window_days * s.bars_per_day;
        e.min_fit_bars = std::min(e.window_bars, 20 * s.bars_per_day);
        return e;
    }

    State(const ReplayInput& in, const ReplayConfig& cfg_in, const SessionSpec& s)
        : cfg(cfg_in), session(s), ecfg(engine_config(cfg_in, s)),
          history(static_cast<size_t>(std::max(ecfg.window_bars, 1)), in.tickers.size()) {
        n = in.tickers.size();
        if (n == 0) throw std::runtime_error("replay: no tickers");
        res.session = session;
        res.student_t = ecfg.spec.dist == Dist::StudentT;
        K = session.bars_per_day;

        for (size_t i = 0; i < n; ++i) {
            std::string up = in.tickers[i];
            for (char& c : up) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            tix[up] = i;
        }
        for (const OptionContract& o : in.options) {
            const long a = find_asset(o.underlying);
            if (a < 0) throw std::runtime_error("option " + o.id + ": unknown underlying " + o.underlying);
            book.push_back({o, static_cast<int>(a)});
        }
        has_book = !book.empty();
        if (has_book) mc = std::make_unique<MonteCarloRevaluation>(cfg.mc_scenarios, cfg.mc_seed);
        const double session_min = static_cast<double>(session.bars_per_day * session.bar_minutes);
        bar_years = session.bar_minutes / session_min / 252.0;

        fills = in.fills;
        std::stable_sort(fills.begin(), fills.end(), [](const Fill& a, const Fill& b) { return a.time.key() < b.time.key(); });

        models = make_models(in.tickers, ecfg);
        garch = static_cast<GarchPortfolioModel*>(models.front().get());
        engine = &garch->engine();
        for (const auto& m : models) res.var_models.push_back(m->info());
        res.model_name = res.var_models.front().name;
        risk.resize(models.size());
        qty = in.initial_quantity;
        qty.resize(n, 0.0);
        last_px.assign(n, kNaN);
        expo.assign(n, 0.0);
        xbook.assign(n, 0.0);
        px_date.assign(n, -1);
        model_px.assign(n, kNaN);
        model_px_date.assign(n, -1);
        missing_row.assign(n, kNaN);
        fc_qty.assign(n, 0.0);
        fc_spot.assign(n, kNaN);
        fc_live.assign(book.size(), 0);
        on_qty.assign(n, 0.0);
    }

    ~State() {
        if (mc_job.valid()) mc_job.wait();
    }

    // Writes a revaluation into its forecast record (the pending one or a scored one).
    void store_mc(size_t id, McOut o) {
        ForecastRecord* r = id == fc_job && have_fc ? &fc : (mc_record[id] != npos ? &res.forecasts[mc_record[id]] : nullptr);
        if (!r) return;  // its forecast was never scored
        r->book_mc = std::move(o.measures);
        r->book_mc_sigma = o.sigma;
        r->mc_us = o.us;
    }

    long find_asset(std::string t) const {
        for (char& c : t) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        const auto it = tix.find(t);
        return it == tix.end() ? -1 : static_cast<long>(it->second);
    }

    double option_value(const BookOption& b, double spot, int ymd, int minute) const {
        const double tt = years_to_expiry(ymd, minute, b.c.expiry_ymd, session);
        return b.c.quantity * b.c.multiplier * bs_price(b.c.type, spot, b.c.strike, tt, b.c.implied_vol, b.c.rate);
    }

    // Collects the revaluation in flight on the worker thread (live), if any.
    void join_mc() {
        if (!mc_job.valid()) return;
        store_mc(mc_job_id, mc_job.get());
        mc_job_id = npos;
    }

    // Offline replay: runs the queued revaluations, in bar order, on a worker thread.
    void run_deferred_mc() {
        if (mc_queue.empty()) return;
        std::vector<McOut> out(mc_queue.size());
        std::thread worker([&] {
            for (size_t k = 0; k < mc_queue.size(); ++k) out[k] = mc_queue[k]();
        });
        worker.join();
        for (size_t k = 0; k < out.size(); ++k) store_mc(k, std::move(out[k]));
        mc_queue.clear();
    }

    void feed(int slot, const double* r) {
        for (auto& m : models) m->on_bar(slot, r);
        history.push(slot, r);
    }

    void forecast_all(int next_slot) {
        for (size_t k = 0; k < models.size(); ++k) models[k]->forecast(next_slot, expo.data(), risk[k]);
    }

    // Daily re-estimation at the close of bar t (once the window is full). True if the engine refitted.
    bool refit(const BarPanel& panel, size_t t) {
        if (engine->history_size() < static_cast<size_t>(ecfg.window_bars)) return false;
        const riskcommon::History window = history.view();
        for (auto& m : models) m->refit(window, expo.data());
        const RefitReport& rep = garch->last_refit();
        models_fitted = rep.n_fitted;
        res.refit_ms.push_back(rep.seconds * 1e3);
        const std::string date = panel.timestamps[t].substr(0, 10);
        res.daily.push_back({date, rep.portfolio_nu, rep.dcc_a, rep.dcc_b, rep.seconds * 1e3, rep.n_fitted,
                             rep.n_inactive, rep.dcc_estimated});
        if (cfg.record_params)
            for (int i = 0; i < static_cast<int>(n); ++i) {
                const AssetModel a = engine->asset(i);
                res.params.push_back({date, a.ticker, a.source, a.fit.status, a.active, a.params,
                                      a.fit.robust_std_err, a.fit.loglik, a.fit.iterations});
            }
        return rep.ready;
    }

    // Overnight forecast issued at the close of bar t.
    void overnight_forecast(const BarPanel& panel, size_t t, double net) {
        if (!cfg.overnight) return;
        const OvernightRisk orisk = engine->compute_overnight(expo.data());
        if (!orisk.available) return;
        on = OvernightRecord{};
        on.forecast_time = panel.timestamps[t];
        on.net = net;
        on.risk = orisk;
        on_qty = qty;
        on_date = panel.date[t];
        have_on = true;
    }

    void step(const BarPanel& panel, const std::vector<double>& ret, const std::vector<int>& slot,
              const std::vector<double>& gaps, size_t t, bool eod, const BarPanel& view, double input_us) {
        // Off-grid rows (pre-open prints, an extended closing call, irregular stamps) are
        // ignored everywhere -- returns, gaps and P&L -- so the three stay consistent.
        if (slot[t] < 0) {
            ++res.off_grid;
            return;
        }
        ++res.bars;
        res.last_time = panel.timestamps[t];
        const bool new_session = panel.date[t] != cur_date;
        if (new_session) {
            ++res.sessions;
            eod_done = false;
        }
        // the bar a pending forecast targeted: next slot of the session, or slot 0 of a new one
        const bool contiguous = new_session ? slot[t] == 0 : slot[t] == cur_slot + 1;

        // ---- 1. overnight outcome: previous close -> end of this first bar
        if (have_on && new_session) {
            if (slot[t] == 0) {
                double p = 0.0;
                for (size_t i = 0; i < n; ++i) {
                    const double c = panel.at(panel.close, t, i);
                    if (valid_price(c) && px_date[i] == on_date) p += on_qty[i] * (c - last_px[i]);
                }
                on.pnl = p;
                on.target_time = panel.timestamps[t];
                res.overnight.push_back(on);
            } else {
                ++res.unscored_overnight;  // late opening: the outcome is not the forecast horizon
            }
            have_on = false;
        }

        // ---- 2. intraday outcome of the bar that just closed (overnight gap excluded): the
        // reference is the asset's previous close of this session, else this bar's open
        auto bar_pnl = [&](const BarPanel& p, const std::vector<double>& px, const std::vector<int>& date) {
            double v = 0.0;
            for (size_t i = 0; i < n; ++i) {
                const double c = p.at(p.close, t, i);
                if (!valid_price(c)) continue;
                const double ref = date[i] == p.date[t] ? px[i] : p.at(p.open, t, i);
                if (valid_price(ref)) v += fc_qty[i] * (c - ref);
            }
            return v;
        };
        // the forecast is scored on the prices as printed; the models that learn from outcomes see
        // the outcome of their own view, so a held bad print never reaches the GARCH tail
        const double pnl = bar_pnl(panel, last_px, px_date);
        const double model_pnl = &view == &panel ? pnl : bar_pnl(view, model_px, model_px_date);
        if (have_fc) {
            if (contiguous && fc.target_slot == slot[t]) {
                fc.pnl = pnl;
                fc.target_time = panel.timestamps[t];
                if (has_book) {  // options alive at forecast time, revalued at this close
                    double v1 = 0.0, v0 = new_session ? 0.0 : fc_option_value;
                    for (size_t b = 0; b < book.size(); ++b) {
                        if (!fc_live[b]) continue;
                        const BookOption& bo = book[b];
                        const size_t a = static_cast<size_t>(bo.asset);
                        const double c = panel.at(panel.close, t, a);
                        v1 += option_value(bo, valid_price(c) ? c : fc_spot[a], panel.date[t], panel.minute[t]);
                        if (new_session) {  // reference: the opening price, gap excluded
                            const double o = panel.at(panel.open, t, a);
                            v0 += option_value(bo, valid_price(o) ? o : fc_spot[a], panel.date[t],
                                               panel.minute[t] - session.bar_minutes);
                        }
                    }
                    fc.book_pnl = pnl + (v1 - v0);
                }
                for (size_t k = 0; k < models.size(); ++k) models[k]->observe_outcome(model_pnl, fc.models[k]);
                res.forecasts.push_back(fc);
                if (fc_job != npos) mc_record[fc_job] = res.forecasts.size() - 1;  // its revaluation lands here
            } else {
                ++res.unscored;  // the targeted bar never printed (halt, gap, late opening)
            }
            have_fc = false;
            fc_job = npos;
        }
        for (size_t i = 0; i < n; ++i) {
            const double c = panel.at(panel.close, t, i);
            if (valid_price(c)) {
                last_px[i] = c;
                px_date[i] = panel.date[t];
            }
            const double m = view.at(view.close, t, i);
            if (valid_price(m)) {
                model_px[i] = m;
                model_px_date[i] = view.date[t];
            }
        }

        // ---- 3-5. timed engine work
        const auto t0 = std::chrono::steady_clock::now();
        bool traded = false;
        while (next_fill < fills.size() && fills[next_fill].time.key() <= panel.time_key[t]) {
            const Fill& f = fills[next_fill++];
            const long a = find_asset(f.ticker);
            if (a < 0) {
                ++res.fills_unknown;
                continue;
            }
            qty[static_cast<size_t>(a)] += f.quantity;
            traded = true;
            ++res.fills_applied;
        }
        // bars that never printed are fed as missing, so the filters advance one step per
        // grid slot (halts, gaps, early closes and late openings)
        if (new_session) {
            if (cur_date >= 0)
                for (int k = cur_slot + 1; k < K; ++k) feed(k, missing_row.data());
            if (cfg.overnight) engine->on_overnight(&gaps[t * n]);
            for (int k = 0; k < slot[t]; ++k) feed(k, missing_row.data());
        } else {
            for (int k = cur_slot + 1; k < slot[t]; ++k) feed(k, missing_row.data());
        }
        feed(slot[t], &ret[t * n]);
        cur_date = panel.date[t];
        cur_slot = slot[t];
        last_t = t;

        // positions marked at the model's view: a held print keeps the last validated close
        double net = 0.0, gross = 0.0;
        for (size_t i = 0; i < n; ++i) {
            expo[i] = std::isfinite(model_px[i]) ? qty[i] * model_px[i] : 0.0;
            net += expo[i];
            gross += std::fabs(expo[i]);
        }
        last_net = net;
        bool refitted = false;
        double rebalance_us = 0.0;  // tail re-calibration after a trade, timed apart from the bar update
        if (eod) {
            refitted = refit(panel, t);
        } else if (traded && engine->ready()) {
            const auto r0 = std::chrono::steady_clock::now();
            for (auto& m : models) m->rebalance(expo.data());
            rebalance_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - r0).count();
            res.rebalance_us.push_back(rebalance_us);
        }
        const int next_slot = (slot[t] + 1) % K;
        forecast_all(next_slot);
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() -
                          rebalance_us + input_us;
        res.engine_seconds += (us + rebalance_us) * 1e-6;  // includes the jump filter (input_us)
        last_timed = !refitted && engine->ready();
        if (last_timed) res.latency_us.push_back(us);

        have_fc = garch->ready();
        if (have_fc) {
            fc = ForecastRecord{};
            fc.forecast_time = panel.timestamps[t];
            fc.target_slot = next_slot;
            fc.net = net;
            fc.gross = gross;
            fc.models = risk;
            fc.update_us = us;
            fc.models_fitted = models_fitted;
            fc_qty = qty;
            fc_spot = model_px;
            if (has_book) book_forecast(panel, t, next_slot);
        }
        if (eod) {
            if (have_fc) overnight_forecast(panel, t, net);
            eod_done = true;
        }
    }

    // delta-linear vs Monte Carlo full revaluation of stocks + options, for the forecast of bar t.
    // The VaR of the bar is already published: the simulation runs on a worker thread (it is the
    // heaviest step of a bar and would evict the engine's data from the caches) and its result
    // reaches the forecast record when it is done, before the record is read.
    void book_forecast(const BarPanel& panel, size_t t, int next_slot) {
        join_mc();  // live: one at a time, so the generator advances in bar order (reproducible)
        std::vector<Position> pos;
        for (size_t i = 0; i < n; ++i) {
            xbook[i] = expo[i];
            if (qty[i] != 0.0) {
                Position p;
                p.asset = static_cast<int>(i);
                p.quantity = qty[i];
                pos.push_back(p);
            }
        }
        fc_option_value = 0.0;
        for (size_t b = 0; b < book.size(); ++b) {
            const BookOption& bo = book[b];
            const size_t a = static_cast<size_t>(bo.asset);
            fc_live[b] = 0;
            if (!std::isfinite(model_px[a])) continue;
            Position p;
            p.kind = Position::Kind::Option;
            p.asset = bo.asset;
            p.quantity = bo.c.quantity;
            p.type = bo.c.type;
            p.strike = bo.c.strike;
            p.expiry = years_to_expiry(panel.date[t], panel.minute[t], bo.c.expiry_ymd, session);
            p.vol = bo.c.implied_vol;
            p.rate = bo.c.rate;
            p.multiplier = bo.c.multiplier;
            if (p.expiry <= 0.0) continue;  // expired: no risk left
            fc_live[b] = 1;
            xbook[a] += position_delta_exposure(p, model_px[a]);
            fc_option_value += position_value(p, model_px[a]);
            pos.push_back(p);
        }
        // delta-linear book with the same multivariate t(nu_p) the simulation draws from (the
        // GPD tail is a portfolio-level object the joint simulation does not reproduce), so
        // MC vs delta-linear isolates the non-linearity (gamma) of the options
        engine->compute(next_slot, xbook.data(), book_delta_risk);
        fc.book_delta.clear();
        for (double cl : ecfg.confidences)
            fc.book_delta.push_back(risk_measure(
                book_delta_risk.mean, book_delta_risk.sigma,
                tail_factors(std::isfinite(book_delta_risk.nu) ? Dist::StudentT : Dist::Normal,
                             book_delta_risk.nu, cl)));
        fc.book_delta_sigma = book_delta_risk.sigma;
        std::vector<double> spot(n);
        for (size_t i = 0; i < n; ++i) spot[i] = std::isfinite(model_px[i]) ? model_px[i] : 0.0;  // as expo
        auto job = [mcr = mc.get(), jf = engine->joint_forecast(next_slot), pos = std::move(pos), spot = std::move(spot),
                    years = bar_years, confs = ecfg.confidences]() {
            const auto m0 = std::chrono::steady_clock::now();
            mcr->simulate(jf);
            const McRiskResult mr = mcr->evaluate(pos, spot, years, confs);
            McOut o;
            o.measures = mr.measures;
            o.sigma = mr.sigma;
            o.us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - m0).count();
            return o;
        };
        fc_job = mc_record.size();
        mc_record.push_back(npos);
        if (!cfg.mc_async) {
            store_mc(fc_job, job());  // on the bar thread, right after the VaR (the original design)
        } else if (defer_mc) {
            mc_queue.push_back(std::move(job));
        } else {
            mc_job = std::async(std::launch::async, std::move(job));
            mc_job_id = fc_job;
        }
    }
};

ReplaySession::ReplaySession(const ReplayInput& in, const ReplayConfig& cfg, const SessionSpec& session)
    : s_(std::make_unique<State>(in, cfg, session)) {}

ReplaySession::~ReplaySession() = default;

void ReplaySession::step(const BarPanel& panel, const std::vector<double>& ret, const std::vector<int>& slot,
                         const std::vector<double>& gaps, size_t t, bool eod, const BarPanel* model_view,
                         double input_us) {
    s_->step(panel, ret, slot, gaps, t, eod, model_view ? *model_view : panel, input_us);
}

void ReplaySession::end_session(const BarPanel& panel) {
    State& s = *s_;
    if (s.cur_date < 0 || s.eod_done) return;
    // as if eod had been known at the bar: a refit bar is not a latency sample, and the state
    // after it is the post-refit risk (the forecast issued there targets a bar that never prints)
    const bool refitted = s.refit(panel, s.last_t);
    if (refitted) {
        if (s.last_timed) s.res.latency_us.pop_back();
        s.last_timed = false;
        s.forecast_all((s.cur_slot + 1) % s.K);
    }
    if (s.engine->ready()) s.overnight_forecast(panel, s.last_t, s.last_net);
    s.eod_done = true;
}

bool ReplaySession::session_open() const { return s_->cur_date >= 0 && !s_->eod_done; }

const ForecastRecord* ReplaySession::issued() {
    s_->join_mc();
    return s_->have_fc ? &s_->fc : nullptr;
}

void ReplaySession::defer_revaluations() { s_->defer_mc = true; }

ReplayResult ReplaySession::finish() {
    State& s = *s_;
    s.join_mc();
    s.run_deferred_mc();
    ReplayResult res = s.res;
    res.quantity = s.qty;
    res.last_price = s.last_px;
    res.exposure = s.expo;
    res.last_risk = s.garch->last_risk();
    res.correlation = s.engine->correlation();
    res.diurnal = s.engine->diurnal().factors();
    res.corr_scale = s.engine->slot_correlation_scale();
    for (int i = 0; i < static_cast<int>(s.n); ++i) res.models.push_back(s.engine->asset(i));
    res.wall_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - s.t_wall).count();
    return res;
}

// ================================================================== whole panel

namespace {

// Last on-grid bar of each session, from the data (a whole panel is known in advance).
std::vector<char> end_of_session_flags(const BarPanel& panel, const std::vector<int>& slot) {
    const size_t T = panel.num_bars();
    std::vector<char> eod(T, 0);
    for (size_t t = T; t-- > 0;) {
        if (slot[t] < 0) continue;
        size_t u = t + 1;
        while (u < T && slot[u] < 0 && panel.date[u] == panel.date[t]) ++u;
        eod[t] = (u >= T || panel.date[u] != panel.date[t]);
    }
    return eod;
}

// The models' input of each bar, prepared inside the bar's own cycle: the jump filter's decision on
// the bar (if on) and the returns and overnight gaps of the models' view. Row t needs rows <= t only,
// so a file and a live feed are prepared the same way, one bar at a time.
class BarInput {
public:
    BarInput(size_t n, const SessionSpec& session, const ReplayConfig& cfg)
        : n_(n), session_(session), filter_on_(cfg.jump_filter), filter_(n, session, cfg.jump), stream_(n) {}

    // Appends row t of `panel` (rows < t already prepared) to the model's view; untimed (in a live
    // system the feed handler writes the row).
    void add_row(const BarPanel& panel, size_t t) {
        slot_.push_back(bar_slot(panel.minute[t], session_));
        ret_.resize((t + 1) * n_, kNaN);
        gaps_.resize((t + 1) * n_, kNaN);
        if (!filter_on_) return;
        if (view_.tickers.empty()) view_.tickers = panel.tickers;
        view_.timestamps.push_back(panel.timestamps[t]);
        view_.time_key.push_back(panel.time_key[t]);
        view_.date.push_back(panel.date[t]);
        view_.minute.push_back(panel.minute[t]);
        for (auto [dst, src] : {std::pair{&view_.open, &panel.open}, std::pair{&view_.high, &panel.high},
                                std::pair{&view_.low, &panel.low}, std::pair{&view_.close, &panel.close},
                                std::pair{&view_.volume, &panel.volume}})
            dst->insert(dst->end(), src->begin() + static_cast<long>(t * n_), src->begin() + static_cast<long>((t + 1) * n_));
    }
    // Decides row t (the last one added); returns the time it took, in microseconds.
    double prepare(const BarPanel& panel, size_t t) {
        const auto t0 = std::chrono::steady_clock::now();
        if (slot_[t] >= 0) {
            if (filter_on_) filter_.apply(view_, t);
            const BarPanel& v = view(panel);
            stream_.step(v.date[t], slot_[t], &v.open[t * n_], &v.close[t * n_], &ret_[t * n_], &gaps_[t * n_]);
            if (filter_on_) {  // a move let in one bar late enters at full size (see restore_accepted)
                const auto& acc = filter_.accepted();
                for (; accepted_done_ < acc.size(); ++accepted_done_) {
                    double& r = ret_[acc[accepted_done_].bar * n_ + acc[accepted_done_].asset];
                    if (std::isfinite(r)) r = acc[accepted_done_].ret;
                }
            }
        }
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    }
    // The session of the last row prepared is over (end-of-day work, untimed).
    void end_session() {
        if (filter_on_) filter_.end_session();
    }
    const BarPanel& view(const BarPanel& panel) const { return filter_on_ ? view_ : panel; }
    const std::vector<double>& ret() const { return ret_; }
    const std::vector<double>& gaps() const { return gaps_; }
    const std::vector<int>& slot() const { return slot_; }
    std::vector<JumpEvent> events() const { return filter_on_ ? filter_.events() : std::vector<JumpEvent>{}; }

private:
    size_t n_;
    SessionSpec session_;
    bool filter_on_;
    JumpFilter filter_;
    ReturnStream stream_;
    size_t accepted_done_ = 0;
    BarPanel view_;  // the jump-filtered copy of the prices (filter on)
    std::vector<double> ret_, gaps_;
    std::vector<int> slot_;
};

}  // namespace

ReplayResult run_replay(const ReplayInput& in, const ReplayConfig& cfg_in) {
    const auto t_wall = std::chrono::steady_clock::now();
    const size_t n = in.tickers.size(), T = in.bars.num_bars();
    if (n == 0 || in.bars.num_assets() != n) throw std::runtime_error("replay: tickers / bars mismatch");

    const SessionSpec session = cfg_in.detect_session ? detect_session(in.bars) : cfg_in.session;
    // The jump filter shapes the model's returns only: the replay steps the prices as printed, so the
    // P&L the forecasts are scored on is the market's.
    const BarPanel& panel = in.bars;
    std::vector<int> slot(T);
    for (size_t t = 0; t < T; ++t) slot[t] = bar_slot(panel.minute[t], session);
    size_t sessions = 0;
    {
        int last_date = -1;
        for (size_t t = 0; t < T; ++t)
            if (slot[t] >= 0 && panel.date[t] != last_date) {
                ++sessions;
                last_date = panel.date[t];
            }
    }
    if (static_cast<int>(sessions) <= cfg_in.window_days)
        throw std::runtime_error("the data has " + std::to_string(sessions) + " sessions; a window of " +
                                 std::to_string(cfg_in.window_days) +
                                 " days needs at least one more session to produce forecasts");
    const std::vector<char> eod = end_of_session_flags(panel, slot);

    ReplaySession replay(in, cfg_in, session);
    replay.defer_revaluations();
    BarInput input(n, session, cfg_in);
    for (size_t t = 0; t < T; ++t) {
        input.add_row(panel, t);
        const double input_us = input.prepare(panel, t);  // counted in the bar's latency
        if (eod[t]) input.end_session();
        replay.step(panel, input.ret(), input.slot(), input.gaps(), t, eod[t] != 0, &input.view(panel), input_us);
    }
    ReplayResult res = replay.finish();
    res.jump_events = input.events();
    res.wall_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_wall).count();
    return res;
}

// ================================================================== live feed

std::vector<std::string> split_feed_line(const std::string& line, char& delimiter) {
    const CsvTable t = parse_csv(line, delimiter);
    delimiter = t.delimiter;
    return t.header;  // the line's only record (empty for a blank line)
}

namespace {

std::string upper_trim(std::string s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
    size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    s = s.substr(b);
    for (char& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

}  // namespace

ReplayResult run_live(const ReplayInput& history, const ReplayConfig& cfg,
                      const std::function<bool(std::string&)>& next_line, LiveStats& st,
                      const std::function<void(const BarPanel&, size_t, const ForecastRecord*)>& on_bar,
                      int utc_offset) {
    const size_t n = history.tickers.size();
    if (n == 0 || history.bars.num_assets() != n) throw std::runtime_error("live: tickers / history mismatch");
    BarPanel panel = history.bars;  // prices as printed; grows by one row per bar
    const SessionSpec session = cfg.detect_session ? detect_session(panel) : cfg.session;
    ReplaySession replay(history, cfg, session);
    BarInput input(n, session, cfg);  // the model's view, prepared bar by bar as in run_replay
    int grid_date = -1;  // date of the last on-grid bar stepped
    {   // the history, as run_replay would (its last bar ends its session)
        std::vector<int> slot(panel.num_bars());
        for (size_t t = 0; t < panel.num_bars(); ++t) slot[t] = bar_slot(panel.minute[t], session);
        const std::vector<char> eod = end_of_session_flags(panel, slot);
        for (size_t t = 0; t < panel.num_bars(); ++t) {
            input.add_row(panel, t);
            const double input_us = input.prepare(panel, t);
            if (eod[t]) input.end_session();
            replay.step(panel, input.ret(), input.slot(), input.gaps(), t, eod[t] != 0, &input.view(panel), input_us);
            if (slot[t] >= 0) grid_date = panel.date[t];
        }
    }

    std::map<std::string, size_t> tix;
    for (size_t i = 0; i < n; ++i) tix[upper_trim(history.tickers[i])] = i;
    struct PendingBar {
        TimePoint time;
        std::vector<double> open, high, low, close, volume;
        std::vector<char> seen;
    };
    std::map<long long, PendingBar> pending;  // bars not closed yet, by close key
    long long stepped = panel.num_bars() ? panel.time_key.back() : std::numeric_limits<long long>::min();
    const long long history_end = stepped;
    int c_ts = -1, c_tk = -1, c_o = -1, c_h = -1, c_l = -1, c_c = -1, c_v = -1;
    bool have_header = false;
    char delimiter = 0;  // detected on the header line, as the file reader does

    // Appends the pending bars closing at or before `until` to the panel and steps them.
    auto close_bars = [&](long long until) {
        while (!pending.empty() && pending.begin()->first <= until) {
            const long long key = pending.begin()->first;
            PendingBar b = std::move(pending.begin()->second);
            pending.erase(pending.begin());
            panel.timestamps.push_back(b.time.str());
            panel.time_key.push_back(key);
            panel.date.push_back(b.time.ymd);
            panel.minute.push_back(b.time.minute);
            for (size_t i = 0; i < n; ++i) {
                panel.open.push_back(b.open[i]);
                panel.high.push_back(b.high[i]);
                panel.low.push_back(b.low[i]);
                panel.close.push_back(b.close[i]);
                panel.volume.push_back(b.volume[i]);
            }
            stepped = key;
            ++st.bars;
            const size_t t = panel.num_bars() - 1;
            input.add_row(panel, t);
            const double input_us = input.prepare(panel, t);
            const std::vector<int>& slot = input.slot();
            if (slot[t] < 0) {
                replay.step(panel, input.ret(), slot, input.gaps(), t, false, &input.view(panel));  // off-grid: ignored
                continue;
            }
            // a new session started while the previous one had not reached its last slot
            if (replay.session_open() && panel.date[t] != grid_date) replay.end_session(panel);
            const bool eod = slot[t] == session.bars_per_day - 1;
            if (eod) input.end_session();  // with the end-of-day work
            replay.step(panel, input.ret(), slot, input.gaps(), t, eod, &input.view(panel), input_us);
            grid_date = panel.date[t];
            if (on_bar) on_bar(panel, t, replay.issued());
        }
    };

    std::string line;
    while (next_line(line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (line.empty()) continue;
        if (line[0] == '#') {
            if (line.rfind("#end", 0) == 0) break;
            if (line.rfind("#clock ", 0) == 0) {
                TimePoint tp;
                if (parse_time(line.substr(7), tp, utc_offset)) {
                    tp.millis = 0;
                    close_bars(tp.key());
                }
            }
            continue;
        }
        const std::vector<std::string> f = split_feed_line(line, delimiter);
        if (f.empty()) continue;
        if (!have_header) {
            for (size_t k = 0; k < f.size(); ++k) {
                const std::string h = upper_trim(f[k]);
                const int c = static_cast<int>(k);
                if (h == "TIMESTAMP" || h == "DATETIME") c_ts = c;
                else if (h == "TICKER" || h == "SYMBOL" || h == "ASSET") c_tk = c;
                else if (h == "OPEN") c_o = c;
                else if (h == "HIGH") c_h = c;
                else if (h == "LOW") c_l = c;
                else if (h == "CLOSE") c_c = c;
                else if (h == "VOLUME") c_v = c;
            }
            if (c_ts < 0 || c_tk < 0 || c_c < 0) throw std::runtime_error("live feed: the header needs timestamp, ticker and close");
            have_header = true;
            continue;
        }
        ++st.rows;
        auto field = [&](int c) { return c >= 0 && static_cast<size_t>(c) < f.size() ? f[static_cast<size_t>(c)] : std::string(); };
        TimePoint tp;
        double o = kNaN, h = kNaN, l = kNaN, c = kNaN, v = kNaN;
        if (!parse_time(field(c_ts), tp, utc_offset) || !parse_number(field(c_c), c) || !parse_number(field(c_o), o) ||
            !parse_number(field(c_h), h) || !parse_number(field(c_l), l) || !parse_number(field(c_v), v)) {
            ++st.bad;
            continue;
        }
        const auto it = tix.find(upper_trim(field(c_tk)));
        if (it == tix.end()) {
            ++st.unknown;
            continue;
        }
        tp.millis = 0;  // bars are identified by their closing minute
        const long long key = tp.key();
        if (key <= history_end) {
            ++st.before_history;
            continue;
        }
        if (key <= stepped) {  // its bar was already closed by the feed's clock
            ++st.late;
            continue;
        }
        PendingBar& b = pending[key];
        if (b.seen.empty()) {
            b.time = tp;
            for (auto* m : {&b.open, &b.high, &b.low, &b.close, &b.volume}) m->assign(n, kNaN);
            b.seen.assign(n, 0);
        }
        const size_t i = it->second;
        if (b.seen[i]) {
            ++st.duplicates;
            continue;
        }
        b.seen[i] = 1;
        b.open[i] = std::isfinite(o) ? o : c;  // missing open -> close, as build_panel
        b.high[i] = h;
        b.low[i] = l;
        b.close[i] = c;
        b.volume[i] = v;
    }
    close_bars(std::numeric_limits<long long>::max());
    replay.end_session(panel);  // the last bar ends its session, as at the end of a file
    ReplayResult res = replay.finish();
    res.jump_events = input.events();
    return res;
}


BacktestReport evaluate_backtest(const ReplayResult& r, const std::vector<double>& confs, int es_sims) {
    BacktestReport rep;
    const int bpd = r.session.bars_per_day;
    auto group = [&](int slot) { return slot < 3 ? 0 : (slot >= bpd - 3 ? 2 : 1); };

    // intraday models, in the order of r.var_models
    std::vector<double> x;
    std::vector<int> slots;
    std::vector<const ForecastRecord*> fr;
    for (const ForecastRecord& f : r.forecasts) {
        if (!std::isfinite(f.pnl)) continue;
        x.push_back(f.pnl);
        slots.push_back(f.target_slot);
        fr.push_back(&f);
    }
    for (size_t m = 0; m < r.var_models.size(); ++m) {
        std::vector<DistForecast> dist;
        std::vector<double> pit;
        for (size_t t = 0; t < fr.size(); ++t) {
            const riskcommon::ModelForecast& f = fr[t]->models[m];
            dist.emplace_back(f.mean, f.sigma, ShockDist(f.shape));
            pit.push_back(forecast_cdf(dist.back(), x[t]));
        }
        rep.berkowitz.push_back({r.var_models[m].name, berkowitz_test(pit)});
        for (size_t k = 0; k < confs.size(); ++k) {
            ModelBacktest mb;
            mb.model = r.var_models[m].name;
            mb.confidence = confs[k];
            std::vector<int> hits, seg_hits[3];
            double loss_sum = 0.0, es_sum = 0.0;
            for (size_t t = 0; t < fr.size(); ++t) {
                const RiskMeasure& rm = fr[t]->models[m].measures[k];
                const int h = -x[t] > rm.var;
                hits.push_back(h);
                seg_hits[group(slots[t])].push_back(h);
                if (h) {
                    loss_sum += -x[t];
                    es_sum += rm.es;
                }
            }
            mb.coverage = coverage_test(hits, 1.0 - confs[k]);
            for (int g = 0; g < 3; ++g) mb.by_time[g] = coverage_test(seg_hits[g], 1.0 - confs[k]);
            mb.has_by_time = true;
            mb.es_ratio = es_sum > 0.0 ? loss_sum / es_sum : kNaN;
            mb.es = acerbi_szekely_test(x, dist, confs[k], es_sims, 1000 + k);
            mb.has_es = true;
            rep.rows.push_back(mb);
        }
    }

    // overnight (close -> end of the next first bar)
    if (!r.overnight.empty()) {
        std::vector<double> xo, pit_o;
        std::vector<DistForecast> fo;
        for (const OvernightRecord& o : r.overnight) {
            if (!std::isfinite(o.pnl) || !std::isfinite(o.risk.sigma)) continue;
            xo.push_back(o.pnl);
            fo.emplace_back(o.risk.mean, o.risk.sigma, o.risk.nu);
            pit_o.push_back(forecast_cdf(fo.back(), o.pnl));
        }
        rep.berkowitz_overnight = berkowitz_test(pit_o);
        for (size_t k = 0; k < confs.size(); ++k) {
            ModelBacktest mb;
            mb.model = "overnight";
            mb.confidence = confs[k];
            std::vector<int> hits;
            double loss_sum = 0.0, es_sum = 0.0;
            for (const OvernightRecord& o : r.overnight) {
                if (!std::isfinite(o.pnl) || k >= o.risk.measures.size()) continue;
                const int h = -o.pnl > o.risk.measures[k].var;
                hits.push_back(h);
                if (h) {
                    loss_sum += -o.pnl;
                    es_sum += o.risk.measures[k].es;
                }
            }
            mb.coverage = coverage_test(hits, 1.0 - confs[k]);
            mb.es_ratio = es_sum > 0.0 ? loss_sum / es_sum : kNaN;
            rep.rows.push_back(mb);
        }
    }

    // option book: Monte Carlo full revaluation vs delta-linear
    const bool has_book = !r.forecasts.empty() && !r.forecasts.front().book_mc.empty();
    if (has_book)
        for (int model = 0; model < 2; ++model)
            for (size_t k = 0; k < confs.size(); ++k) {
                ModelBacktest mb;
                mb.model = model == 0 ? "book MC full reval" : "book delta-linear";
                mb.confidence = confs[k];
                std::vector<int> hits;
                double loss_sum = 0.0, es_sum = 0.0;
                for (const ForecastRecord& f : r.forecasts) {
                    if (!std::isfinite(f.book_pnl)) continue;
                    const std::vector<RiskMeasure>& v = model == 0 ? f.book_mc : f.book_delta;
                    if (k >= v.size()) continue;
                    const int h = -f.book_pnl > v[k].var;
                    hits.push_back(h);
                    if (h) {
                        loss_sum += -f.book_pnl;
                        es_sum += v[k].es;
                    }
                }
                mb.coverage = coverage_test(hits, 1.0 - confs[k]);
                mb.es_ratio = es_sum > 0.0 ? loss_sum / es_sum : kNaN;
                rep.rows.push_back(mb);
            }
    return rep;
}

}  // namespace garcht
