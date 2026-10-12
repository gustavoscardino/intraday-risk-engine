#include "garcht/replay.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>

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

ReplayResult run_replay(const ReplayInput& in, const ReplayConfig& cfg_in) {
    const auto t_wall = std::chrono::steady_clock::now();
    ReplayResult res;
    const BarPanel& panel = in.bars;
    const size_t n = in.tickers.size(), T = panel.num_bars();
    if (n == 0 || panel.num_assets() != n) throw std::runtime_error("replay: tickers / bars mismatch");

    const SessionSpec session = cfg_in.detect_session ? detect_session(panel) : cfg_in.session;
    res.session = session;
    EngineConfig ecfg = cfg_in.engine;
    ecfg.bars_per_day = session.bars_per_day;
    ecfg.window_bars = cfg_in.window_days * session.bars_per_day;
    ecfg.min_fit_bars = std::min(ecfg.window_bars, 20 * session.bars_per_day);
    res.student_t = ecfg.spec.dist == Dist::StudentT;
    res.model_name = !res.student_t ? "GARCH-N" : (ecfg.tail_model == TailModel::Evt ? "GARCH-t/EVT" : "GARCH-t");

    std::vector<double> ret, gaps;
    std::vector<int> slot;
    bar_returns(panel, session, ret, slot);
    overnight_returns(panel, slot, gaps);
    {
        int last_date = -1;
        for (size_t t = 0; t < T; ++t)
            if (slot[t] >= 0 && panel.date[t] != last_date) {
                ++res.sessions;
                last_date = panel.date[t];
            }
    }
    if (static_cast<int>(res.sessions) <= cfg_in.window_days)
        throw std::runtime_error("the data has " + std::to_string(res.sessions) + " sessions; a window of " +
                                 std::to_string(cfg_in.window_days) +
                                 " days needs at least one more session to produce forecasts");
    // last on-grid bar of each session
    std::vector<char> eod(T, 0);
    for (size_t t = T; t-- > 0;) {
        if (slot[t] < 0) continue;
        size_t u = t + 1;
        while (u < T && slot[u] < 0 && panel.date[u] == panel.date[t]) ++u;
        eod[t] = (u >= T || panel.date[u] != panel.date[t]);
    }

    std::map<std::string, size_t> tix;
    for (size_t i = 0; i < n; ++i) {
        std::string up = in.tickers[i];
        for (char& c : up) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        tix[up] = i;
    }
    auto find_asset = [&](std::string t) -> long {
        for (char& c : t) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        const auto it = tix.find(t);
        return it == tix.end() ? -1 : static_cast<long>(it->second);
    };

    // option book
    struct BookOption {
        OptionContract c;
        int asset;
    };
    std::vector<BookOption> book;
    for (const OptionContract& o : in.options) {
        const long a = find_asset(o.underlying);
        if (a < 0) throw std::runtime_error("option " + o.id + ": unknown underlying " + o.underlying);
        book.push_back({o, static_cast<int>(a)});
    }
    const bool has_book = !book.empty();
    std::unique_ptr<MonteCarloRevaluation> mc;
    if (has_book) mc = std::make_unique<MonteCarloRevaluation>(cfg_in.mc_scenarios, cfg_in.mc_seed);
    const double session_min = static_cast<double>(session.bars_per_day * session.bar_minutes);
    const double bar_years = session.bar_minutes / session_min / 252.0;
    auto option_value = [&](const BookOption& b, double spot, int ymd, int minute) {
        const double tt = years_to_expiry(ymd, minute, b.c.expiry_ymd, session);
        return b.c.quantity * b.c.multiplier * bs_price(b.c.type, spot, b.c.strike, tt, b.c.implied_vol, b.c.rate);
    };

    std::vector<Fill> fills = in.fills;
    std::stable_sort(fills.begin(), fills.end(), [](const Fill& a, const Fill& b) { return a.time.key() < b.time.key(); });

    IntradayRiskEngine engine(in.tickers, ecfg);
    std::vector<double> qty = in.initial_quantity;
    qty.resize(n, 0.0);
    std::vector<double> last_px(n, kNaN), expo(n, 0.0), xbook(n, 0.0);
    std::vector<int> px_date(n, -1);  // session date of last_px[i]
    const std::vector<double> missing_row(n, kNaN);
    size_t next_fill = 0;
    PortfolioRisk risk, book_delta_risk;
    int models_fitted = 0;

    // pending forecasts
    bool have_fc = false;
    ForecastRecord fc;
    PortfolioRisk fc_risk;
    std::vector<double> fc_qty(n, 0.0), fc_spot(n, kNaN);
    std::vector<char> fc_live(book.size(), 0);  // options valued at forecast time
    double fc_option_value = 0.0;
    bool have_on = false;
    int on_date = -1;
    OvernightRecord on;
    std::vector<double> on_qty(n, 0.0);

    int cur_date = -1, cur_slot = -1;  // last on-grid bar fed to the engine
    const int K = session.bars_per_day;

    for (size_t t = 0; t < T; ++t) {
        // Off-grid rows (pre-open prints, an extended closing call, irregular stamps) are
        // ignored everywhere -- returns, gaps and P&L -- so the three stay consistent.
        if (slot[t] < 0) {
            ++res.off_grid;
            continue;
        }
        ++res.bars;
        res.last_time = panel.timestamps[t];
        const bool new_session = panel.date[t] != cur_date;
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
        double pnl = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double c = panel.at(panel.close, t, i);
            if (!valid_price(c)) continue;
            const double ref = px_date[i] == panel.date[t] ? last_px[i] : panel.at(panel.open, t, i);
            if (valid_price(ref)) pnl += fc_qty[i] * (c - ref);
        }
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
                engine.observe_outcome(pnl, fc_risk);
                res.forecasts.push_back(fc);
            } else {
                ++res.unscored;  // the targeted bar never printed (halt, gap, late opening)
            }
            have_fc = false;
        }
        for (size_t i = 0; i < n; ++i) {
            const double c = panel.at(panel.close, t, i);
            if (valid_price(c)) {
                last_px[i] = c;
                px_date[i] = panel.date[t];
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
                for (int k = cur_slot + 1; k < K; ++k) engine.on_bar(k, missing_row.data());
            if (cfg_in.overnight) engine.on_overnight(&gaps[t * n]);
            for (int k = 0; k < slot[t]; ++k) engine.on_bar(k, missing_row.data());
        } else {
            for (int k = cur_slot + 1; k < slot[t]; ++k) engine.on_bar(k, missing_row.data());
        }
        engine.on_bar(slot[t], &ret[t * n]);
        cur_date = panel.date[t];
        cur_slot = slot[t];

        double net = 0.0, gross = 0.0;
        for (size_t i = 0; i < n; ++i) {
            expo[i] = std::isfinite(last_px[i]) ? qty[i] * last_px[i] : 0.0;
            net += expo[i];
            gross += std::fabs(expo[i]);
        }
        bool refitted = false;
        if (eod[t] && engine.history_size() >= static_cast<size_t>(ecfg.window_bars)) {
            const RefitReport rep = engine.refit(expo.data());
            refitted = rep.ready;
            models_fitted = rep.n_fitted;
            res.refit_ms.push_back(rep.seconds * 1e3);
            const std::string date = panel.timestamps[t].substr(0, 10);
            res.daily.push_back({date, rep.portfolio_nu, rep.dcc_a, rep.dcc_b, rep.seconds * 1e3, rep.n_fitted,
                                 rep.n_inactive, rep.dcc_estimated});
            if (cfg_in.record_params)
                for (int i = 0; i < static_cast<int>(n); ++i) {
                    const AssetModel a = engine.asset(i);
                    res.params.push_back({date, a.ticker, a.source, a.fit.status, a.active, a.params,
                                          a.fit.robust_std_err, a.fit.loglik, a.fit.iterations});
                }
        } else if (traded && engine.ready()) {
            engine.recalibrate_tail(expo.data());
        }
        const int next_slot = (slot[t] + 1) % K;
        engine.compute(next_slot, expo.data(), risk);
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        res.engine_seconds += us * 1e-6;
        if (!refitted && engine.ready()) res.latency_us.push_back(us);

        have_fc = engine.ready();
        if (!have_fc) continue;
        fc = ForecastRecord{};
        fc.forecast_time = panel.timestamps[t];
        fc.target_slot = next_slot;
        fc.net = net;
        fc.gross = gross;
        fc.mean = risk.mean;
        fc.sigma = risk.sigma;
        fc.nu = risk.nu;
        fc.tail = risk.tail;
        fc.ewma_sigma = risk.ewma_sigma;
        fc.garch = risk.garch;
        fc.ewma = risk.ewma;
        fc.update_us = us;
        fc.models_fitted = models_fitted;
        fc_risk = risk;
        fc_qty = qty;
        fc_spot = last_px;

        if (has_book) {  // delta-linear vs Monte Carlo full revaluation of stocks + options
            const auto m0 = std::chrono::steady_clock::now();
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
                if (!std::isfinite(last_px[a])) continue;
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
                xbook[a] += position_delta_exposure(p, last_px[a]);
                fc_option_value += position_value(p, last_px[a]);
                pos.push_back(p);
            }
            // delta-linear book with the same multivariate t(nu_p) the simulation draws from (the
            // GPD tail is a portfolio-level object the joint simulation does not reproduce), so
            // MC vs delta-linear isolates the non-linearity (gamma) of the options
            engine.compute(next_slot, xbook.data(), book_delta_risk);
            fc.book_delta.clear();
            for (double cl : ecfg.confidences)
                fc.book_delta.push_back(risk_measure(
                    book_delta_risk.mean, book_delta_risk.sigma,
                    tail_factors(std::isfinite(book_delta_risk.nu) ? Dist::StudentT : Dist::Normal,
                                 book_delta_risk.nu, cl)));
            fc.book_delta_sigma = book_delta_risk.sigma;
            std::vector<double> spot(n);
            for (size_t i = 0; i < n; ++i) spot[i] = std::isfinite(last_px[i]) ? last_px[i] : 0.0;  // as expo
            mc->simulate(engine.joint_forecast(next_slot));
            const McRiskResult mr = mc->evaluate(pos, spot, bar_years, ecfg.confidences);
            fc.book_mc = mr.measures;
            fc.book_mc_sigma = mr.sigma;
            fc.mc_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - m0).count();
        }

        if (eod[t] && cfg_in.overnight) {
            const OvernightRisk orisk = engine.compute_overnight(expo.data());
            if (orisk.available) {
                on = OvernightRecord{};
                on.forecast_time = panel.timestamps[t];
                on.net = net;
                on.risk = orisk;
                on_qty = qty;
                on_date = panel.date[t];
                have_on = true;
            }
        }
    }

    res.quantity = qty;
    res.last_price = last_px;
    res.exposure = expo;
    res.last_risk = risk;
    res.correlation = engine.correlation();
    res.diurnal = engine.diurnal().factors();
    res.corr_scale = engine.slot_correlation_scale();
    for (int i = 0; i < static_cast<int>(n); ++i) res.models.push_back(engine.asset(i));
    res.wall_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_wall).count();
    return res;
}

BacktestReport evaluate_backtest(const ReplayResult& r, const std::vector<double>& confs, int es_sims) {
    BacktestReport rep;
    const int bpd = r.session.bars_per_day;
    auto group = [&](int slot) { return slot < 3 ? 0 : (slot >= bpd - 3 ? 2 : 1); };

    // intraday models
    std::vector<double> x;
    std::vector<int> slots;
    std::vector<DistForecast> fg, fe;
    for (const ForecastRecord& f : r.forecasts) {
        if (!std::isfinite(f.pnl)) continue;
        x.push_back(f.pnl);
        slots.push_back(f.target_slot);
        fg.emplace_back(f.mean, f.sigma, f.tail);
        fe.emplace_back(0.0, f.ewma_sigma, std::numeric_limits<double>::infinity());
    }
    std::vector<double> pit_g(x.size()), pit_e(x.size());
    for (size_t t = 0; t < x.size(); ++t) {
        pit_g[t] = forecast_cdf(fg[t], x[t]);
        pit_e[t] = forecast_cdf(fe[t], x[t]);
    }
    rep.berkowitz_garch = berkowitz_test(pit_g);
    rep.berkowitz_ewma = berkowitz_test(pit_e);

    for (int model = 0; model < 2; ++model)
        for (size_t k = 0; k < confs.size(); ++k) {
            ModelBacktest mb;
            mb.model = model == 0 ? r.model_name : "EWMA-N (ref)";
            mb.confidence = confs[k];
            std::vector<int> hits, seg_hits[3];
            double loss_sum = 0.0, es_sum = 0.0;
            size_t j = 0;
            for (const ForecastRecord& f : r.forecasts) {
                if (!std::isfinite(f.pnl)) continue;
                const RiskMeasure& m = model == 0 ? f.garch[k] : f.ewma[k];
                const int h = -f.pnl > m.var;
                hits.push_back(h);
                seg_hits[group(slots[j])].push_back(h);
                if (h) {
                    loss_sum += -f.pnl;
                    es_sum += m.es;
                }
                ++j;
            }
            mb.coverage = coverage_test(hits, 1.0 - confs[k]);
            for (int g = 0; g < 3; ++g) mb.by_time[g] = coverage_test(seg_hits[g], 1.0 - confs[k]);
            mb.has_by_time = true;
            mb.es_ratio = es_sum > 0.0 ? loss_sum / es_sum : kNaN;
            mb.es = acerbi_szekely_test(x, model == 0 ? fg : fe, confs[k], es_sims, 1000 + k);
            mb.has_es = true;
            rep.rows.push_back(mb);
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
