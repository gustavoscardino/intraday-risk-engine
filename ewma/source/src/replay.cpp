#include "ewmar/replay.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <stdexcept>

namespace ewmar {

namespace {

std::string upper(std::string s) {
    for (char& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

std::string fmt(const char* f, double a, double b = 0.0) {
    char buf[160];
    std::snprintf(buf, sizeof buf, f, a, b);
    return buf;
}

// A forecast waiting for its outcome.
struct Pending {
    size_t made_at = 0;  // panel bar of the forecast
    ForecastRecord rec;
    std::vector<double> qty, spot;
    std::vector<char> live;  // options valued at forecast time
    double option_value = 0.0;
};

// A price whose |z| exceeded the limit, decided with the next bar.
struct Suspect {
    size_t asset = 0;
    double r_orig = 0.0, ref_px = 0.0, close = 0.0, mark_before = kNaN;
    ReturnState rs_before;
};

}  // namespace

ReplayResult run_replay(const ReplayInput& in, const ReplayConfig& cfg) {
    const auto t_wall = std::chrono::steady_clock::now();
    ReplayResult res;
    const size_t n = in.tickers.size();
    if (n == 0 || in.feed.tickers.size() != n) throw std::runtime_error("replay: tickers / feed mismatch");
    const SessionSpec session = cfg.detect_session ? detect_session(in.feed.rows) : cfg.session;
    res.session = session;
    res.dq = cfg.dq;
    const DqConfig& dqc = cfg.dq_cfg;
    const int K = session.bars_per_day;

    // ---- panel: through the DQ layer or a plain sorted load
    Panel P;
    std::vector<std::uint16_t> flags;
    if (cfg.dq) {
        res.ingest = ingest_feed(in.feed, session, dqc);
        P = std::move(res.ingest.panel);
        res.ingest.panel = Panel{};
        flags = res.ingest.flags;
        res.events = res.ingest.events;
    } else {
        P = sorted_panel(in.feed, session, &res.panel_stats);
        flags.assign(P.bars() * n, 0);
    }
    const size_t T = P.bars();
    {
        std::set<int> days(P.ymd.begin(), P.ymd.end());
        res.sessions = days.size();
    }
    if (static_cast<int>(res.sessions) <= cfg.window_days)
        throw std::runtime_error("the data has " + std::to_string(res.sessions) + " sessions; a window of " +
                                 std::to_string(cfg.window_days) + " days needs at least one more session");
    std::vector<char> eod(T, 0);
    for (size_t t = 0; t < T; ++t) eod[t] = t + 1 == T || P.ymd[t + 1] != P.ymd[t];

    EngineConfig ecfg = cfg.engine;
    ecfg.bars_per_day = K;
    ecfg.window_bars = cfg.window_days * K;
    EwmaEngine eng(n, ecfg);
    ModeController modes(dqc);

    std::map<std::string, size_t> tix;
    for (size_t i = 0; i < n; ++i) tix[upper(in.tickers[i])] = i;
    struct BookOption {
        OptionContract c;
        size_t asset;
    };
    std::vector<BookOption> book;
    for (const OptionContract& o : in.options) {
        const auto it = tix.find(upper(o.underlying));
        if (it == tix.end()) throw std::runtime_error("option " + o.id + ": unknown underlying " + o.underlying);
        book.push_back({o, it->second});
    }
    res.has_book = !book.empty();
    auto option_value = [&](const BookOption& b, double spot, int ymd, int minute) {
        const double tt = years_to_expiry(ymd, minute, b.c.expiry_ymd, session);
        return b.c.quantity * b.c.multiplier * bs_price(b.c.call, spot, b.c.strike, tt, b.c.implied_vol, b.c.rate);
    };

    std::vector<double> qty = in.initial_quantity;
    qty.resize(n, 0.0);
    std::vector<ReturnState> rs(n);
    std::vector<double> mark(n, kNaN);                 // last clean close (exposures)
    std::vector<double> pnl_px(n, kNaN);               // P&L reference, lagging one bar
    std::vector<int> pnl_date(n, -1);
    std::vector<double> x(n, 0.0), xbook(n, 0.0), r(n), gap(n), ref_used(n);
    const std::vector<double> nan_row(n, kNaN);
    std::vector<ReturnState> rs_before(n);
    std::deque<Pending> queue;
    std::vector<Suspect> suspects;
    std::vector<double> fed_prev(n, kNaN);
    int fed_prev_slot = 0;
    bool fed_prev_frozen = false;
    bool have_on = false;
    int on_date = -1;
    OvernightRecord on;
    std::vector<double> on_qty(n, 0.0);
    PortfolioRisk risk, last_valid, book_risk;
    size_t next_fill = 0;
    int cur_date = -1, cur_slot = -1;
    auto add_event = [&](size_t bar, size_t asset, std::uint16_t f, const std::string& detail) {
        res.ingest.cell_flags[cell_key(P.key[bar], static_cast<int>(asset))] |= f;
        flags[bar * n + asset] |= f;
        res.events.push_back({P.stamp[bar], in.tickers[asset], flag_name(f), detail});
    };

    for (size_t t = 0; t <= T; ++t) {
        // ---- 1. decide the suspicious prices of bar t-1 with this bar's close
        if (!suspects.empty()) {
            const size_t u = t - 1;
            std::vector<double> row = fed_prev;
            bool refeed = false;
            for (const Suspect& s : suspects) {
                const size_t a = s.asset;
                const double c = t < T ? P.at(P.close, t, a) : kNaN;
                if (!valid_price(c)) {
                    ++res.unresolved;
                    res.events.push_back({P.stamp[u], in.tickers[a], "SUSPECT",
                                          "no next print to confirm: winsorized value kept"});
                    continue;
                }
                const double move = std::log(s.close / s.ref_px), after = std::log(c / s.ref_px);
                if (std::fabs(after) < dqc.revert_fraction * std::fabs(move)) {  // reverted: bad tick
                    ++res.spikes;
                    add_event(u, a, kSpike,
                              fmt("bad tick: move of %+.2f%% reverted by the next bar (%+.2f%% left): price excluded",
                                  100.0 * move, 100.0 * after));
                    res.alerts.push_back({P.stamp[u], "warning", "BAD_TICK",
                                          in.tickers[a] + fmt(" price excluded (move of %+.2f%% reverted)", 100.0 * move)});
                    P.at(P.close, u, a) = kNaN;
                    rs[a] = s.rs_before;  // bar t-1 replayed without its close
                    rs[a].step(P.ymd[u], P.slot[u], P.at(P.open, u, a), kNaN);
                    mark[a] = s.mark_before;
                    row[a] = kNaN;
                } else {
                    ++res.shocks_kept;
                    add_event(u, a, kShockKept,
                              fmt("move of %+.2f%% confirmed by the next bar (%+.2f%%): real shock, kept", 100.0 * move,
                                  100.0 * after));
                    row[a] = s.r_orig;
                }
                refeed = true;
            }
            if (refeed && !fed_prev_frozen) {
                if (eng.undo_last_bar()) {
                    eng.on_bar(fed_prev_slot, row.data());
                } else {  // a refit closed bar t-1: the correction reaches the next refit
                    for (const Suspect& s : suspects) eng.amend_last_bar(s.asset, row[s.asset]);
                }
            }
            suspects.clear();
        }

        // ---- 2. score the outcome of bar t-1
        if (t >= 1) {
            const size_t u = t - 1;
            const bool new_sess = u == 0 || P.ymd[u] != P.ymd[u - 1];
            const bool contiguous = new_sess ? P.slot[u] == 0 : P.slot[u] == P.slot[u - 1] + 1;
            if (have_on && new_sess) {
                if (P.slot[u] == 0) {
                    double p = 0.0;
                    for (size_t i = 0; i < n; ++i) {
                        const double c = P.at(P.close, u, i);
                        if (valid_price(c) && pnl_date[i] == on_date && valid_price(pnl_px[i])) p += on_qty[i] * (c - pnl_px[i]);
                    }
                    on.pnl = p;
                    on.target_time = P.stamp[u];
                    res.overnight.push_back(on);
                } else {
                    ++res.unscored_overnight;  // late opening
                }
                have_on = false;
            }
            while (!queue.empty() && queue.front().made_at + 1 < u) {
                queue.pop_front();
                ++res.unscored;
            }
            if (!queue.empty() && queue.front().made_at + 1 == u) {
                Pending& f = queue.front();
                if (contiguous && f.rec.target_slot == P.slot[u]) {
                    double pnl = 0.0;
                    for (size_t i = 0; i < n; ++i) {
                        const double c = P.at(P.close, u, i);
                        if (!valid_price(c)) continue;
                        const double ref = pnl_date[i] == P.ymd[u] ? pnl_px[i] : P.at(P.open, u, i);
                        if (valid_price(ref)) pnl += f.qty[i] * (c - ref);
                    }
                    f.rec.pnl = pnl;
                    f.rec.target_time = P.stamp[u];
                    if (res.has_book) {  // options alive at forecast time, revalued at this close
                        double v1 = 0.0, v0 = new_sess ? 0.0 : f.option_value;
                        for (size_t b = 0; b < book.size(); ++b) {
                            if (!f.live[b]) continue;
                            const size_t a = book[b].asset;
                            const double c = P.at(P.close, u, a);
                            v1 += option_value(book[b], valid_price(c) ? c : f.spot[a], P.ymd[u], P.minute[u]);
                            if (new_sess) {
                                const double o = P.at(P.open, u, a);
                                v0 += option_value(book[b], valid_price(o) ? o : f.spot[a], P.ymd[u],
                                                   P.minute[u] - session.bar_minutes);
                            }
                        }
                        f.rec.book_pnl = pnl + (v1 - v0);
                    }
                    res.forecasts.push_back(std::move(f.rec));
                } else {
                    ++res.unscored;  // the targeted bar never printed
                }
                queue.pop_front();
            }
            for (size_t i = 0; i < n; ++i) {
                const double c = P.at(P.close, u, i);
                if (valid_price(c)) {
                    pnl_px[i] = c;
                    pnl_date[i] = P.ymd[u];
                }
            }
        }
        if (t == T) break;

        // ---- 3. the new bar
        ++res.bars;
        res.last_time = P.stamp[t];
        const auto t0 = std::chrono::steady_clock::now();
        const int slot = P.slot[t], ymd = P.ymd[t];
        const bool new_session = ymd != cur_date;
        while (next_fill < in.fills.size() && in.fills[next_fill].time.key() <= P.key[t]) {
            const Fill& f = in.fills[next_fill++];
            const auto it = tix.find(upper(f.ticker));
            if (it == tix.end()) {
                ++res.fills_unknown;
                continue;
            }
            qty[it->second] += f.quantity;
            ++res.fills_applied;
        }
        for (size_t i = 0; i < n; ++i) {
            rs_before[i] = rs[i];
            const double o = P.at(P.open, t, i);
            ref_used[i] = ymd != rs[i].day || !valid_price(rs[i].ref) ? o : rs[i].ref;
            r[i] = rs[i].step(ymd, slot, o, P.at(P.close, t, i), &gap[i]);
        }
        // slots that never printed advance the filters with missing rows
        if (new_session) {
            if (cur_date >= 0)
                for (int k = cur_slot + 1; k < K; ++k) eng.on_bar(k, nan_row.data());
            eng.on_overnight(gap.data());
            for (int k = 0; k < slot; ++k) eng.on_bar(k, nan_row.data());
        } else {
            for (int k = cur_slot + 1; k < slot; ++k) eng.on_bar(k, nan_row.data());
        }

        // SPIKE rule against the forecast sigma of each asset
        std::uint16_t* ft = &flags[t * n];
        std::vector<double> fed = r;
        if (cfg.dq && eng.ready() && modes.mode() != DqMode::Freeze) {
            std::vector<size_t> cand;
            std::vector<double> sds;
            for (size_t i = 0; i < n; ++i) {
                if (!std::isfinite(r[i]) || !valid_price(ref_used[i])) continue;
                const double sd = eng.asset_sigma(i, slot);
                if (!(sd > 0.0) || std::fabs(r[i]) <= dqc.spike_z * sd) continue;
                cand.push_back(i);
                sds.push_back(sd);
            }
            const double market_min = std::max<double>(dqc.market_wide_assets, dqc.market_wide_share * static_cast<double>(n));
            if (!cand.empty() && static_cast<double>(cand.size()) >= market_min) {
                for (size_t c = 0; c < cand.size(); ++c) {  // a market move: real, kept at once
                    ++res.market_wide;
                    add_event(t, cand[c], kShockKept,
                              fmt("|z| = %.1f with %.0f assets jumping together: market move, kept",
                                  std::fabs(r[cand[c]]) / sds[c], static_cast<double>(cand.size())));
                }
            } else {
                for (size_t c = 0; c < cand.size(); ++c) {
                    const size_t i = cand[c];
                    Suspect s;
                    s.asset = i;
                    s.r_orig = r[i];
                    s.ref_px = ref_used[i];
                    s.close = P.at(P.close, t, i);
                    s.mark_before = mark[i];
                    s.rs_before = rs_before[i];
                    suspects.push_back(s);
                    ++res.suspects;
                    fed[i] = (r[i] > 0.0 ? 1.0 : -1.0) * dqc.spike_z * sds[c];
                    add_event(t, i, kSuspect,
                              fmt("|z| = %.1f above the limit: update winsorized, decided with the next bar",
                                  std::fabs(r[i]) / sds[c]));
                }
            }
        }
        for (size_t i = 0; i < n; ++i) {
            const double c = P.at(P.close, t, i);
            if (valid_price(c) && !(ft[i] & kSuspect)) mark[i] = c;
        }
        double net = 0.0, gross = 0.0;
        for (size_t i = 0; i < n; ++i) {
            x[i] = std::isfinite(mark[i]) ? qty[i] * mark[i] : 0.0;
            net += x[i];
            gross += std::fabs(x[i]);
        }

        // score and mode
        double score = 1.0;
        DqMode mode = DqMode::Normal;
        if (cfg.dq) {
            score = dq_score(ft, x.data(), n, dqc);
            const DqMode before = modes.mode();
            mode = modes.update(score);
            if (mode != before) {
                const char* sev = mode == DqMode::Freeze ? "critical" : (mode == DqMode::Degraded ? "warning" : "info");
                res.alerts.push_back({P.stamp[t], sev, "MODE",
                                      std::string(to_string(before)) + " -> " + to_string(mode) +
                                          fmt(" (dq_score %.3f)", score)});
            }
            for (size_t i = 0; i < n; ++i)
                if ((ft[i] & kStale) && !(t > 0 && (flags[(t - 1) * n + i] & kStale)))
                    res.alerts.push_back({P.stamp[t], "warning", "STALE", in.tickers[i] + " quote frozen: price ignored"});
        }
        ++res.mode_bars[static_cast<int>(mode)];
        const bool frozen = mode == DqMode::Freeze;

        // engine update (FREEZE: the bar enters the history as missing; filters unchanged)
        const std::vector<double>& feed_row = frozen ? nan_row : fed;
        eng.on_bar(slot, feed_row.data());
        fed_prev = feed_row;
        fed_prev_slot = slot;
        fed_prev_frozen = frozen;
        cur_date = ymd;
        cur_slot = slot;

        bool refitted = false;
        if (eod[t] && eng.history_size() >= static_cast<size_t>(ecfg.window_bars)) {
            const RefitReport rep = eng.refit(x.data());
            refitted = rep.ready;
            res.refit_ms.push_back(rep.seconds * 1e3);
            res.daily.push_back({P.stamp[t].substr(0, 10), rep.lambda, rep.nu, rep.lambda_on, rep.nu_on, rep.nu_obs,
                                 rep.seconds * 1e3});
        }
        const int next_slot = (slot + 1) % K;
        eng.compute(next_slot, x.data(), risk);
        if (!frozen && risk.ready) last_valid = risk;
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        res.engine_seconds += us * 1e-6;
        if (!refitted && eng.ready()) res.latency_us.push_back(us);

        if (eng.ready()) {
            const PortfolioRisk& pub = frozen && last_valid.ready ? last_valid : risk;
            Pending f;
            f.made_at = t;
            ForecastRecord& rec = f.rec;
            rec.forecast_time = P.stamp[t];
            rec.target_slot = next_slot;
            rec.net = net;
            rec.gross = gross;
            rec.sigma_ref = pub.ref.sigma;
            rec.sigma = pub.normal.sigma;
            rec.lambda = pub.lambda;
            rec.nu = pub.nu;
            rec.ref = pub.ref.m;
            rec.normal = pub.normal.m;
            rec.student = pub.student.m;
            rec.dq_score = score;
            rec.mode = mode;
            rec.frozen = frozen;
            rec.update_us = us;
            res.frozen_forecasts += frozen;
            f.qty = qty;
            f.spot = mark;
            if (res.has_book) {  // delta-linear book: option deltas added to the underlyings
                xbook = x;
                f.live.assign(book.size(), 0);
                for (size_t b = 0; b < book.size(); ++b) {
                    const BookOption& bo = book[b];
                    const double S = mark[bo.asset];
                    const double tt = years_to_expiry(ymd, P.minute[t], bo.c.expiry_ymd, session);
                    if (!valid_price(S) || !(tt > 0.0)) continue;
                    f.live[b] = 1;
                    xbook[bo.asset] += bo.c.quantity * bo.c.multiplier * S *
                                       bs_delta(bo.c.call, S, bo.c.strike, tt, bo.c.implied_vol, bo.c.rate);
                    f.option_value += option_value(bo, S, ymd, P.minute[t]);
                }
                eng.compute(next_slot, xbook.data(), book_risk);
                rec.book_sigma = book_risk.student.sigma;
                rec.book = book_risk.student.m;
            }
            queue.push_back(std::move(f));
        }
        if (eod[t] && ecfg.overnight) {
            const OvernightRisk o = eng.compute_overnight(x.data());
            if (o.available) {
                on = OvernightRecord{};
                on.forecast_time = P.stamp[t];
                on.net = net;
                on.risk = o;
                on_qty = qty;
                on_date = ymd;
                have_on = true;
            }
        }
    }
    res.unscored += queue.size() > 0 ? queue.size() - 1 : 0;  // the last forecast has no outcome

    res.quantity = qty;
    res.last_price = mark;
    res.exposure = x;
    res.last_risk = risk;
    res.correlation = eng.correlation();
    res.diurnal_pooled = eng.diurnal().pooled();
    res.diurnal.resize(static_cast<size_t>(K) * n);
    for (int k = 0; k < K; ++k)
        for (size_t i = 0; i < n; ++i) res.diurnal[static_cast<size_t>(k) * n + i] = eng.diurnal().factor(k, i);
    res.lambda = eng.lambda();
    res.nu = eng.nu();
    std::stable_sort(res.events.begin(), res.events.end(),
                     [](const DqEvent& a, const DqEvent& b) { return a.time < b.time; });
    res.wall_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_wall).count();
    return res;
}

// ============================================================== backtest

BacktestReport evaluate_backtest(const ReplayResult& r, const std::vector<double>& confs, int es_sims) {
    BacktestReport rep;
    const int bpd = r.session.bars_per_day;
    auto group = [&](int slot) { return slot < 3 ? 0 : (slot >= bpd - 3 ? 2 : 1); };
    const double inf = std::numeric_limits<double>::infinity();

    struct Model {
        const char* name;
        int kind;  // 0 ref, 1 normal, 2 student
    };
    const Model models[] = {{"EWMA-N (ref)", 0}, {"EWMA-N", 1}, {"EWMA-t", 2}};
    std::vector<double> x;
    std::vector<int> slots;
    std::vector<const ForecastRecord*> fr;
    for (const ForecastRecord& f : r.forecasts) {
        if (!std::isfinite(f.pnl)) continue;
        x.push_back(f.pnl);
        slots.push_back(f.target_slot);
        fr.push_back(&f);
    }
    for (const Model& m : models) {
        std::vector<DistForecast> dist;
        std::vector<double> pit;
        for (size_t t = 0; t < fr.size(); ++t) {
            const ForecastRecord& f = *fr[t];
            DistForecast d;
            d.sigma = m.kind == 0 ? f.sigma_ref : f.sigma;
            d.nu = m.kind == 2 ? f.nu : inf;
            dist.push_back(d);
            pit.push_back(forecast_cdf(d, x[t]));
        }
        rep.berkowitz.push_back({m.name, berkowitz_test(pit)});
        for (size_t k = 0; k < confs.size(); ++k) {
            ModelBacktest mb;
            mb.model = m.name;
            mb.confidence = confs[k];
            std::vector<int> hits, seg[3];
            double loss = 0.0, es = 0.0;
            for (size_t t = 0; t < fr.size(); ++t) {
                const ForecastRecord& f = *fr[t];
                const RiskMeasure& rm = m.kind == 0 ? f.ref[k] : (m.kind == 1 ? f.normal[k] : f.student[k]);
                const int h = -x[t] > rm.var;
                hits.push_back(h);
                seg[group(slots[t])].push_back(h);
                if (h) {
                    loss += -x[t];
                    es += rm.es;
                }
            }
            mb.coverage = coverage_test(hits, 1.0 - confs[k]);
            for (int g = 0; g < 3; ++g) mb.by_time[g] = coverage_test(seg[g], 1.0 - confs[k]);
            mb.has_by_time = true;
            mb.es_ratio = es > 0.0 ? loss / es : kNaN;
            mb.es = acerbi_szekely_z2(x, dist, confs[k], es_sims, 1000 + k);
            mb.has_es = true;
            rep.rows.push_back(mb);
        }
    }

    // overnight: close -> end of the next first bar
    if (!r.overnight.empty()) {
        for (int kind = 1; kind <= 2; ++kind) {
            std::vector<double> pit;
            for (const OvernightRecord& o : r.overnight)
                if (std::isfinite(o.pnl)) pit.push_back(forecast_cdf({o.risk.sigma, kind == 2 ? o.risk.nu : inf}, o.pnl));
            const char* name = kind == 1 ? "overnight EWMA-N" : "overnight EWMA-t";
            rep.berkowitz.push_back({name, berkowitz_test(pit)});
            for (size_t k = 0; k < confs.size(); ++k) {
                ModelBacktest mb;
                mb.model = name;
                mb.confidence = confs[k];
                std::vector<int> hits;
                double loss = 0.0, es = 0.0;
                for (const OvernightRecord& o : r.overnight) {
                    if (!std::isfinite(o.pnl)) continue;
                    const RiskMeasure& rm = kind == 1 ? o.risk.normal[k] : o.risk.student[k];
                    const int h = -o.pnl > rm.var;
                    hits.push_back(h);
                    if (h) {
                        loss += -o.pnl;
                        es += rm.es;
                    }
                }
                mb.coverage = coverage_test(hits, 1.0 - confs[k]);
                mb.es_ratio = es > 0.0 ? loss / es : kNaN;
                rep.rows.push_back(mb);
            }
        }
    }

    // option book, delta-linear (EWMA-t)
    if (r.has_book)
        for (size_t k = 0; k < confs.size(); ++k) {
            ModelBacktest mb;
            mb.model = "book delta-linear";
            mb.confidence = confs[k];
            std::vector<int> hits;
            double loss = 0.0, es = 0.0;
            for (const ForecastRecord& f : r.forecasts) {
                if (!std::isfinite(f.book_pnl) || k >= f.book.size()) continue;
                const int h = -f.book_pnl > f.book[k].var;
                hits.push_back(h);
                if (h) {
                    loss += -f.book_pnl;
                    es += f.book[k].es;
                }
            }
            mb.coverage = coverage_test(hits, 1.0 - confs[k]);
            mb.es_ratio = es > 0.0 ? loss / es : kNaN;
            rep.rows.push_back(mb);
        }
    return rep;
}

}  // namespace ewmar
