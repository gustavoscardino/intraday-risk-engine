#include "ewmar/data_quality.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <unordered_set>

namespace ewmar {

const char* flag_name(std::uint16_t f) {
    switch (f) {
        case kGap: return "GAP";
        case kDuplicate: return "DUPLICATE";
        case kOutOfOrder: return "OUT_OF_ORDER";
        case kDelay: return "DELAY";
        case kStale: return "STALE";
        case kSpike: return "SPIKE";
        case kSuspect: return "SUSPECT";
        case kShockKept: return "SHOCK_KEPT";
        default: return "?";
    }
}

std::string flags_to_string(std::uint16_t flags) {
    std::string s;
    for (std::uint16_t b = 1; b != 0 && b <= kShockKept; b = static_cast<std::uint16_t>(b << 1))
        if (flags & b) s += (s.empty() ? "" : "|") + std::string(flag_name(b));
    return s;
}

const char* to_string(DqMode m) {
    switch (m) {
        case DqMode::Normal: return "NORMAL";
        case DqMode::Degraded: return "DEGRADED";
        case DqMode::Freeze: return "FREEZE";
    }
    return "?";
}

// ============================================================== ingestion

IngestResult ingest_feed(const FeedFile& feed, const SessionSpec& s, const DqConfig& cfg) {
    IngestResult res;
    const size_t n = feed.tickers.size();
    const long long K = s.bars_per_day;

    // trading-day index of each date (the exchange calendar), so that lateness counts grid bars
    std::set<int> days;
    for (const RawRow& r : feed.rows)
        if (s.slot_of(r.time.minute) >= 0) days.insert(r.time.ymd);
    std::unordered_map<int, long long> day_index;
    long long di = 0;
    for (int d : days) day_index[d] = di++;
    auto ordinal = [&](const RawRow& r) { return day_index[r.time.ymd] * K + s.slot_of(r.time.minute); };

    auto flag_cell = [&](const TimePoint& tp, int asset, std::uint16_t f, const std::string& detail) {
        res.cell_flags[cell_key(tp.key(), asset)] |= f;
        res.events.push_back({tp.str(), feed.tickers[static_cast<size_t>(asset)], flag_name(f), detail});
    };

    // ---- arrival rules, in file order
    std::map<long long, std::vector<const RawRow*>> accepted;  // ordinal -> rows (one per asset)
    std::unordered_set<long long> taken;                       // (ordinal, asset) accepted
    std::vector<long long> wm(n, -1);
    long long wm_global = -1;
    for (const RawRow& r : feed.rows) {
        if (s.slot_of(r.time.minute) < 0) {
            ++res.off_grid;
            continue;
        }
        const long long o = ordinal(r);
        const long long cell = o * 4096 + r.asset;
        const size_t a = static_cast<size_t>(r.asset);
        if (taken.count(cell)) {
            ++res.duplicates;
            flag_cell(r.time, r.asset, kDuplicate, "repeated row discarded (file line " + std::to_string(r.line) + ")");
            continue;
        }
        if (wm_global - o > cfg.lateness_bars) {
            ++res.delayed;
            flag_cell(r.time, r.asset, kDelay,
                      "arrived " + std::to_string(wm_global - o) + " bars late, after the bar closed: discarded");
            continue;
        }
        if (o < wm[a]) {
            ++res.out_of_order;
            flag_cell(r.time, r.asset, kOutOfOrder, "arrived " + std::to_string(wm[a] - o) + " bar(s) after a newer row; accepted");
        }
        taken.insert(cell);
        accepted[o].push_back(&r);
        wm[a] = std::max(wm[a], o);
        wm_global = std::max(wm_global, o);
        ++res.accepted;
    }

    // ---- panel of the accepted rows
    Panel& p = res.panel;
    p.tickers = feed.tickers;
    const size_t T = accepted.size();
    for (auto* m : {&p.open, &p.high, &p.low, &p.close, &p.volume}) m->assign(T * n, kNaN);
    res.flags.assign(T * n, 0);
    size_t t = 0;
    for (const auto& kv : accepted) {
        const TimePoint tp = kv.second.front()->time;
        p.stamp.push_back(tp.str());
        p.key.push_back(tp.key());
        p.ymd.push_back(tp.ymd);
        p.minute.push_back(tp.minute);
        p.slot.push_back(s.slot_of(tp.minute));
        for (const RawRow* r : kv.second) {
            const size_t cell = t * n + static_cast<size_t>(r->asset);
            p.open[cell] = r->open;
            p.high[cell] = r->high;
            p.low[cell] = r->low;
            p.close[cell] = r->close;
            p.volume[cell] = r->volume;
        }
        ++t;
    }

    // ---- GAP and STALE, bar by bar
    struct Run {
        double last = kNaN;
        std::vector<size_t> bars;  // bars of the current unchanged, zero-volume run
    };
    std::vector<Run> run(n);
    for (t = 0; t < T; ++t) {
        TimePoint tp;
        tp.ymd = p.ymd[t];
        tp.minute = p.minute[t];
        for (size_t i = 0; i < n; ++i) {
            const size_t cell = t * n + i;
            if (!valid_price(p.close[cell])) {
                ++res.gaps;
                res.flags[cell] |= kGap;
                flag_cell(tp, static_cast<int>(i), kGap, "no row for this bar");
                continue;  // a gap neither extends nor breaks a frozen run
            }
            const double c = p.close[cell];
            const bool frozen = valid_price(run[i].last) && c == run[i].last && p.volume[cell] == 0.0;
            run[i].last = c;
            if (!frozen) {
                run[i].bars.clear();
                continue;
            }
            run[i].bars.push_back(t);
            const size_t len = run[i].bars.size();
            if (static_cast<int>(len) < cfg.stale_bars) continue;
            if (static_cast<int>(len) == cfg.stale_bars)  // earlier bars of the run were already used
                for (size_t k = 0; k + 1 < len; ++k) {
                    const size_t u = run[i].bars[k];
                    TimePoint tu;
                    tu.ymd = p.ymd[u];
                    tu.minute = p.minute[u];
                    res.flags[u * n + i] |= kStale;
                    ++res.stale;
                    flag_cell(tu, static_cast<int>(i), kStale, "frozen quote (flagged retroactively; already used)");
                }
            ++res.stale;
            res.flags[cell] |= kStale;
            flag_cell(tp, static_cast<int>(i), kStale,
                      "close unchanged with volume 0 for " + std::to_string(len) + " bars: price ignored");
            p.close[cell] = p.open[cell] = p.high[cell] = p.low[cell] = kNaN;  // treated as missing
        }
    }
    // flags of off-panel rows already in cell_flags; mirror the panel ones (OUT_OF_ORDER, DUPLICATE)
    for (t = 0; t < T; ++t)
        for (size_t i = 0; i < n; ++i) {
            const auto it = res.cell_flags.find(cell_key(p.key[t], static_cast<int>(i)));
            if (it != res.cell_flags.end()) res.flags[t * n + i] |= it->second;
        }
    return res;
}

// ============================================================== score and modes

double dq_score(const std::uint16_t* flags, const double* x, size_t n, const DqConfig& cfg) {
    double wsum = 0.0;
    for (size_t i = 0; i < n; ++i) wsum += std::isfinite(x[i]) ? std::fabs(x[i]) : 0.0;
    double score = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double w = wsum > 0.0 ? (std::isfinite(x[i]) ? std::fabs(x[i]) : 0.0) / wsum : 1.0 / static_cast<double>(n);
        if (w == 0.0) continue;
        const std::uint16_t f = flags[i];
        double pen = 0.0;
        if (f & kGap) pen += cfg.pen_gap;
        if (f & kStale) pen += cfg.pen_stale;
        if (f & kSuspect) pen += cfg.pen_suspect;
        if (f & kSpike) pen += cfg.pen_spike;
        if (f & kOutOfOrder) pen += cfg.pen_out_of_order;
        if (f & kDuplicate) pen += cfg.pen_duplicate;
        score += w * std::clamp(1.0 - pen, 0.0, 1.0);
    }
    return score;
}

DqMode ModeController::level(double score) const {
    if (score >= cfg_.normal_threshold) return DqMode::Normal;
    if (score >= cfg_.freeze_threshold) return DqMode::Degraded;
    return DqMode::Freeze;
}

DqMode ModeController::update(double score) {
    const DqMode lv = level(score);
    if (lv > mode_) {  // worse: immediately
        mode_ = lv;
        better_run_ = 0;
    } else if (lv < mode_) {  // better: only after a run of qualifying bars
        run_worst_ = better_run_ == 0 ? lv : std::max(run_worst_, lv);
        if (++better_run_ >= cfg_.recover_bars) {
            mode_ = run_worst_;
            better_run_ = 0;
        }
    } else {
        better_run_ = 0;
    }
    return mode_;
}

// ============================================================== detection against the truth

std::vector<DetectionRow> evaluate_detection(const FeedFile& clean, const FeedFile& recv, const SessionSpec& s,
                                             const std::unordered_map<long long, std::uint16_t>& cell_flags,
                                             int lateness_bars) {
    struct Cell {
        double open, close, volume;
    };
    std::unordered_map<long long, Cell> c_cells;
    for (const RawRow& r : clean.rows)
        if (s.slot_of(r.time.minute) >= 0) c_cells.emplace(cell_key(r.time.key(), r.asset), Cell{r.open, r.close, r.volume});
    // grid-bar ordinal of each row, as the arrival rules count lateness
    std::set<int> days;
    for (const RawRow& r : recv.rows)
        if (s.slot_of(r.time.minute) >= 0) days.insert(r.time.ymd);
    std::unordered_map<int, long long> day_index;
    long long di = 0;
    for (int d : days) day_index[d] = di++;
    std::unordered_map<long long, int> count;
    std::unordered_map<long long, Cell> r_cells;
    std::unordered_set<long long> truth_ooo, truth_delay;
    std::vector<long long> wm(recv.tickers.size(), -1);
    long long wm_global = -1;
    for (const RawRow& r : recv.rows) {
        if (s.slot_of(r.time.minute) < 0) continue;
        const long long k = cell_key(r.time.key(), r.asset);
        const long long o = day_index[r.time.ymd] * s.bars_per_day + s.slot_of(r.time.minute);
        if (count[k]++ == 0) {
            r_cells.emplace(k, Cell{r.open, r.close, r.volume});
            // arrived after the feed had moved more than lateness_bars past its bar: late; otherwise
            // after a newer row of the same stock: out of order
            if (wm_global - o > lateness_bars) truth_delay.insert(k);
            else if (r.time.key() < wm[static_cast<size_t>(r.asset)]) truth_ooo.insert(k);
        }
        wm[static_cast<size_t>(r.asset)] = std::max(wm[static_cast<size_t>(r.asset)], r.time.key());
        wm_global = std::max(wm_global, o);
    }
    std::unordered_set<long long> truth_gap, truth_dup, truth_stale, truth_spike;
    for (const auto& kv : c_cells)
        if (!r_cells.count(kv.first)) truth_gap.insert(kv.first);
    for (const auto& kv : count)
        if (kv.second > 1) truth_dup.insert(kv.first);
    for (const auto& kv : r_cells) {
        const auto it = c_cells.find(kv.first);
        if (it == c_cells.end()) continue;
        const Cell &a = it->second, &b = kv.second;
        const bool differs = a.close != b.close || a.open != b.open || a.volume != b.volume;
        if (!differs) continue;
        if (b.volume == 0.0) truth_stale.insert(kv.first);
        else if (a.close != b.close) truth_spike.insert(kv.first);
    }
    struct Spec {
        std::uint16_t flag;
        const std::unordered_set<long long>* truth;
    };
    const Spec specs[] = {{kGap, &truth_gap},      {kDuplicate, &truth_dup}, {kOutOfOrder, &truth_ooo},
                          {kDelay, &truth_delay},  {kStale, &truth_stale},   {kSpike, &truth_spike}};
    std::vector<DetectionRow> out;
    for (const Spec& sp : specs) {
        DetectionRow d;
        d.flag = flag_name(sp.flag);
        d.injected = sp.truth->size();
        for (const auto& kv : cell_flags) {
            if (!(kv.second & sp.flag)) continue;
            if (sp.truth->count(kv.first)) ++d.detected;
            else ++d.other;
        }
        d.recall = d.injected > 0 ? static_cast<double>(d.detected) / static_cast<double>(d.injected) : kNaN;
        out.push_back(d);
    }
    return out;
}

}  // namespace ewmar
