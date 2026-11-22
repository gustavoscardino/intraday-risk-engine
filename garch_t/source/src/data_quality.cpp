#include "garcht/data_quality.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace garcht {

namespace {
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
bool valid_price(double v) { return std::isfinite(v) && v > 0.0; }
double median(const std::deque<double>& d) {
    std::vector<double> v(d.begin(), d.end());
    std::nth_element(v.begin(), v.begin() + static_cast<long>(v.size() / 2), v.end());
    return v[v.size() / 2];
}
}  // namespace

JumpFilter::JumpFilter(size_t n_assets, const SessionSpec& session, JumpFilterConfig cfg)
    : n_(n_assets), session_(session), cfg_(cfg), assets_(n_assets),
      slot_norm_(static_cast<size_t>(std::max(session.bars_per_day, 1))) {}

double JumpFilter::median_abs(const Asset& a) const {
    return static_cast<int>(a.abs_ret.size()) < cfg_.min_returns ? kNaN : median(a.abs_ret);
}

double JumpFilter::slot_factor(int slot) const {
    const auto& d = slot_norm_[static_cast<size_t>(slot)];
    if (static_cast<int>(d.size()) < cfg_.min_slot_returns) return 1.0;
    const double f = median(d);
    return f > 0.0 ? f : 1.0;
}

void JumpFilter::apply(BarPanel& p, size_t t) {
    const int slot = bar_slot(p.minute[t], session_);
    if (slot < 0) return;  // off-grid rows are ignored by the engine
    const int date = p.date[t];
    std::vector<double> ret(n_, kNaN);
    std::vector<char> decided(n_, 0);  // the return enters now, whatever its size
    for (size_t i = 0; i < n_; ++i) {
        Asset& a = assets_[i];
        if (a.pending && a.last_date != date) a.pending = false;  // a new session: the held print is gone
        const double c = p.close[t * n_ + i];
        if (!valid_price(c)) continue;
        const double ref = a.last_date == date ? a.last_close : p.open[t * n_ + i];
        if (!valid_price(ref)) continue;
        ret[i] = std::log(c / ref);
        if (a.pending) {  // the bar after a held jump decides what it was
            const double undone = -std::log(c / a.held_close) / a.held_ret;  // share of the jump undone
            const bool bad = undone > cfg_.revert_share;
            events_.push_back({p.timestamps[t], p.tickers[i], bad ? "BAD_TICK" : "SHOCK_ACCEPTED", a.held_ret, 0.0});
            a.pending = false;
            decided[i] = !bad;
        }
    }
    const double factor = slot_factor(slot);
    std::vector<size_t> jumps;
    std::vector<double> zs(n_, 0.0);
    for (size_t i = 0; i < n_; ++i) {
        if (!std::isfinite(ret[i]) || decided[i]) continue;
        const double m = median_abs(assets_[i]);
        if (!(m > 0.0)) continue;
        zs[i] = std::fabs(ret[i]) / (1.4826 * m * factor);
        if (zs[i] > cfg_.z) jumps.push_back(i);
    }
    const double market = std::max<double>(cfg_.market_assets, cfg_.market_share * static_cast<double>(n_));
    const bool market_move = static_cast<double>(jumps.size()) >= market;
    for (size_t i : jumps) {
        if (market_move) {
            events_.push_back({p.timestamps[t], p.tickers[i], "MARKET_MOVE", ret[i], zs[i]});
            continue;
        }
        Asset& a = assets_[i];
        events_.push_back({p.timestamps[t], p.tickers[i], "HELD", ret[i], zs[i]});
        a.pending = true;
        a.held_close = p.close[t * n_ + i];
        a.held_ret = ret[i];
        p.close[t * n_ + i] = kNaN;  // as if the bar had not printed
        ret[i] = kNaN;
    }
    auto& norm = slot_norm_[static_cast<size_t>(slot)];
    for (size_t i = 0; i < n_; ++i) {
        if (!std::isfinite(ret[i])) continue;
        Asset& a = assets_[i];
        const double m = median_abs(a);
        if (m > 0.0) {
            norm.push_back(std::fabs(ret[i]) / m);
            if (static_cast<int>(norm.size()) > cfg_.slot_lookback) norm.pop_front();
        }
        a.abs_ret.push_back(std::fabs(ret[i]));
        if (static_cast<int>(a.abs_ret.size()) > cfg_.lookback) a.abs_ret.pop_front();
        a.last_close = p.close[t * n_ + i];
        a.last_date = date;
    }
}

}  // namespace garcht
