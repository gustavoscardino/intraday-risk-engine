#include "riskcommon/jump_filter.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace riskcommon {

namespace {
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
bool valid_price(double v) { return std::isfinite(v) && v > 0.0; }
}  // namespace

void JumpFilter::Window::push(double v, size_t cap) {
    if (ring_.size() < cap) {
        ring_.push_back(v);
        return;
    }
    ring_[head_] = v;  // the oldest value leaves
    head_ = (head_ + 1) % cap;
}

double JumpFilter::Window::median(std::vector<double>& scratch) const {
    scratch.assign(ring_.begin(), ring_.end());
    std::nth_element(scratch.begin(), scratch.begin() + static_cast<long>(scratch.size() / 2), scratch.end());
    return scratch[scratch.size() / 2];
}

JumpFilter::JumpFilter(std::vector<std::string> tickers, int bars_per_day, JumpFilterConfig cfg)
    : n_(tickers.size()), tickers_(std::move(tickers)), cfg_(cfg),
      lookback_(std::max(1, cfg.lookback_sessions * bars_per_day)),
      min_returns_(std::max(1, cfg.min_sessions * bars_per_day)), assets_(n_),
      slots_(static_cast<size_t>(std::max(bars_per_day, 1))), ret_(n_), zs_(n_), decided_(n_) {
    if (bars_per_day < 1) throw std::invalid_argument("JumpFilter: bars_per_day < 1");
}

void JumpFilter::end_session() {
    if (!dirty_) return;
    for (Asset& a : assets_)
        a.scale_median = static_cast<int>(a.abs_ret.size()) < min_returns_ ? kNaN : a.abs_ret.median(scratch_);
    for (Slot& s : slots_) {
        const double f = static_cast<int>(s.norm.size()) < cfg_.min_slot_returns ? 1.0 : s.norm.median(scratch_);
        s.factor = f > 0.0 ? f : 1.0;
    }
    dirty_ = false;
}

void JumpFilter::apply(size_t bar, const std::string& stamp, int date, int slot, const double* open, double* close) {
    if (slot < 0 || slot >= static_cast<int>(slots_.size())) return;  // off-grid rows are not decided
    if (date != date_) {  // a new session: the scales of the sessions completed so far
        end_session();
        date_ = date;
    }
    std::vector<double>& ret = ret_;
    std::vector<char>& decided = decided_;  // the return enters now, whatever its size
    std::fill(ret.begin(), ret.end(), kNaN);
    std::fill(decided.begin(), decided.end(), char(0));
    for (size_t i = 0; i < n_; ++i) {
        Asset& a = assets_[i];
        if (a.pending && a.last_date != date) {  // the session ended first: the held print never enters
            events_.push_back({stamp, tickers_[i], "UNDECIDED", a.held_ret, 0.0, bar, i});
            a.pending = false;
        }
        const double c = close[i];
        if (!valid_price(c)) continue;
        const double ref = a.last_date == date ? a.last_close : open[i];
        if (!valid_price(ref)) continue;
        ret[i] = std::log(c / ref);
        if (a.pending) {  // the bar after a held jump decides what it was
            const double undone = -std::log(c / a.held_close) / a.held_ret;  // share of the jump undone
            const bool bad = undone > cfg_.revert_share;
            events_.push_back({stamp, tickers_[i], bad ? "BAD_TICK" : "SHOCK_ACCEPTED", a.held_ret, 0.0, bar, i});
            a.pending = false;
            decided[i] = !bad;
            if (!bad) accepted_.push_back({bar, i, ret[i]});
        }
    }
    Slot& s = slots_[static_cast<size_t>(slot)];
    std::vector<size_t>& jumps = jumps_;
    std::vector<double>& zs = zs_;
    jumps.clear();
    std::fill(zs.begin(), zs.end(), 0.0);
    for (size_t i = 0; i < n_; ++i) {
        if (!std::isfinite(ret[i]) || decided[i]) continue;
        const double m = assets_[i].scale_median;
        if (!(m > 0.0)) continue;
        zs[i] = std::fabs(ret[i]) / (1.4826 * m * s.factor);
        if (zs[i] > cfg_.z) jumps.push_back(i);
    }
    const double market = std::max<double>(cfg_.market_assets, cfg_.market_share * static_cast<double>(n_));
    const bool market_move = static_cast<double>(jumps.size()) >= market;
    for (size_t i : jumps) {
        if (market_move) {
            events_.push_back({stamp, tickers_[i], "MARKET_MOVE", ret[i], zs[i], bar, i});
            continue;
        }
        Asset& a = assets_[i];
        events_.push_back({stamp, tickers_[i], "HELD", ret[i], zs[i], bar, i});
        if (a.last_date != date) {  // held at its first bar: the session's reference is this bar's open
            a.last_close = open[i];
            a.last_date = date;
        }
        a.pending = true;
        a.held_close = close[i];
        a.held_ret = ret[i];
        close[i] = kNaN;  // as if the bar had not printed
        ret[i] = kNaN;
    }
    for (size_t i = 0; i < n_; ++i) {  // kept returns feed the scales of the next sessions
        if (!std::isfinite(ret[i])) continue;
        Asset& a = assets_[i];
        if (a.scale_median > 0.0) s.norm.push(std::fabs(ret[i]) / a.scale_median, static_cast<size_t>(cfg_.slot_lookback));
        a.abs_ret.push(std::fabs(ret[i]), static_cast<size_t>(lookback_));
        a.last_close = close[i];
        a.last_date = date;
        dirty_ = true;
    }
}

void JumpFilter::restore_accepted(std::vector<double>& ret) const {
    for (const Accepted& a : accepted_) {
        const size_t k = a.bar * n_ + a.asset;
        if (k < ret.size() && std::isfinite(ret[k])) ret[k] = a.ret;
    }
}

JumpFilter::Counts JumpFilter::counts() const {
    Counts c;
    for (const JumpEvent& e : events_) {
        if (e.action == "HELD") ++c.held;
        else if (e.action == "BAD_TICK") ++c.bad_ticks;
        else if (e.action == "SHOCK_ACCEPTED") ++c.accepted;
        else if (e.action == "UNDECIDED") ++c.undecided;
        else if (e.action == "MARKET_MOVE") ++c.market_moves;
    }
    return c;
}

}  // namespace riskcommon
