#include "garcht/data_quality.hpp"

namespace garcht {

JumpFilter::JumpFilter(size_t n_assets, const SessionSpec& session, JumpFilterConfig cfg)
    : n_(n_assets), session_(session), cfg_(cfg) {}

void JumpFilter::apply(BarPanel& p, size_t t) {
    if (!core_) {
        std::vector<std::string> names = p.tickers;
        names.resize(n_);
        core_ = std::make_unique<riskcommon::JumpFilter>(names, session_.bars_per_day, cfg_);
    }
    const int slot = bar_slot(p.minute[t], session_);
    core_->apply(t, p.timestamps[t], p.date[t], slot, &p.open[t * n_], &p.close[t * n_]);
}

void JumpFilter::end_session() {
    if (core_) core_->end_session();
}

const std::vector<JumpEvent>& JumpFilter::events() const {
    static const std::vector<JumpEvent> none;
    return core_ ? core_->events() : none;
}

void JumpFilter::restore_accepted(std::vector<double>& ret) const {
    if (core_) core_->restore_accepted(ret);
}

const std::vector<riskcommon::JumpFilter::Accepted>& JumpFilter::accepted() const {
    static const std::vector<riskcommon::JumpFilter::Accepted> none;
    return core_ ? core_->accepted() : none;
}

riskcommon::JumpFilter::Counts JumpFilter::counts() const {
    return core_ ? core_->counts() : riskcommon::JumpFilter::Counts{};
}

}  // namespace garcht
