#include "riskcommon/model.hpp"

#include <algorithm>
#include <cstdio>

namespace riskcommon {

namespace {
bool has_nu(const ModelInfo& m) { return m.family == Family::StudentT || m.family == Family::TBodyGpd; }
void put(std::string& s, const char* fmt, double v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, fmt, v);
    s += buf;
}
}  // namespace

HistoryRing::HistoryRing(size_t capacity, size_t assets)
    : cap_(capacity > 0 ? capacity : 1), n_(assets), rows_(cap_ * assets, kNaN), slots_(cap_, 0) {}

void HistoryRing::push(int slot, const double* r) {
    std::copy(r, r + n_, rows_.begin() + static_cast<long>(head_ * n_));
    slots_[head_] = slot;
    head_ = (head_ + 1) % cap_;
    if (count_ < cap_) ++count_;
}

History HistoryRing::view() {
    const size_t start = (head_ + cap_ - count_) % cap_;
    chrono_.resize(count_ * n_);
    chrono_slots_.resize(count_);
    for (size_t c = 0; c < count_; ++c) {
        const size_t s = (start + c) % cap_;
        std::copy(rows_.begin() + static_cast<long>(s * n_), rows_.begin() + static_cast<long>((s + 1) * n_),
                  chrono_.begin() + static_cast<long>(c * n_));
        chrono_slots_[c] = slots_[s];
    }
    History h;
    h.returns = chrono_.data();
    h.slots = chrono_slots_.data();
    h.bars = count_;
    h.assets = n_;
    return h;
}

std::vector<std::string> forecast_columns(const ModelInfo& m, const std::vector<std::string>& tags) {
    std::vector<std::string> c{m.prefix + "sigma"};
    if (has_nu(m)) c.push_back(m.prefix + "nu");
    for (const std::string& t : tags) c.push_back(m.prefix + "var_" + t);
    for (const std::string& t : tags) c.push_back(m.prefix + "es_" + t);
    for (const std::string& p : m.parameters) c.push_back(m.prefix + p);
    return c;
}

std::string format_forecast(const ModelInfo& m, const ModelForecast& f) {
    std::string s;
    put(s, ",%.4f", f.sigma);
    if (has_nu(m)) put(s, ",%.4f", f.shape.nu);
    for (const RiskMeasure& r : f.measures) put(s, ",%.2f", r.var);
    for (const RiskMeasure& r : f.measures) put(s, ",%.2f", r.es);
    for (size_t k = 0; k < m.parameters.size(); ++k) put(s, ",%.6g", k < f.parameters.size() ? f.parameters[k] : kNaN);
    return s;
}

}  // namespace riskcommon
