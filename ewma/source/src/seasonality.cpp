#include "ewmar/seasonality.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "ewmar/market_data.hpp"

namespace ewmar {

const char* to_string(DiurnalMode m) {
    switch (m) {
        case DiurnalMode::Pooled: return "pooled";
        case DiurnalMode::PerAsset: return "per-asset";
        case DiurnalMode::Off: return "off";
    }
    return "?";
}

DiurnalProfile::DiurnalProfile(int bars_per_day, size_t assets, DiurnalMode mode)
    : K_(bars_per_day), n_(assets), mode_(mode) {
    if (bars_per_day < 1) throw std::invalid_argument("DiurnalProfile: bars_per_day < 1");
    f_.assign(static_cast<size_t>(K_) * n_, 1.0);
    pooled_.assign(static_cast<size_t>(K_), 1.0);
}

namespace {
// rescales the finite entries of v so that their mean square is 1; returns false if impossible
bool normalize_rms(std::vector<double>& v) {
    double ss = 0.0;
    int k = 0;
    for (double x : v)
        if (std::isfinite(x)) {
            ss += x * x;
            ++k;
        }
    if (k == 0 || !(ss > 0.0)) return false;
    const double s = std::sqrt(ss / k);
    for (double& x : v)
        if (std::isfinite(x)) x /= s;
    return true;
}
}  // namespace

void DiurnalProfile::estimate(const std::vector<double>& rows, const std::vector<int>& slots, int min_obs) {
    const size_t K = static_cast<size_t>(K_), n = n_, T = slots.size();
    std::fill(f_.begin(), f_.end(), 1.0);
    std::fill(pooled_.begin(), pooled_.end(), 1.0);
    if (mode_ == DiurnalMode::Off || T == 0) return;

    // per-asset profile of the winsorized mean absolute return
    std::vector<double> own(K * n, kNaN);
    for (size_t i = 0; i < n; ++i) {
        std::vector<double> a;
        a.reserve(T);
        for (size_t t = 0; t < T; ++t) {
            const double r = rows[t * n + i];
            if (std::isfinite(r)) a.push_back(std::fabs(r));
        }
        if (a.size() < 10) continue;
        const size_t q = std::min(a.size() - 1, static_cast<size_t>(0.995 * static_cast<double>(a.size())));
        std::nth_element(a.begin(), a.begin() + static_cast<long>(q), a.end());
        const double cap = a[q];  // 99.5% quantile of |r|: one bad tick does not shape the U
        std::vector<double> sum(K, 0.0);
        std::vector<int> cnt(K, 0);
        for (size_t t = 0; t < T; ++t) {
            const double r = rows[t * n + i];
            const int k = slots[t];
            if (!std::isfinite(r) || k < 0 || k >= K_) continue;
            sum[static_cast<size_t>(k)] += std::min(std::fabs(r), cap);
            ++cnt[static_cast<size_t>(k)];
        }
        std::vector<double> m(K, kNaN);
        for (size_t k = 0; k < K; ++k)
            if (cnt[k] >= min_obs && sum[k] > 0.0) m[k] = sum[k] / cnt[k];
        if (!normalize_rms(m)) continue;
        for (size_t k = 0; k < K; ++k) own[k * n + i] = m[k];
    }

    // pooled profile: average of the per-asset profiles
    for (size_t k = 0; k < K; ++k) {
        double s = 0.0;
        int c = 0;
        for (size_t i = 0; i < n; ++i)
            if (std::isfinite(own[k * n + i])) {
                s += own[k * n + i];
                ++c;
            }
        pooled_[k] = c > 0 ? s / c : kNaN;
    }
    if (!normalize_rms(pooled_)) std::fill(pooled_.begin(), pooled_.end(), 1.0);
    for (double& v : pooled_)
        if (!std::isfinite(v)) v = 1.0;

    for (size_t k = 0; k < K; ++k)
        for (size_t i = 0; i < n; ++i) {
            const double v = own[k * n + i];
            f_[k * n + i] = mode_ == DiurnalMode::PerAsset && std::isfinite(v) ? v : pooled_[k];
        }
}

}  // namespace ewmar
