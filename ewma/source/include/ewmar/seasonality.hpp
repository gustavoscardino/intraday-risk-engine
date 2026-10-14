// Intraday seasonality ("U" shape of volatility along the session). The factor s_k of slot k
// is the mean absolute return of that slot, winsorized, and normalized so that mean_k s_k^2 = 1:
// a return divided by s_k has the same variance at every time of the day.
#pragma once

#include <cstddef>
#include <vector>

namespace ewmar {

enum class DiurnalMode { Pooled, PerAsset, Off };
const char* to_string(DiurnalMode m);

class DiurnalProfile {
public:
    DiurnalProfile() = default;
    DiurnalProfile(int bars_per_day, size_t assets, DiurnalMode mode);

    // rows: row-major [bar][asset] raw returns (NaN = missing); slots: slot of each row.
    // Slots with fewer than `min_obs` observations take the pooled factor (or 1).
    void estimate(const std::vector<double>& rows, const std::vector<int>& slots, int min_obs = 5);

    double factor(int slot, size_t asset) const { return f_[static_cast<size_t>(slot) * n_ + asset]; }
    const std::vector<double>& pooled() const { return pooled_; }  // one factor per slot
    int bars_per_day() const { return K_; }
    DiurnalMode mode() const { return mode_; }

private:
    int K_ = 1;
    size_t n_ = 0;
    DiurnalMode mode_ = DiurnalMode::Pooled;
    std::vector<double> f_;       // [slot][asset]
    std::vector<double> pooled_;  // [slot]
};

}  // namespace ewmar
