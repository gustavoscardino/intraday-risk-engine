// Intraday (diurnal) volatility profile.
//
// Intraday returns have a strong U-shaped volatility pattern (high at the open and
// close). GARCH is fitted on deseasonalized returns y_t = r_t / s_{k(t)}, where k(t) is
// the bar slot within the session, and the forecast for the next bar is rescaled by
// s_{k(t+1)} (Andersen & Bollerslev, 1997). Factors are normalized so mean(s_k^2) = 1.
#pragma once

#include <cstddef>
#include <vector>

namespace garcht {

class DiurnalProfile {
public:
    DiurnalProfile() = default;
    explicit DiurnalProfile(int n_slots) : s_(static_cast<size_t>(n_slots), 1.0) {}  // flat

    // Robust estimate from (slot, return) pairs, pooling any number of assets:
    // s_k proportional to the mean of |r| in slot k, with |r| winsorized at
    // `cap_multiple` times the overall mean |r|. Slots with fewer than `min_count`
    // observations get the average factor. Non-finite returns are ignored.
    static DiurnalProfile estimate(const double* r, const int* slot, size_t n, int n_slots,
                                   double cap_multiple = 8.0, int min_count = 10);

    // Per-asset profiles shrunk towards the common shape by empirical Bayes: for each slot,
    // asset i's normalized mean |r| is pulled towards the equal-weight average over assets
    // with weight tau^2 / (tau^2 + noise_i), where noise_i is the sampling variance of asset
    // i's slot mean and tau^2 the between-asset variance net of noise (one hyperparameter
    // estimated over all slots). Homogeneous assets collapse onto the common shape;
    // persistent asset-specific shapes are kept.
    // r is row-major [obs][asset], slot[obs] the session slot of each row.
    static std::vector<DiurnalProfile> estimate_per_asset(const double* r, const int* slot, size_t n_obs,
                                                          size_t n_assets, int n_slots,
                                                          double cap_multiple = 8.0, int min_count = 10);

    double operator[](int slot) const { return s_[static_cast<size_t>(slot)]; }
    int size() const { return static_cast<int>(s_.size()); }
    const std::vector<double>& factors() const { return s_; }

private:
    std::vector<double> s_;
};

}  // namespace garcht
