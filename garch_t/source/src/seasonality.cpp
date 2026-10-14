#include "garcht/seasonality.hpp"

#include <cmath>
#include <stdexcept>

namespace garcht {

DiurnalProfile DiurnalProfile::estimate(const double* r, const int* slot, size_t n, int n_slots,
                                        double cap_multiple, int min_count) {
    if (n_slots <= 0) throw std::invalid_argument("diurnal profile: n_slots must be positive");
    DiurnalProfile prof(n_slots);

    double total = 0.0;
    size_t count = 0;
    for (size_t i = 0; i < n; ++i)
        if (std::isfinite(r[i]) && slot[i] >= 0 && slot[i] < n_slots) {
            total += std::fabs(r[i]);
            ++count;
        }
    if (count == 0 || !(total > 0.0)) return prof;
    const double cap = cap_multiple * total / static_cast<double>(count);

    std::vector<double> sum(static_cast<size_t>(n_slots), 0.0);
    std::vector<int> cnt(static_cast<size_t>(n_slots), 0);
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(r[i]) || slot[i] < 0 || slot[i] >= n_slots) continue;
        sum[static_cast<size_t>(slot[i])] += std::fmin(std::fabs(r[i]), cap);
        ++cnt[static_cast<size_t>(slot[i])];
    }

    // Slot means; slots without enough data receive the average of the others.
    std::vector<double> m(static_cast<size_t>(n_slots), 0.0);
    double avg = 0.0;
    int good = 0;
    for (size_t k = 0; k < m.size(); ++k)
        if (cnt[k] >= min_count && sum[k] > 0.0) {
            m[k] = sum[k] / cnt[k];
            avg += m[k];
            ++good;
        }
    if (good == 0) return prof;
    avg /= good;
    for (size_t k = 0; k < m.size(); ++k)
        if (!(cnt[k] >= min_count && sum[k] > 0.0)) m[k] = avg;

    double ms = 0.0;
    for (double v : m) ms += v * v;
    const double norm = std::sqrt(ms / n_slots);
    for (size_t k = 0; k < m.size(); ++k) prof.s_[k] = m[k] / norm;
    return prof;
}

std::vector<DiurnalProfile> DiurnalProfile::estimate_per_asset(const double* r, const int* slot, size_t T,
                                                               size_t n, int n_slots, double cap_multiple,
                                                               int min_count) {
    if (n_slots <= 0) throw std::invalid_argument("diurnal profile: n_slots must be positive");
    std::vector<DiurnalProfile> out(n, DiurnalProfile(n_slots));
    if (n == 0 || T == 0) return out;
    const size_t K = static_cast<size_t>(n_slots);

    // pooled slot means of winsorized |r|, normalized by the pooled overall mean
    std::vector<int> all_slots(T * n);
    for (size_t t = 0; t < T; ++t)
        for (size_t i = 0; i < n; ++i) all_slots[t * n + i] = slot[t];
    const DiurnalProfile pooled = estimate(r, all_slots.data(), T * n, n_slots, cap_multiple, min_count);
    if (n == 1) {
        out[0] = pooled;
        return out;
    }

    // per-asset moments of winsorized |r| by slot
    std::vector<double> m(n * K, 0.0), noise(n * K, 0.0);
    std::vector<char> ok(n * K, 0), asset_ok(n, 0);
    for (size_t i = 0; i < n; ++i) {
        double total = 0.0;
        size_t cnt = 0;
        for (size_t t = 0; t < T; ++t) {
            const double v = r[t * n + i];
            if (std::isfinite(v) && slot[t] >= 0 && slot[t] < n_slots) {
                total += std::fabs(v);
                ++cnt;
            }
        }
        if (cnt == 0 || !(total > 0.0)) continue;
        const double overall = total / static_cast<double>(cnt), cap = cap_multiple * overall;
        std::vector<double> s1(K, 0.0), s2(K, 0.0);
        std::vector<int> c(K, 0);
        for (size_t t = 0; t < T; ++t) {
            const double v = r[t * n + i];
            if (!std::isfinite(v) || slot[t] < 0 || slot[t] >= n_slots) continue;
            const double a = std::fmin(std::fabs(v), cap) / overall;
            const size_t k = static_cast<size_t>(slot[t]);
            s1[k] += a;
            s2[k] += a * a;
            ++c[k];
        }
        asset_ok[i] = 1;
        for (size_t k = 0; k < K; ++k) {
            // a slot that never moved (e.g. an auction bar printed without a price change)
            // carries no scale information: it takes the common shape, as in estimate()
            if (c[k] < min_count || !(s1[k] > 0.0)) continue;
            const double mean = s1[k] / c[k];
            const double var = std::max(0.0, s2[k] / c[k] - mean * mean);
            m[i * K + k] = mean;
            // floor: a (nearly) constant |r| sample would otherwise get full weight 1
            noise[i * K + k] = std::max(var, 0.1 * mean * mean) / c[k];
            ok[i * K + k] = 1;
        }
    }
    // Shrinkage target per slot: equal-weight average of the assets' normalized profiles
    // (scale-free; the raw pooled profile would be dominated by the most volatile assets).
    // The between-asset variance tau^2 is a single hyperparameter estimated from all slots
    // (exchangeable slots): with few assets a per-slot estimate would be far too noisy.
    double pm = 0.0;
    for (size_t k = 0; k < K; ++k) pm += pooled[static_cast<int>(k)];
    pm /= static_cast<double>(K);
    std::vector<double> target(K);
    double between_sum = 0.0, noise_sum = 0.0;
    int slots_used = 0;
    for (size_t k = 0; k < K; ++k) {
        double mean_m = 0.0, mean_noise = 0.0;
        int cnt = 0;
        for (size_t i = 0; i < n; ++i)
            if (ok[i * K + k]) {
                mean_m += m[i * K + k];
                mean_noise += noise[i * K + k];
                ++cnt;
            }
        target[k] = pooled[static_cast<int>(k)] / pm;
        if (cnt >= 2) {
            mean_m /= cnt;
            mean_noise /= cnt;
            target[k] = mean_m;
            double var_between = 0.0;
            for (size_t i = 0; i < n; ++i)
                if (ok[i * K + k]) var_between += (m[i * K + k] - mean_m) * (m[i * K + k] - mean_m);
            between_sum += var_between / (cnt - 1);
            noise_sum += mean_noise;
            ++slots_used;
        }
    }
    const double tau2 = slots_used > 0 ? std::max(0.0, (between_sum - noise_sum) / slots_used) : 0.0;
    for (size_t k = 0; k < K; ++k)
        for (size_t i = 0; i < n; ++i) {
            double v = target[k];
            if (ok[i * K + k] && tau2 > 0.0) {
                const double w = tau2 / (tau2 + noise[i * K + k]);
                v = target[k] + w * (m[i * K + k] - target[k]);
            }
            out[i].s_[k] = std::max(v, 1e-6);
        }
    for (size_t i = 0; i < n; ++i) {
        if (!asset_ok[i]) {
            out[i] = pooled;
            continue;
        }
        double ms = 0.0;
        for (double v : out[i].s_) ms += v * v;
        const double norm = std::sqrt(ms / static_cast<double>(K));
        for (double& v : out[i].s_) v /= norm;
    }
    return out;
}

}  // namespace garcht
