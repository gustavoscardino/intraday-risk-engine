#pragma once
#include "rcaviar/shared_data.hpp"
#include <array>
#include <string>

namespace rcaviar {
enum class RefitSchedule { Once, Daily };
struct ReplayInput {
    std::vector<Window> windows;
    std::vector<std::int64_t> return_times, session_starts, phase_starts;
    std::vector<double> returns;
    // Shared mode: session slot of each return and bars per session, for the diurnal profile.
    std::vector<int> return_slots;
    int bars_per_day=0;
    std::int64_t cutoff=0;
    int horizon_seconds=300;
    std::size_t training_sessions=1;
    bool shared=false;
    std::string description;
};
ReplayInput shared_input(const SharedData& data);
ReplayInput price_input(const std::vector<Price>& prices, bool reset_on_gap=false);
struct ReplayConfig {
    RefitSchedule refit=RefitSchedule::Once;
    bool compare=false; // HF: five staggered, non-overlapping 5m baseline filters
    double exposure=100000, confidence=.99, ewma_lambda=.94;
    // EWMA lambda estimated at each calibration (Gaussian predictive QLIKE, as in the ewma
    // module); false = ewma_lambda fixed. Diurnal: baselines run on r / s_slot (shared mode
    // only), with the U profile re-estimated at each calibration, as in ewma and garch_t.
    bool estimate_lambda=true, diurnal=true;
    // Model-stage latency is timed over batches of this many updates (clock resolution).
    std::size_t benchmark_batch=1000, benchmark_min_updates=2000000;
    FitOptions fit_options;
};
struct Forecast {
    std::int64_t forecast_time=0, target_time=0;
    double aggregate_return=0, realized_variance=0, target_return=0;
    // Quantiles are on signed returns: r for longs, -r for shorts.
    std::array<double,3> quantile{}, var{};
    bool evaluated=false;
};
struct BaselineCalibration {
    std::size_t phase=0;
    double ewma_variance=0, ewma_lambda=0;
    std::array<double,5> garch_params{}; // mu, omega, alpha, beta, nu
    std::array<double,2> elapsed_ms{};
    std::array<std::string,2> status;
};
struct Calibration {
    std::int64_t effective_time=0, training_start=0, training_end=0;
    FitResult caviar;
    std::array<double,3> elapsed_ms{};
    std::array<std::string,3> status;
    std::vector<BaselineCalibration> baselines;
    std::vector<double> diurnal;  // baseline U factor per slot; empty = no adjustment
};
struct HourCoverage {
    int hour=0;  // hour of day in which the target interval starts (exchange clock)
    garcht::CoverageTest coverage;
};
struct ModelSummary {
    std::string model;
    garcht::CoverageTest coverage;
    std::vector<HourCoverage> by_hour;
    // Latency per model update (+ quantile + monetary VaR), from batch timings:
    // mean = total time / updates; percentiles over the per-update mean of each batch.
    double mean_pinball=0, mean_ns=0, p50_ns=0, p95_ns=0, p99_ns=0, updates_per_second=0;
    std::size_t state_bytes=0, benchmark_updates=0;
};
struct ReplayResult {
    bool diurnal=false;  // baselines deseasonalized
    std::vector<Forecast> forecasts;
    std::vector<Calibration> calibrations;
    std::vector<ModelSummary> summaries;
};
// A common set of target timestamps and one calibration calendar for all models.
ReplayResult run_replay(const ReplayInput& input, const ReplayConfig& config = {});
void write_results(const ReplayInput& input, const ReplayConfig& config,
                   const ReplayResult& result, const std::string& folder);
// Quantile belongs to sign(exposure)*return. Short loss is -exposure*expm1(r).
double position_var(double exposure, double signed_quantile);
}
