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
    FitOptions fit_options;
};
struct Forecast {
    std::int64_t forecast_time=0, target_time=0;
    double aggregate_return=0, realized_variance=0, target_return=0;
    // Quantiles are on signed returns: r for longs, -r for shorts.
    std::array<double,3> quantile{}, var{}, latency_ns{};
    bool evaluated=false;
};
struct BaselineCalibration {
    std::size_t phase=0;
    double ewma_variance=0;
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
};
struct ModelSummary {
    std::string model;
    garcht::CoverageTest coverage;
    double mean_pinball=0, mean_ns=0, p50_ns=0, p95_ns=0, p99_ns=0, updates_per_second=0;
    std::size_t state_bytes=0;
};
struct ReplayResult {
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
