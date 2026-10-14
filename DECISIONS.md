INTRADAY VAR ENGINE - DECISIONS
Status: Minimal scope
Date: 2026-10-14

1. GOAL

Implement and compare three intraday VaR models in a low-latency C++ engine:

1. EWMA-Normal
2. GARCH-t
3. Realized-CAViaR

The focus is the trade-off between statistical quality and computational cost.

Expected Shortfall / CVaR is out of scope.

2. MODELS

2.1 EWMA-Normal

Purpose:
Traditional low-cost baseline.

Variance:

    sigma_t^2 =
        lambda * sigma_(t-1)^2
        + (1 - lambda) * r_t^2

VaR:

    VaR_t =
        exposure_t * z_alpha * sigma_t

2.2 GARCH-t

Purpose:
More sophisticated volatility-model baseline with heavy tails.

Variance:

    sigma_t^2 =
        omega
        + alpha * r_(t-1)^2
        + beta * sigma_(t-1)^2

Innovations:

    z_t ~ Student-t(nu)

VaR is obtained from the conditional volatility and Student-t quantile.

2.3 Realized-CAViaR

Purpose:
Main model.

Initial specification:

    q_(t+1) =
        beta_0
        + beta_1 * q_t
        + beta_2 * |R_t|
        + beta_3 * sqrt(RV_t)

where:

    q_t
        Conditional lower-tail return quantile.

    R_t
        Aggregate return over the information window.

    RV_t
        Realized variance computed from higher-frequency returns:

            RV_t = sum_j r_(t,j)^2

    beta_0 ... beta_3
        Parameters estimated offline by minimizing quantile loss.

For 99% VaR:

    tau = 0.01

The model directly estimates the 1% conditional return quantile instead
of estimating volatility and assuming a return distribution.

3. INITIAL TIME CONFIGURATION

Working configuration:

    VaR confidence level:        99%
    VaR forecast horizon:       5 minutes
    Forecast update frequency:  1 minute
    Market sampling interval:   5 seconds
    Realized-measure window:    5 minutes

Operational forecasts may overlap.

For statistical backtesting, the primary evaluation should use
non-overlapping 5-minute observations when required by the test.

4. IMPLEMENTATION

Primary language:

    C++20 or C++23

Build system:

    CMake

Suggested minimal libraries:

    Boost.Math
        Student-t distribution and quantiles.

    Eigen
        Linear algebra if required.

    NLopt
        Offline parameter optimization.

    GoogleTest
        Unit tests.

    Google Benchmark
        Microbenchmarks.

No distributed stream-processing framework is required for the first
version.

No Kafka / Redpanda is required for the first version.

No database is required for the critical path.

No web frontend is required.

5. MINIMAL PIPELINE

Historical market data
|
v
Replay
|
v
Sampling / return calculation
|
+----------------------+
| |
v v
Aggregate return Realized variance
| |
+----------+-----------+
|
v
+----------------------+
| EWMA-Normal |
| GARCH-t |
| Realized-CAViaR |
+----------------------+
|
v
VaR predictions
|
v
Backtest + benchmark

6. CALIBRATION

Model parameters are calibrated offline.

EWMA-Normal:
lambda is fixed or calibrated from training data.

GARCH-t:
estimate omega, alpha, beta, and nu.

Realized-CAViaR:
estimate beta_0, beta_1, beta_2, and beta_3 by minimizing pinball /
quantile loss.

No numerical optimization is performed in the online hot path.

7. BACKTESTING

All three models must be evaluated on the same out-of-sample periods.

Required statistical metrics:

    - VaR exception rate
    - Kupiec unconditional coverage test
    - Christoffersen independence test
    - Christoffersen conditional coverage test
    - quantile / pinball loss

Optional:

    - Dynamic Quantile test

8. PERFORMANCE EVALUATION

Required:

    - time per model update
    - p50 latency
    - p95 latency
    - p99 latency
    - throughput
    - memory / state size

The main comparison is:

    statistical quality
            versus
    computational cost

9. DATA QUALITY

Keep data-quality handling minimal.

Required checks:

    - invalid timestamps
    - duplicate observations
    - out-of-order observations
    - missing / stale data
    - invalid prices

Extreme but valid market returns must not automatically be discarded.

A complex dq_score and NORMAL / DEGRADED / FREEZE state machine are not
required for the first version.

10. OUT OF SCOPE

For the first version:

    - Expected Shortfall / CVaR
    - Monte Carlo VaR
    - EVT
    - neural-network models
    - multi-factor covariance models
    - distributed deployment
    - Kafka / Redpanda
    - REST / WebSocket APIs
    - dashboards
    - complex alerting
    - complex data-quality scoring
    - production persistence

11. OPEN DECISIONS

Only the following decisions are still required before implementation:

A. Market and instrument
Example: B3 equity, B3 future, US equity, crypto.

B. Historical high-frequency data source
Required granularity should support approximately 5-second sampling.

C. Confirm the working temporal configuration:
confidence = 99%
horizon = 5 min
update = 1 min
sampling = 5 s
realized window = 5 min

D. Confirm whether the first implementation should use:
one instrument only
or
multiple independent instruments evaluated separately.

A true multi-asset portfolio model is outside the minimal scope.

12. RESEARCH QUESTION

Does Realized-CAViaR provide materially better intraday VaR forecasts
than EWMA-Normal and GARCH-t while retaining sufficiently low
computational cost for real-time execution in C++?
