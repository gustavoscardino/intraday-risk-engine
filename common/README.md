# common — code shared by the three engines

A small C++17 library (`riskcommon`) that `ewma`, `garch_t` and `realized_caviar` compile into their
own builds (`add_subdirectory(../../common ...)`). It holds the parts of the system that must be the
same in every engine for the model comparison to be fair.

| Header | What it is | Used by |
|---|---|---|
| `riskcommon/model.hpp` | the common model interface: `ModelInfo` (name, column prefix, distribution family, parameters), `ModelForecast` (mean, sigma, standardized shape, VaR / ES per confidence level), `PortfolioModel` (on_bar / refit / forecast / observe_outcome / rebalance), `HistoryRing`, and the output columns of a model | ewma, garch_t |
| `riskcommon/riskmetrics.hpp` | the RiskMetrics baseline (EWMA covariance of raw returns, lambda 0.94, Normal), one implementation for both portfolio engines | ewma, garch_t |
| `riskcommon/jump_filter.hpp` | the causal jump filter: an isolated price jump is held one bar, removed if the next bar undoes it (bad tick), let in at full size otherwise; jumps of many stocks at once are a market move and are kept | all three |

## The model interface

Each portfolio engine keeps its VaR models in a list built by `make_models` (`ewma/source/src/models.cpp`,
`garch_t/source/src/models.cpp`). The replay, the backtest and the output files iterate over that
list, so adding a model means writing one class that implements `riskcommon::PortfolioModel` and
adding it to `make_models`; nothing downstream names a specific model. Each model's columns in
`risk_15m.csv` are `{prefix}sigma`, `{prefix}nu` (t families), `{prefix}var_<c>`, `{prefix}es_<c>`
and `{prefix}<parameter>`.

| Engine | Models (prefix) |
|---|---|
| ewma | EWMA-N (ref) `ref_`, EWMA-N `n_`, EWMA-t `t_` (main) |
| garch_t | GARCH-t/EVT (no prefix, main), EWMA-N (ref) `ref_` |

realized_caviar forecasts quantiles of one return series with univariate filters that have another
life cycle (calibrated on pinball losses); it keeps its models in one orchestrator and uses the
shared jump filter.

## The jump filter

`JumpFilter::apply` decides one bar at a time, using that bar and earlier ones only, so a file and a
live feed are filtered the same way. The robust scale of each stock (1.4826 x the median absolute
return of its last 10 sessions, times an intraday factor per slot) is calibrated on the sessions
completed so far and stays fixed during a session (`end_session()`), so a bar costs O(n). The filter
changes what the models see (the "model view" of the prices); every engine scores its forecasts on
the prices as printed, so switching it off changes the forecasts, never the outcomes they are scored on.

## Tests

`tests/test_common.cpp` (normal quantiles, the baseline's missing-data rule and forecast, the
history ring, the output columns, the filter's decisions and causality) is built as
`riskcommon_tests` and runs in the `ctest` of each engine.
