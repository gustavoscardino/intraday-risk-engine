// WebAssembly entry points of the matched comparison, for the dashboard (dashboard/engine/).
//
// The same C++ as the native engine: the dashboard calls run() for one stock with the chosen
// parameters and receives the forecasts of the four models, their backtest, the latency measured
// in the browser and the calibrated parameters, as JSON. The shared datasets are packaged in the
// module's virtual file system under /data/<dataset>/ (assets.csv, prices_15m.csv).
//
// Built by dashboard/tools/build_engine.(ps1|sh) with Emscripten; see dashboard/README.md.
#include <emscripten/bind.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include "garcht/market_data.hpp"
#include "garcht/risk.hpp"
#include "rcaviar/replay.hpp"

namespace {

// Minimal JSON writer: numbers with 10 significant digits, NaN / inf as null.
class Json {
public:
    Json& raw(const std::string& s) { out_ += s; return *this; }
    Json& key(const char* k) { sep(); out_ += '"'; out_ += k; out_ += "\":"; fresh_ = true; return *this; }
    Json& str(const std::string& s) {
        sep(); out_ += '"';
        for (char c : s) {
            if (c == '"' || c == '\\') { out_ += '\\'; out_ += c; }
            else if (static_cast<unsigned char>(c) < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); out_ += b; }
            else out_ += c;
        }
        out_ += '"'; return *this;
    }
    Json& num(double v) {
        sep();
        if (!std::isfinite(v)) { out_ += "null"; return *this; }
        char b[32]; std::snprintf(b, sizeof b, "%.10g", v); out_ += b; return *this;
    }
    Json& integer(long long v) { sep(); out_ += std::to_string(v); return *this; }
    Json& boolean(bool v) { sep(); out_ += v ? "true" : "false"; return *this; }
    Json& open(char c) { sep(); out_ += c; fresh_ = true; return *this; }
    Json& close(char c) { out_ += c; fresh_ = false; return *this; }
    template <class F> Json& array(const std::vector<F>& v) { open('['); for (const auto& x : v) num(static_cast<double>(x)); return close(']'); }
    std::string take() { return std::move(out_); }

private:
    void sep() { if (!fresh_) out_ += ','; fresh_ = false; }
    std::string out_;
    bool fresh_ = true;
};

std::string error_json(const std::string& message) {
    Json j; j.open('{').key("error").str(message).close('}'); return j.take();
}

}  // namespace

// Tickers of a packaged dataset, in assets.csv order: {"tickers":[...]}.
std::string tickers(const std::string& dataset) {
    try {
        const auto csv = garcht::read_csv("/data/" + dataset + "/assets.csv");
        const int c = csv.column("ticker");
        if (c < 0) throw std::runtime_error("assets.csv has no ticker column");
        Json j; j.open('{').key("tickers").open('[');
        for (const auto& row : csv.rows) j.str(row.at(static_cast<std::size_t>(c)));
        j.close(']').close('}');
        return j.take();
    } catch (const std::exception& e) { return error_json(e.what()); }
}

// One matched replay. lambda <= 0 estimates the EWMA lambda at each calibration; window_days <= 0
// uses half of the sessions (at most 60); benchmark_updates is the minimum number of timed
// updates per model (batches of 1000).
std::string run(const std::string& dataset, const std::string& ticker, int window_days, double confidence,
                double lambda, bool diurnal, double exposure, int benchmark_updates) {
    try {
        const auto t0 = std::chrono::steady_clock::now();
        auto data = rcaviar::read_shared_data("/data/" + dataset, ticker, window_days);
        auto input = rcaviar::shared_input(data);
        rcaviar::ReplayConfig cfg;
        cfg.compare = true;
        cfg.refit = rcaviar::RefitSchedule::Daily;
        cfg.confidence = confidence;
        cfg.exposure = exposure;
        cfg.diurnal = diurnal;
        cfg.estimate_lambda = !(lambda > 0);
        if (lambda > 0) cfg.ewma_lambda = lambda;
        cfg.benchmark_min_updates = static_cast<std::size_t>(std::max(benchmark_updates, 1000));
        const auto result = rcaviar::run_replay(input, cfg);
        const double elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        const std::size_t models = result.summaries.size();
        Json j;
        j.open('{');
        j.key("dataset").str(dataset).key("ticker").str(ticker);
        j.key("window_days").integer(static_cast<long long>(data.training_sessions));
        j.key("sessions").integer(static_cast<long long>(data.sessions));
        j.key("confidence").num(confidence).key("exposure").num(exposure);
        j.key("diurnal").boolean(result.diurnal).key("lambda_estimated").boolean(cfg.estimate_lambda);
        j.key("elapsed_ms").num(elapsed_ms);
        j.key("models").open('[');
        for (const auto& s : result.summaries) j.str(s.model);
        j.close(']');

        // forecasts, column-wise; quantiles are log-return quantiles in the asset's direction
        std::vector<double> t, y, R, RV;
        std::vector<int> ev;
        std::vector<std::vector<double>> q(models);
        for (const auto& f : result.forecasts) {
            t.push_back(static_cast<double>(f.forecast_time));
            y.push_back(f.target_return);
            ev.push_back(f.evaluated ? 1 : 0);
            R.push_back(f.aggregate_return);
            RV.push_back(f.realized_variance);
            for (std::size_t m = 0; m < models; ++m) q[m].push_back(cfg.exposure < 0 ? -f.quantile[m] : f.quantile[m]);
        }
        j.key("forecasts").open('{');
        j.key("t").array(t).key("y").array(y).key("eval").array(ev).key("R").array(R).key("RV").array(RV);
        j.key("q").open('[');
        for (const auto& col : q) j.array(col);
        j.close(']').close('}');

        j.key("summaries").open('[');
        for (const auto& s : result.summaries) {
            const auto& c = s.coverage;
            j.open('{').key("model").str(s.model).key("n").integer(c.n).key("exceptions").integer(c.exceptions)
                .key("rate").num(c.rate).key("rate_lo").num(c.rate_lo).key("rate_hi").num(c.rate_hi)
                .key("kupiec_p").num(c.p_uc).key("ind_p").num(c.p_ind).key("cc_p").num(c.p_cc)
                .key("pinball").num(s.mean_pinball).key("mean_ns").num(s.mean_ns).key("p50_ns").num(s.p50_ns)
                .key("p95_ns").num(s.p95_ns).key("p99_ns").num(s.p99_ns)
                .key("state_bytes").integer(static_cast<long long>(s.state_bytes));
            j.key("by_hour").open('[');
            for (const auto& h : s.by_hour) j.open('[').integer(h.hour).integer(h.coverage.n).integer(h.coverage.exceptions).close(']');
            j.close(']').close('}');
        }
        j.close(']');

        j.key("calibrations").open('[');
        for (const auto& c : result.calibrations) {
            const auto& p = c.caviar.params;
            j.open('{').key("t").integer(c.effective_time).key("status").str(c.status[0]);
            j.key("caviar").open('[').num(p.beta0).num(p.beta1).num(p.beta2).num(p.beta3).close(']');
            if (!c.baselines.empty()) {
                const auto& b = c.baselines.front();
                j.key("lambda").num(b.ewma_lambda).key("nu_ewma").num(b.ewma_nu);
                j.key("garch").open('[').num(b.garch_params[1]).num(b.garch_params[2]).num(b.garch_params[3]).num(b.garch_params[4]).close(']');
            }
            j.close('}');
        }
        j.close(']');
        j.key("diurnal_profile");
        if (!result.calibrations.empty()) j.array(result.calibrations.back().diurnal); else j.open('[').close(']');
        j.close('}');
        return j.take();
    } catch (const std::exception& e) { return error_json(e.what()); }
}

// VaR and ES factors per unit of sigma: Normal (nu <= 0) or unit-variance Student t.
std::string tail(double nu, double confidence) {
    const auto f = garcht::tail_factors(nu > 0 ? garcht::Dist::StudentT : garcht::Dist::Normal, nu, confidence);
    Json j; j.open('{').key("var").num(f.var).key("es").num(f.es).close('}'); return j.take();
}

EMSCRIPTEN_BINDINGS(rcaviar_engine) {
    emscripten::function("tickers", &tickers);
    emscripten::function("run", &run);
    emscripten::function("tail", &tail);
}
