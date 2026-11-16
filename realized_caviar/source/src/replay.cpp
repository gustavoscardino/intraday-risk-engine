#include "rcaviar/replay.hpp"
#include "garcht/garch.hpp"
#include "garcht/math.hpp"
#include "ewmar/ewma.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <stdexcept>

namespace rcaviar {
namespace {
using Clock=std::chrono::steady_clock;
constexpr std::array<const char*,3> names{"Realized-CAViaR","EWMA-N","GARCH-t"};
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double,std::nano>(Clock::now()-start).count();
}
std::size_t session_at(const ReplayInput& in,std::int64_t time) {
    return static_cast<std::size_t>(std::upper_bound(in.session_starts.begin(),in.session_starts.end(),time)-in.session_starts.begin()-1);
}
std::size_t phase_at(const ReplayInput& in,std::int64_t time) {
    if (in.shared) return 0;
    const auto anchor=std::upper_bound(in.phase_starts.begin(),in.phase_starts.end(),time)-1;
    return static_cast<std::size_t>((time-*anchor)/60%5);
}
struct BaselineState {
    ewmar::EwmaCovariance ewma{1};
    garcht::GarchFilter garch;
    double student_tail=0;
};
void validate(const ReplayInput& in,const ReplayConfig& cfg) {
    if (in.windows.empty() || in.session_starts.empty() || in.training_sessions==0 || in.horizon_seconds<=0)
        throw std::invalid_argument("replay needs windows, sessions and a positive horizon/training window");
    if (!std::isfinite(cfg.exposure) || !std::isfinite(cfg.confidence) || cfg.confidence<=.5 || cfg.confidence>=1
        || !std::isfinite(cfg.ewma_lambda) || cfg.ewma_lambda<=0 || cfg.ewma_lambda>=1)
        throw std::invalid_argument("invalid exposure, confidence or EWMA lambda");
    if (cfg.compare && in.horizon_seconds!=(in.shared?900:300))
        throw std::invalid_argument("unsupported comparison horizon");
    if (!std::is_sorted(in.session_starts.begin(),in.session_starts.end())
        || std::adjacent_find(in.session_starts.begin(),in.session_starts.end())!=in.session_starts.end()
        || in.session_starts.front()>in.windows.front().forecast_time)
        throw std::invalid_argument("invalid session boundaries");
    if (cfg.compare && !in.shared && (in.phase_starts.empty() || in.phase_starts.front()>in.windows.front().forecast_time
        || !std::is_sorted(in.phase_starts.begin(),in.phase_starts.end())
        || std::adjacent_find(in.phase_starts.begin(),in.phase_starts.end())!=in.phase_starts.end()))
        throw std::invalid_argument("invalid high-frequency phase anchors");
    std::int64_t last=-1;
    for (const auto& w:in.windows) {
        if (w.forecast_time<=last || w.target_time-w.forecast_time!=in.horizon_seconds
            || !std::isfinite(w.observation.aggregate_return) || !std::isfinite(w.observation.realized_variance)
            || w.observation.realized_variance<0 || std::isinf(w.observation.target_return))
            throw std::invalid_argument("invalid replay window");
        last=w.forecast_time;
    }
    if (in.return_times.size()!=in.returns.size() || (cfg.compare && in.returns.empty())
        || (!in.return_times.empty() && in.return_times.front()<in.session_starts.front())
        || (cfg.compare && !in.shared && in.return_times.front()<in.phase_starts.front()))
        throw std::invalid_argument("return grid is absent or inconsistent");
    last=-1;
    for (std::size_t i=0;i<in.return_times.size();++i) {
        if (in.return_times[i]<=last || std::isinf(in.returns[i])) throw std::invalid_argument("invalid return grid");
        last=in.return_times[i];
    }
}
}
ReplayInput shared_input(const SharedData& data) {
    ReplayInput in;
    in.windows=data.windows; in.return_times=data.return_times; in.returns=data.returns;
    in.session_starts=data.session_starts; in.cutoff=data.cutoff; in.horizon_seconds=900;
    in.training_sessions=data.training_sessions; in.shared=true;
    in.description="Shared 15m bars; preceding 1h features; no overnight targets. Sessions="+std::to_string(data.sessions)
        +"; initial training sessions="+std::to_string(data.training_sessions)+"; bad rows="+std::to_string(data.bad_rows)
        +"; duplicates="+std::to_string(data.duplicates);
    return in;
}
ReplayInput price_input(const std::vector<Price>& prices,bool reset_on_gap) {
    ReplayInput in;
    in.windows=make_windows(prices,reset_on_gap);
    if (in.windows.empty()) throw std::invalid_argument("no complete observed feature window");
    in.cutoff=prices[prices.size()/2].timestamp;
    in.session_starts.push_back(prices.front().timestamp);
    in.phase_starts.push_back(prices.front().timestamp);
    std::size_t gaps=0;
    for (std::size_t i=1;i<prices.size();++i) {
        const bool gap=prices[i].timestamp-prices[i-1].timestamp!=5;
        if (gap || prices[i].timestamp/86400!=prices[i-1].timestamp/86400) in.session_starts.push_back(prices[i].timestamp);
        if (gap) { ++gaps; in.phase_starts.push_back(prices[i].timestamp); }
    }
    in.training_sessions=session_at(in,in.cutoff)+1;
    // Five shifted 5m timelines. Each individual phase has non-overlapping
    // innovations, and its next observation is exactly the forecast's 5m target.
    for (const auto& w:in.windows) {
        in.return_times.push_back(w.forecast_time);
        in.returns.push_back(w.observation.aggregate_return);
    }
    in.description="5s samples; 5m horizon; 1m updates; gaps="+std::to_string(gaps)
        +"; gap policy="+(reset_on_gap?std::string("reset features, carry quantile"):std::string("reject"))
        +"; daily refits use timeline days and explicit gaps as session boundaries.";
    return in;
}
double position_var(double exposure,double q) {
    if (!std::isfinite(exposure) || !std::isfinite(q) || q>0) throw std::invalid_argument("invalid signed position quantile");
    const double value=exposure<0 ? -exposure*std::expm1(-q) : -exposure*std::expm1(q);
    if (!std::isfinite(value)) throw std::overflow_error("monetary VaR overflow");
    return value;
}
ReplayResult run_replay(const ReplayInput& in,const ReplayConfig& cfg) {
    validate(in,cfg);
    const auto first=std::lower_bound(in.windows.begin(),in.windows.end(),in.cutoff,
        [](const Window& w,std::int64_t t){return w.forecast_time<t;});
    if (first==in.windows.end()) throw std::invalid_argument("no out-of-sample forecasts");
    ReplayResult result;
    const double sign=cfg.exposure<0?-1.0:1.0, tau=1-cfg.confidence;
    const std::size_t models=cfg.compare?3:1;
    std::optional<Filter> caviar;
    const std::size_t phases=in.shared?1:5;
    std::vector<BaselineState> baselines(phases);
    const double normal_tail=garcht::math::norm_ppf(tau);
    std::size_t previous_session=std::numeric_limits<std::size_t>::max(), return_cursor=0;
    std::int64_t next_evaluation=-1;
    auto calibrate=[&](std::size_t session,std::int64_t effective,std::int64_t end) {
        Calibration record;
        record.effective_time=effective; record.training_end=end;
        const std::size_t start_session=session>in.training_sessions?session-in.training_sessions:0;
        record.training_start=in.session_starts[start_session];
        std::vector<Observation> training;
        for (const auto& w:in.windows) {
            if (w.forecast_time>=effective) break;
            if (w.forecast_time<record.training_start) continue;
            auto row=w.observation;
            row.target_return=w.target_time<=end?sign*row.target_return:std::nan("");
            training.push_back(row);
        }
        auto start=Clock::now();
        try {
            auto options=cfg.fit_options; options.tau=tau;
            record.caviar=fit(training,options);
            caviar.emplace(record.caviar.params,record.caviar.last_quantile);
            record.status[0]=record.caviar.converged?"fitted":"not_converged";
        } catch (const std::exception&) {
            if (!caviar) throw;
            record.caviar=result.calibrations.back().caviar;
            record.caviar.last_quantile=caviar->quantile();
            record.status[0]="previous_fit";
        }
        record.elapsed_ms[0]=elapsed(start)/1e6;
        if (cfg.compare) {
            const auto begin=std::lower_bound(in.return_times.begin(),in.return_times.end(),record.training_start);
            const auto stop=std::upper_bound(in.return_times.begin(),in.return_times.end(),end);
            std::vector<std::vector<double>> histories(phases);
            for (auto it=begin;it!=stop;++it)
                histories[phase_at(in,*it)].push_back(sign*in.returns[static_cast<std::size_t>(it-in.return_times.begin())]);
            for (std::size_t phase=0;phase<phases;++phase) {
                auto& state=baselines[phase];
                const auto& history=histories[phase];
                BaselineCalibration b; b.phase=phase;
                start=Clock::now();
                const auto observed=std::count_if(history.begin(),history.end(),[](double r){return std::isfinite(r);});
                if (observed<100) {
                    if (result.calibrations.empty()) throw std::invalid_argument("baseline phase needs at least 100 observed training returns; use --caviar-only for short histories");
                    b.status[0]="previous_fit";
                } else {
                    state.ewma.set(ewmar::initial_covariance(history,1,0,history.size()));
                    for (double r:history) state.ewma.update(&r,cfg.ewma_lambda);
                    b.status[0]="fitted";
                }
                b.ewma_variance=state.ewma(0,0);
                b.elapsed_ms[0]=elapsed(start)/1e6;
                start=Clock::now();
                garcht::FitOptions options; options.compute_std_errors=false;
                const auto fitted=garcht::fit_garch(history,garcht::GarchSpec{},options);
                if (!fitted.usable()) {
                    if (result.calibrations.empty()) throw std::runtime_error("initial GARCH fit failed (phase "+std::to_string(phase)
                        +"): "+fitted.message+"; use --caviar-only for short histories");
                    b.status[1]="previous_fit";
                } else {
                    state.garch.reset(fitted.params,fitted.next_variance);
                    state.student_tail=garcht::math::std_t_ppf(tau,fitted.params.nu);
                    b.status[1]=garcht::to_string(fitted.status);
                }
                const auto& p=state.garch.params();
                b.garch_params={p.mu,p.omega,p.alpha,p.beta,p.nu};
                b.elapsed_ms[1]=elapsed(start)/1e6;
                record.elapsed_ms[1]+=b.elapsed_ms[0]; record.elapsed_ms[2]+=b.elapsed_ms[1];
                record.baselines.push_back(b);
            }
            return_cursor=static_cast<std::size_t>(stop-in.return_times.begin());
        }
        result.calibrations.push_back(record);
    };
    for (auto it=first;it!=in.windows.end();++it) {
        const auto& w=*it;
        const auto session=session_at(in,w.forecast_time);
        if (!caviar || (cfg.refit==RefitSchedule::Daily && session!=previous_session)) {
            // Advance old baseline state before a possible failed refit/fallback.
            const auto end=caviar?in.session_starts[session]-1:in.cutoff;
            if (caviar && cfg.compare) while (return_cursor<in.returns.size() && in.return_times[return_cursor]<=end) {
                auto& state=baselines[phase_at(in,in.return_times[return_cursor])];
                double r=sign*in.returns[return_cursor++]; state.ewma.update(&r,cfg.ewma_lambda); state.garch.update(r);
            }
            calibrate(session,w.forecast_time,end);
        }
        previous_session=session;
        Forecast f;
        f.forecast_time=w.forecast_time; f.target_time=w.target_time;
        f.aggregate_return=w.observation.aggregate_return; f.realized_variance=w.observation.realized_variance;
        f.target_return=w.observation.target_return;
        auto start=Clock::now();
        f.quantile[0]=caviar->update(f.aggregate_return,f.realized_variance);
        f.var[0]=position_var(cfg.exposure,f.quantile[0]);
        f.latency_ns[0]=elapsed(start);
        if (cfg.compare) {
            auto& selected=baselines[phase_at(in,w.forecast_time)];
            const auto begin=return_cursor;
            while (return_cursor<in.returns.size() && in.return_times[return_cursor]<=w.forecast_time) ++return_cursor;
            start=Clock::now();
            for (std::size_t j=begin;j<return_cursor;++j) {
                double r=sign*in.returns[j]; baselines[phase_at(in,in.return_times[j])].ewma.update(&r,cfg.ewma_lambda);
            }
            f.quantile[1]=normal_tail*std::sqrt(selected.ewma(0,0)); f.var[1]=position_var(cfg.exposure,f.quantile[1]);
            f.latency_ns[1]=elapsed(start);
            start=Clock::now();
            for (std::size_t j=begin;j<return_cursor;++j) baselines[phase_at(in,in.return_times[j])].garch.update(sign*in.returns[j]);
            f.quantile[2]=selected.garch.mean()+selected.student_tail*std::sqrt(selected.garch.variance()); f.var[2]=position_var(cfg.exposure,f.quantile[2]);
            f.latency_ns[2]=elapsed(start);
        }
        // Selection depends on target intervals, not future target availability.
        if (f.forecast_time>=next_evaluation) {
            next_evaluation=f.target_time;
            f.evaluated=std::isfinite(f.target_return);
        }
        result.forecasts.push_back(f);
    }
    for (std::size_t m=0;m<models;++m) {
        ModelSummary summary; summary.model=names[m];
        std::vector<int> hits; std::vector<bool> adjacent; std::vector<double> latency;
        std::int64_t previous_target=-1;
        for (const auto& f:result.forecasts) {
            latency.push_back(f.latency_ns[m]);
            summary.mean_ns+=f.latency_ns[m];
            if (!f.evaluated) continue;
            hits.push_back(sign*f.target_return<f.quantile[m]);
            adjacent.push_back(f.forecast_time==previous_target);
            previous_target=f.target_time;
            summary.mean_pinball+=pinball(sign*f.target_return,f.quantile[m],tau);
        }
        if (hits.empty()) throw std::invalid_argument("no observed out-of-sample targets");
        summary.coverage=coverage_with_boundaries(hits,adjacent,tau);
        summary.mean_pinball/=static_cast<double>(hits.size());
        summary.mean_ns/=static_cast<double>(latency.size());
        std::sort(latency.begin(),latency.end());
        auto pct=[&](double p){return latency[static_cast<std::size_t>(p*static_cast<double>(latency.size()-1))];};
        summary.p50_ns=pct(.5); summary.p95_ns=pct(.95); summary.p99_ns=pct(.99);
        summary.updates_per_second=summary.mean_ns>0?1e9/summary.mean_ns:std::numeric_limits<double>::infinity();
        summary.state_bytes=m==0?sizeof(Filter):m==1?phases*(sizeof(ewmar::EwmaCovariance)+sizeof(double)+sizeof(char))+sizeof(double)
            :phases*(sizeof(garcht::GarchFilter)+sizeof(double));
        result.summaries.push_back(summary);
    }
    return result;
}
void write_results(const ReplayInput& in,const ReplayConfig& cfg,const ReplayResult& result,const std::string& folder) {
    std::filesystem::create_directories(folder);
    auto output=[&](const char* name){
        std::ofstream file(std::filesystem::path(folder)/name);
        if (!file) throw std::runtime_error(std::string("cannot write ")+name);
        file << std::setprecision(17); return file;
    };
    auto forecasts=output("forecasts.csv"), risk=output(in.shared?"risk_15m.csv":"risk_5m.csv"), backtest=output("backtest_summary.csv"),
        parameters=output("params_daily.csv"), benchmark=output("benchmark.csv"), report=output("report.txt");
    forecasts << "model,forecast_time,target_time,aggregate_return,realized_variance,quantile,exposure,var,target_return,realized_pnl,exception,evaluated\n";
    risk << "model,forecast_time,target_time,exposure,confidence,var,realized_pnl,evaluated\n";
    for (const auto& f:result.forecasts) for (std::size_t m=0;m<result.summaries.size();++m) {
        const bool observed=std::isfinite(f.target_return);
        const double pnl=observed?cfg.exposure*std::expm1(f.target_return):std::nan("");
        // Output quantile uses the original asset-return direction (upper tail for shorts).
        const double q=cfg.exposure<0?-f.quantile[m]:f.quantile[m];
        forecasts << names[m] << ',' << f.forecast_time << ',' << f.target_time << ',' << f.aggregate_return << ',' << f.realized_variance
            << ',' << q << ',' << cfg.exposure << ',' << f.var[m] << ',' << f.target_return << ',' << pnl << ',';
        if (observed) forecasts << ((cfg.exposure<0?-f.target_return:f.target_return)<f.quantile[m]);
        forecasts << ',' << f.evaluated << '\n';
        risk << names[m] << ',' << f.forecast_time << ',' << f.target_time << ',' << cfg.exposure << ',' << cfg.confidence
            << ',' << f.var[m] << ',' << pnl << ',' << f.evaluated << '\n';
    }
    backtest << "model,confidence,n,exceptions,rate,expected_rate,kupiec_p,christoffersen_ind_p,conditional_coverage_p,mean_pinball\n";
    benchmark << "model,mean_ns,p50_ns,p95_ns,p99_ns,updates_per_second,state_bytes\n";
    report << "Matched intraday VaR replay\n" << in.description << "\nConfidence: " << cfg.confidence << "; signed exposure: " << cfg.exposure
        << "\nRefit: " << (cfg.refit==RefitSchedule::Daily?"daily rolling":"once") << "; calibrations: " << result.calibrations.size()
        << "\nInitial training cutoff: " << in.cutoff << "; horizon seconds: " << in.horizon_seconds
        << "\nOOS forecast times: " << result.forecasts.size()
        << "\nReturns/quantiles are log returns. Monetary VaR and P&L use expm1 (no linear P&L approximation).\n"
        << "Evaluation: same observed non-overlapping targets for every model; independence excludes session/gap transitions.\n"
        << "EWMA: fixed lambda=" << cfg.ewma_lambda << "; GARCH: zero-mean Student-t; no diurnal adjustment.\n"
        << (in.shared?"Baselines: one filter each, raw 15m returns, exact next-bar targets.\n"
            :"Baselines: five staggered filters each, non-overlapping 5m returns per phase, exact next 5m targets; no variance scaling approximation.\n")
        << "Latency: model update(s) + return quantile + monetary VaR; prepared inputs, no CSV/features/scoring/refit; includes clock overhead.\n"
        << "Baselines consume all intervening observations of their timeline; CAViaR consumes complete feature windows.\n"
        << "State bytes: persistent filter and cached quantiles only; excludes history, prepared features, allocator overhead and measurements.\n"
        << "Shorts fit the lower tail of -return. Coverage tests are asymptotic; small tail samples have low power.\n";
    for (const auto& s:result.summaries) {
        const auto& c=s.coverage;
        backtest << s.model << ',' << cfg.confidence << ',' << c.n << ',' << c.exceptions << ',' << c.rate << ',' << 1-cfg.confidence
            << ',' << c.p_uc << ',' << c.p_ind << ',' << c.p_cc << ',' << s.mean_pinball << '\n';
        benchmark << s.model << ',' << s.mean_ns << ',' << s.p50_ns << ',' << s.p95_ns << ',' << s.p99_ns << ',' << s.updates_per_second << ',' << s.state_bytes << '\n';
        report << s.model << ": n=" << c.n << " exceptions=" << c.exceptions << " rate=" << c.rate << " Kupiec_p=" << c.p_uc
            << " independence_p=" << c.p_ind << " conditional_coverage_p=" << c.p_cc << " mean_pinball=" << s.mean_pinball
            << "\n  mean_ns=" << s.mean_ns << " p50_ns=" << s.p50_ns << " p95_ns=" << s.p95_ns << " p99_ns=" << s.p99_ns
            << " updates_per_second=" << s.updates_per_second << " state_bytes=" << s.state_bytes << '\n';
    }
    parameters << "effective_time,training_start,training_end,model,phase,status,fit_ms,beta0,beta1,beta2,beta3,initial_quantile,last_quantile,mean_pinball,constant_pinball,evaluations,ewma_variance,mu,omega,alpha,beta,nu\n";
    for (const auto& c:result.calibrations) {
        auto emit=[&](std::size_t model,std::size_t phase,const std::string& status,double ms,const std::array<double,15>& values){
            parameters << c.effective_time << ',' << c.training_start << ',' << c.training_end << ',' << names[model] << ',' << phase << ',' << status << ',' << ms;
            for (double value:values) { parameters << ','; if (std::isfinite(value)) parameters << value; }
            parameters << '\n';
        };
        const auto& p=c.caviar.params;
        std::array<double,15> values; values.fill(std::nan(""));
        values[0]=p.beta0; values[1]=p.beta1; values[2]=p.beta2; values[3]=p.beta3;
        values[4]=c.caviar.initial_quantile; values[5]=c.caviar.last_quantile;
        values[6]=c.caviar.mean_loss; values[7]=c.caviar.constant_loss; values[8]=c.caviar.evaluations;
        emit(0,0,c.status[0],c.elapsed_ms[0],values);
        for (const auto& b:c.baselines) {
            values.fill(std::nan("")); values[9]=b.ewma_variance;
            emit(1,b.phase,b.status[0],b.elapsed_ms[0],values);
            values.fill(std::nan(""));
            std::copy(b.garch_params.begin(),b.garch_params.end(),values.begin()+10);
            emit(2,b.phase,b.status[1],b.elapsed_ms[1],values);
        }
    }
    for (auto* file:{&forecasts,&risk,&backtest,&parameters,&benchmark,&report}) {
        file->close(); if (!*file) throw std::runtime_error("output write failed");
    }
}
}
