#include "rcaviar/data.hpp"
#include "rcaviar/shared_data.hpp"
#include "rcaviar/replay.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
int checks=0;
void require(bool condition, const char* message) { ++checks; if (!condition) throw std::runtime_error(message); }
void close(double actual, double expected, double tolerance=1e-12) { require(std::abs(actual-expected)<=tolerance,"numeric mismatch"); }
template<class F> void rejects(F fn) { bool rejected=false; try { fn(); } catch (const std::exception&) { rejected=true; } require(rejected,"invalid input accepted"); }
}
int main() {
    try {
        rcaviar::Filter filter({-.001,.8,-.2,-.3},-.01);
        close(filter.update(-.002,1e-6),-.0097);
        close(rcaviar::pinball(-.02,-.01,.01),.0099);
        close(rcaviar::pinball(.02,-.01,.01),.0003);
        close(rcaviar::long_var(100000,-.01),100000*(1-std::exp(-.01)),1e-10);
        rejects([&]{filter.update(0,-1);}); close(filter.quantile(),-.0097);
        rejects([&]{filter.update(std::numeric_limits<double>::quiet_NaN(),0);});
        rejects([]{rcaviar::Filter f({0,1,0,0},-.01);});
        rejects([]{rcaviar::long_var(-1,-.01);});
        rejects([]{rcaviar::pinball(0,0,1);});
        std::vector<rcaviar::Price> prices;
        for (int i=0;i<=240;++i) prices.push_back({5*i,100*std::exp(.0001*i)});
        const auto windows=rcaviar::make_windows(prices);
        require(windows.size()==16,"pending forecasts must be emitted without future prices");
        require(std::isnan(windows.back().observation.target_return),"pending forecast must have no target");
        rcaviar::StreamingWindows streaming;
        std::size_t emitted=0;
        for (const auto& price:prices) if (const auto w=streaming.push(price)) {
            close(w->observation.aggregate_return,windows[emitted].observation.aggregate_return);
            close(w->observation.realized_variance,windows[emitted].observation.realized_variance);
            require(w->forecast_time==windows[emitted++].forecast_time,"stream/replay cadence differs");
            require(std::isnan(w->observation.target_return),"stream sees future labels");
        }
        require(emitted==windows.size(),"stream/replay count differs");
        auto gapped=prices;
        for (std::size_t i=120;i<gapped.size();++i) gapped[i].timestamp+=3600;
        rejects([&]{rcaviar::make_windows(gapped);});
        const auto reset=rcaviar::make_windows(gapped,true);
        require(reset.size()==11,"session gap must restart feature warmup");
        require(std::isnan(reset.front().observation.target_return),"target must not cross session gap");
        close(windows[0].observation.aggregate_return,.006);
        close(windows[0].observation.realized_variance,60e-8);
        close(windows[0].observation.target_return,.006);
        require(windows[0].forecast_time==300 && windows[0].target_time==600,"target timing");
        require(windows[1].forecast_time-windows[0].forecast_time==60,"update cadence");
        require(windows[5].forecast_time==windows[0].target_time,"non-overlapping evaluation");
        auto future_changed=prices; future_changed[70].price*=2;
        const auto changed=rcaviar::make_windows(future_changed);
        close(changed[0].observation.aggregate_return,windows[0].observation.aggregate_return);
        close(changed[0].observation.realized_variance,windows[0].observation.realized_variance);
        auto bad=prices; bad[1].timestamp=0; rejects([&]{rcaviar::make_windows(bad);});
        bad=prices; bad[1].timestamp=6; rejects([&]{rcaviar::make_windows(bad);});
        bad=prices; bad[1].price=0; rejects([&]{rcaviar::make_windows(bad);});
        // Valid extreme returns are retained, including their RV contribution.
        require(changed[1].observation.realized_variance>.5,"extreme return discarded");
        const auto generated=rcaviar::make_windows(rcaviar::synthetic_prices(12000));
        std::vector<rcaviar::Observation> training;
        for (std::size_t i=0;i<400;++i) training.push_back(generated[i].observation);
        const auto fitted=rcaviar::fit(training);
        require(fitted.params.valid(),"fit parameters invalid");
        require(fitted.mean_loss<=fitted.constant_loss+1e-15,"fit worse than constant candidate");
        rcaviar::Filter replay(fitted.params,fitted.initial_quantile);
        double loss=0;
        for (const auto& row:training) loss+=rcaviar::pinball(row.target_return,replay.update(row.aggregate_return,row.realized_variance),.01);
        close(loss/static_cast<double>(training.size()),fitted.mean_loss);
        close(replay.quantile(),fitted.last_quantile);
        rcaviar::Filter resumed(fitted.params,fitted.last_quantile);
        close(resumed.update(generated[400].observation.aggregate_return,generated[400].observation.realized_variance),replay.update(generated[400].observation.aggregate_return,generated[400].observation.realized_variance));
        auto scaled=training;
        for (auto& row:scaled) { row.aggregate_return*=100; row.target_return*=100; row.realized_variance*=10000; }
        const auto scaled_fit=rcaviar::fit(scaled);
        close(scaled_fit.mean_loss,100*fitted.mean_loss,1e-10);
        rejects([]{rcaviar::fit({});});
        training[0].target_return=std::numeric_limits<double>::infinity(); rejects([&]{rcaviar::fit(training);});
        const auto coverage=garcht::coverage_test({0,0,1,0,0},.01);
        require(coverage.exceptions==1 && coverage.n==5,"coverage counts");
        require(std::isfinite(coverage.p_uc) && std::isfinite(coverage.p_ind) && std::isfinite(coverage.p_cc),"coverage p-values");
        std::vector<garcht::BarRow> bars;
        for (int day=0;day<2;++day) for (int bar=0;bar<8;++bar) {
            garcht::BarRow row;
            row.time={20261005+day,615+15*bar,0}; row.ticker="TEST";
            row.open=100*std::exp(.001*bar); row.close=100*std::exp(.001*(bar+1));
            bars.push_back(row);
        }
        const auto panel=garcht::build_panel(bars,{"TEST"});
        const auto shared=rcaviar::make_bar_windows(panel,0);
        require(shared.size()==8,"session-safe window count");
        close(shared[0].observation.aggregate_return,.004);
        close(shared[0].observation.realized_variance,4e-6);
        close(shared[0].observation.target_return,.001);
        require(shared[0].target_time-shared[0].forecast_time==900,"shared target horizon");
        require(shared[1].forecast_time==shared[0].target_time,"shared non-overlapping targets");
        require(shared[4].forecast_time-shared[3].target_time>900,"overnight window included");
        auto future_panel=panel;
        future_panel.close[4]*=1.1;
        const auto future_windows=rcaviar::make_bar_windows(future_panel,0);
        close(future_windows[0].observation.aggregate_return,shared[0].observation.aggregate_return);
        close(future_windows[0].observation.realized_variance,shared[0].observation.realized_variance);
        require(future_windows[0].observation.target_return!=shared[0].observation.target_return,"future target unchanged");
        auto missing_panel=panel; missing_panel.close[3]=std::nan("");
        require(rcaviar::make_bar_windows(missing_panel,0).size()==4,"missing price window accepted");
        auto missing_target=panel; missing_target.close[4]=std::nan("");
        const auto unscored=rcaviar::make_bar_windows(missing_target,0);
        require(unscored[0].forecast_time==shared[0].forecast_time && std::isnan(unscored[0].observation.target_return),"missing future target suppressed observed features");
        close(unscored[0].observation.realized_variance,shared[0].observation.realized_variance);
        auto partial=scaled; partial[10].target_return=std::nan("");
        const auto partial_fit=rcaviar::fit(partial);
        rcaviar::Filter partial_filter(partial_fit.params,partial_fit.initial_quantile);
        double partial_loss=0;
        for (const auto& row:partial) {
            const double q=partial_filter.update(row.aggregate_return,row.realized_variance);
            if (std::isfinite(row.target_return)) partial_loss+=rcaviar::pinball(row.target_return,q,.01);
        }
        close(partial_loss/static_cast<double>(partial.size()-1),partial_fit.mean_loss);
        close(partial_filter.quantile(),partial_fit.last_quantile);
        rejects([&]{rcaviar::make_bar_windows(panel,1);});
        const std::vector<int> hits={0,0,1,0,1,1,0};
        const auto contiguous=rcaviar::coverage_with_boundaries(hits,std::vector<bool>(hits.size(),true),.01);
        const auto original=garcht::coverage_test(hits,.01);
        close(contiguous.lr_ind,original.lr_ind); close(contiguous.p_cc,original.p_cc);
        const auto segmented=rcaviar::coverage_with_boundaries({0,1,1,0},{false,true,false,true},.01);
        close(segmented.lr_ind,4*std::log(2.0));
        const auto isolated=rcaviar::coverage_with_boundaries({0,1},{false,false},.01);
        require(std::isnan(isolated.p_ind),"undefined independence must be NaN");
        rejects([]{rcaviar::coverage_with_boundaries({0},{},.01);});
        close(rcaviar::position_var(-100000,-.01),100000*std::expm1(.01));
        close(rcaviar::position_var(100000,-.01),rcaviar::long_var(100000,-.01));
        close(rcaviar::position_var(0,-.01),0);
        rejects([]{rcaviar::position_var(1,.01);});
        // End-to-end causal replay, aligned daily calibration, and short direction.
        rcaviar::ReplayInput input;
        input.shared=true; input.horizon_seconds=900; input.training_sessions=5;
        for (int day=0;day<8;++day) {
            const std::int64_t base=day*86400;
            input.session_starts.push_back(base);
            for (int slot=0;slot<28;++slot) {
                const auto time=base+(slot+1)*900;
                const double r=.002*std::sin(day*28+slot)+.001*std::cos(.37*(day*28+slot));
                input.return_times.push_back(time); input.returns.push_back(r);
                if (slot>=3 && slot<27) {
                    double aggregate=0,rv=0;
                    for (int j=0;j<4;++j) { const double x=input.returns[input.returns.size()-1-static_cast<std::size_t>(j)]; aggregate+=x; rv+=x*x; }
                    const double target=.002*std::sin(day*28+slot+1)+.001*std::cos(.37*(day*28+slot+1));
                    input.windows.push_back({time,time+900,{aggregate,rv,target}});
                }
            }
        }
        input.cutoff=input.return_times[5*28-1];
        rcaviar::ReplayConfig config; config.compare=true; config.refit=rcaviar::RefitSchedule::Daily;
        config.fit_options.starts=2; config.fit_options.max_iterations=120;
        const auto matched=rcaviar::run_replay(input,config);
        require(matched.calibrations.size()==3,"missing daily refits");
        require(matched.summaries.size()==3,"missing baseline models");
        for (const auto& s:matched.summaries) require(s.coverage.n==72,"models must use identical target sets");
        for (const auto& c:matched.calibrations) require(c.training_end<c.effective_time,"calibration sees forecast future");
        require(matched.calibrations[1].training_start==input.session_starts[1],"window must roll by session");
        auto perturbed=input;
        const auto first_time=matched.forecasts.front().forecast_time;
        for (std::size_t j=0;j<perturbed.returns.size();++j) if (perturbed.return_times[j]>first_time) perturbed.returns[j]*=2;
        for (auto& w:perturbed.windows) if (w.target_time>first_time) w.observation.target_return*=2;
        const auto causal=rcaviar::run_replay(perturbed,config);
        for (std::size_t m=0;m<3;++m) close(causal.forecasts.front().quantile[m],matched.forecasts.front().quantile[m]);
        config.refit=rcaviar::RefitSchedule::Once;
        const auto once=rcaviar::run_replay(input,config);
        require(once.calibrations.size()==1,"once must not refit daily");
        auto sparse=input;
        for (auto& w:sparse.windows) if (w.forecast_time>=5*86400 && w.forecast_time<6*86400) w.observation.target_return=std::nan("");
        config.refit=rcaviar::RefitSchedule::Daily;
        const auto fallback=rcaviar::run_replay(sparse,config);
        require(fallback.calibrations[1].status[0]=="previous_fit","failed rolling fit must preserve previous model");
        for (const auto& s:fallback.summaries) require(s.coverage.n==48,"missing targets must not change model sample alignment");
        config.refit=rcaviar::RefitSchedule::Once;
        config.exposure=-200000;
        const auto short_run=rcaviar::run_replay(input,config);
        for (const auto& f:short_run.forecasts) for (std::size_t m=0;m<3;++m)
            close(f.var[m],200000*std::expm1(-f.quantile[m]),1e-10);
        // Diurnal baselines: with a strong U shape, the same filter state gives a wider VaR for a
        // target at a volatile slot; returns enter the filters divided by their slot factor.
        auto seasonal=input;
        seasonal.bars_per_day=28;
        for (std::size_t j=0;j<seasonal.returns.size();++j) {
            const int slot=static_cast<int>(j%28);
            seasonal.return_slots.push_back(slot);
            seasonal.returns[j]*=(slot<4 || slot>23)?3.0:1.0;
        }
        rcaviar::ReplayConfig seasonal_config; seasonal_config.compare=true; seasonal_config.refit=rcaviar::RefitSchedule::Once;
        seasonal_config.fit_options.starts=2; seasonal_config.fit_options.max_iterations=120;
        const auto deseasonalized=rcaviar::run_replay(seasonal,seasonal_config);
        require(deseasonalized.diurnal,"diurnal adjustment not applied");
        require(deseasonalized.calibrations.front().diurnal.size()==28,"profile missing");
        const auto& u=deseasonalized.calibrations.front().diurnal;
        require(u[0]>2*u[12] && u[27]>2*u[12],"U profile not recovered");
        seasonal_config.diurnal=false;
        const auto raw=rcaviar::run_replay(seasonal,seasonal_config);
        require(!raw.diurnal && raw.calibrations.front().diurnal.empty(),"diurnal flag ignored");
        for (const auto& s:deseasonalized.summaries) require(s.benchmark_updates>=1000 && s.mean_ns>0,"batch benchmark missing");
        rcaviar::ReplayConfig reference=seasonal_config; reference.estimate_lambda=false; reference.ewma_lambda=.94;
        require(rcaviar::run_replay(seasonal,reference).summaries[1].model=="EWMA-N (ref)","RiskMetrics name");
        require(deseasonalized.summaries[1].model=="EWMA-N","estimated EWMA name");
        auto wrong=input; wrong.horizon_seconds=60; rejects([&]{rcaviar::run_replay(wrong,config);});
        config.confidence=1; rejects([&]{rcaviar::run_replay(input,config);});
        std::cout << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
