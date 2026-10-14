#include "rcaviar/data.hpp"
#include "rcaviar/shared_data.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

using Clock = std::chrono::steady_clock;
namespace {
struct Scores { std::vector<int> hits; std::vector<bool> adjacent; double loss=0; };
struct Latency {
    std::vector<double> ns;
    template<class F> double measure(F fn) {
        const auto start=Clock::now(); const double result=fn();
        ns.push_back(std::chrono::duration<double,std::nano>(Clock::now()-start).count());
        return result;
    }
    void print(std::ostream& out, const std::string& name, std::size_t state_bytes) {
        std::sort(ns.begin(),ns.end());
        double sum=0; for (double n:ns) sum+=n;
        const auto percentile=[&](double p){ return ns[static_cast<std::size_t>(p*static_cast<double>(ns.size()-1))]; };
        out << name << ": mean_ns=" << sum/static_cast<double>(ns.size()) << " p50_ns=" << percentile(.5)
            << " p95_ns=" << percentile(.95) << " p99_ns=" << percentile(.99)
            << " updates_per_second=" << 1e9*static_cast<double>(ns.size())/sum << " state_bytes=" << state_bytes << '\n';
    }
};
}
int main(int argc, char** argv) {
    try {
        std::string input, dataset, ticker, output="results/synthetic";
        int window_days=0;
        for (int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            if (arg=="--help") {
                std::cout << "realized_caviar [--prices timestamp_price.csv | --data dataset_folder --ticker SYMBOL]\n"
                    << "                 [--window-days N] [--out folder]\n"
                    << "Default: seeded instrument; 99% VaR, 5m horizon, 1m update, 5s samples.\n"
                    << "Shared data: 15m horizon/update, 1h realized window, session-safe features.\n";
                return 0;
            }
            if ((arg!="--prices" && arg!="--data" && arg!="--ticker" && arg!="--window-days" && arg!="--out") || i+1>=argc)
                throw std::invalid_argument("unknown or incomplete option: "+arg);
            const std::string value=argv[++i];
            if (arg=="--prices") input=value;
            else if (arg=="--data") dataset=value;
            else if (arg=="--ticker") ticker=value;
            else if (arg=="--out") output=value;
            else {
                std::size_t consumed=0;
                window_days=std::stoi(value,&consumed);
                if (consumed!=value.size() || window_days<=0) throw std::invalid_argument("window-days must be a positive integer");
            }
        }
        const bool shared=!dataset.empty();
        if ((shared && (!input.empty() || ticker.empty())) || (!shared && (!ticker.empty() || window_days!=0)))
            throw std::invalid_argument("use --data with --ticker; --prices is a separate mode; --window-days requires --data");
        std::vector<rcaviar::Window> windows;
        std::int64_t cutoff=0;
        std::string metadata;
        if (shared) {
            const auto loaded=rcaviar::read_shared_data(dataset,ticker,window_days);
            windows=loaded.windows; cutoff=loaded.cutoff;
            metadata="Dataset: "+dataset+"; ticker: "+ticker+"\nSessions: "+std::to_string(loaded.sessions)
                +"; training sessions: "+std::to_string(loaded.training_sessions)
                +"\nCSV bad rows: "+std::to_string(loaded.bad_rows)+"; duplicates: "+std::to_string(loaded.duplicates)
                +"\nTiming: 15m horizon/update; preceding 1h information window (four 15m returns).\n"
                +"Returns use garch_t conventions. Overnight/missing feature windows are excluded; missing targets are unscored.\n"
                +"Quantile state is carried across sessions/gaps; no updates during skipped windows.\n";
        } else {
            const auto prices=input.empty()?rcaviar::synthetic_prices():rcaviar::read_prices(input);
            windows=rcaviar::make_windows(prices);
            cutoff=prices[prices.size()/2].timestamp;
            metadata="Source: "+(input.empty()?std::string("seeded high-frequency simulation"):input)
                +"\nTiming: 5m horizon, 1m updates, 5s samples; continuous timeline.\n";
        }
        std::vector<rcaviar::Observation> training;
        for (const auto& w:windows) if (w.target_time<=cutoff) training.push_back(w.observation);
        const auto start=Clock::now();
        const auto fit=rcaviar::fit(training);
        const double fit_ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
        if (windows.back().forecast_time<cutoff) throw std::invalid_argument("not enough out-of-sample windows");
        std::filesystem::create_directories(output);
        std::ofstream csv(std::filesystem::path(output)/"forecasts.csv"), report(std::filesystem::path(output)/"report.txt");
        if (!csv || !report) throw std::runtime_error("cannot write outputs");
        csv << "forecast_time,target_time,aggregate_return,realized_variance,quantile,var_long_100000,target_return,exception,evaluated\n" << std::setprecision(17);
        rcaviar::Filter filter(fit.params,fit.initial_quantile);
        Scores scores; Latency latency; std::size_t oos=0;
        std::int64_t previous_target=-1;
        for (const auto& w:windows) {
            if (w.forecast_time<cutoff) { filter.update(w.observation.aggregate_return,w.observation.realized_variance); continue; }
            const auto& obs=w.observation;
            const double q=latency.measure([&]{return filter.update(obs.aggregate_return,obs.realized_variance);});
            const bool evaluate=oos++%(shared?1:5)==0 && std::isfinite(obs.target_return);
            if (evaluate) {
                scores.hits.push_back(obs.target_return<q);
                scores.adjacent.push_back(w.forecast_time==previous_target);
                previous_target=w.target_time;
                scores.loss+=rcaviar::pinball(obs.target_return,q,.01);
            }
            csv << w.forecast_time << ',' << w.target_time << ',' << obs.aggregate_return << ',' << obs.realized_variance << ',' << q << ',' << rcaviar::long_var(100000,q)
                << ',' << obs.target_return << ',' << (std::isfinite(obs.target_return)?(obs.target_return<q?"1":"0"):"") << ',' << evaluate << '\n';
        }
        report << std::setprecision(10) << "Learning experiment: one instrument; 99% VaR.\n" << metadata << "Training cutoff (local timeline seconds): " << cutoff
            << "\nTraining windows (targets end <= cutoff): " << training.size() << "\nOOS forecasts: " << oos
            << "\nFit ms: " << fit_ms << "\nFit converged: " << fit.converged << "\nFit evaluations: " << fit.evaluations
            << "\nbeta0,beta1,beta2,beta3: " << fit.params.beta0 << ',' << fit.params.beta1 << ',' << fit.params.beta2 << ',' << fit.params.beta3
            << "\nTraining pinball: " << fit.mean_loss << " constant baseline: " << fit.constant_loss
            << "\nCoverage uses non-overlapping targets; independence transitions exclude sessions/gaps. P-values are asymptotic.\n"
            << "Latency includes per-call clock overhead; state size excludes fit/history and timing vectors.\n";
        if (scores.hits.empty()) throw std::invalid_argument("no observed out-of-sample targets");
        const auto c=rcaviar::coverage_with_boundaries(scores.hits,scores.adjacent,.01);
        report << "Realized-CAViaR: n=" << c.n << " exceptions=" << c.exceptions << " rate=" << c.rate
            << " Kupiec_p=" << c.p_uc << " independence_p=" << c.p_ind << " conditional_coverage_p=" << c.p_cc
            << " mean_pinball=" << scores.loss/static_cast<double>(c.n) << '\n';
        latency.print(report,"Realized-CAViaR",sizeof(filter));
        csv.close(); report.close();
        if (!csv || !report) throw std::runtime_error("output write failed");
        std::cout << "Wrote " << output << "/forecasts.csv and report.txt\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << "error: " << error.what() << '\n'; return 1; }
}
