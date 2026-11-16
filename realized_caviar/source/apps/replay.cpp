#include "rcaviar/replay.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
double number(const std::string& value) {
    std::size_t used=0; const double result=std::stod(value,&used);
    if (used!=value.size() || !std::isfinite(result)) throw std::invalid_argument("invalid numeric argument");
    return result;
}
int positive_integer(const std::string& value) {
    std::size_t used=0; const int result=std::stoi(value,&used);
    if (used!=value.size() || result<=0) throw std::invalid_argument("expected positive integer");
    return result;
}
}
int main(int argc,char** argv) {
    try {
        std::string prices,dataset,ticker,output="results/synthetic",refit,diurnal;
        int window_days=0; bool reset_on_gap=false, caviar_only=false;
        rcaviar::ReplayConfig cfg;
        for (int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            if (arg=="--help") {
                std::cout << "realized_caviar [--prices timestamp_price.csv | --data folder --ticker SYMBOL]\n"
                    << "  --out folder             output folder\n"
                    << "  --window-days N          initial/rolling training sessions (shared mode)\n"
                    << "  --refit once|daily       default: daily for shared data, once for 5s data\n"
                    << "  --exposure E             signed monetary exposure (default 100000)\n"
                    << "  --confidence C           VaR confidence in (.5,1), default .99\n"
                    << "  --lambda L               fixed EWMA lambda in (0,1); default: estimated at each calibration\n"
                    << "  --diurnal on|off         shared mode: baselines on r / U-profile factor (default on)\n"
                    << "                           EWMA is named EWMA-N (ref) with --diurnal off --lambda .94 (RiskMetrics)\n"
                    << "  --caviar-only            disable matched EWMA/GARCH comparison\n"
                    << "  --reset-on-gap           5s mode: restart features after gaps, carry quantile\n"
                    << "  --prices FILE            shared mode: alternate 15m CSV; otherwise strict 5s CSV\n"
                    << "Shared: exact one-bar 15m targets. 5s: 5m targets/1m updates, staggered 5m baselines.\n";
                return 0;
            }
            if (arg=="--reset-on-gap") { reset_on_gap=true; continue; }
            if (arg=="--caviar-only") { caviar_only=true; continue; }
            if (arg!="--prices" && arg!="--data" && arg!="--ticker" && arg!="--window-days" && arg!="--out"
                && arg!="--refit" && arg!="--exposure" && arg!="--confidence" && arg!="--lambda" && arg!="--diurnal")
                throw std::invalid_argument("unknown option: "+arg);
            if (i+1>=argc) throw std::invalid_argument("incomplete option: "+arg);
            const std::string value=argv[++i];
            if (arg=="--prices") prices=value;
            else if (arg=="--data") dataset=value;
            else if (arg=="--ticker") ticker=value;
            else if (arg=="--out") output=value;
            else if (arg=="--window-days") window_days=positive_integer(value);
            else if (arg=="--refit") refit=value;
            else if (arg=="--exposure") cfg.exposure=number(value);
            else if (arg=="--confidence") cfg.confidence=number(value);
            else if (arg=="--diurnal") diurnal=value;
            else { cfg.ewma_lambda=number(value); cfg.estimate_lambda=false; }
        }
        const bool shared=!dataset.empty();
        if ((shared && (ticker.empty() || reset_on_gap)) || (!shared && (!ticker.empty() || window_days!=0)))
            throw std::invalid_argument("--data requires --ticker; --window-days requires --data; --reset-on-gap is for 5s data");
        if (!refit.empty() && refit!="daily" && refit!="once") throw std::invalid_argument("--refit must be once or daily");
        if (!diurnal.empty() && diurnal!="on" && diurnal!="off") throw std::invalid_argument("--diurnal must be on or off");
        if (!shared && diurnal=="on") throw std::invalid_argument("--diurnal applies to shared-data sessions only");
        cfg.diurnal=diurnal!="off";
        cfg.refit=(refit=="daily" || (refit.empty() && shared))?rcaviar::RefitSchedule::Daily:rcaviar::RefitSchedule::Once;
        cfg.compare=!caviar_only;
        auto input=shared?rcaviar::shared_input(rcaviar::read_shared_data(dataset,ticker,window_days,prices))
            :rcaviar::price_input(prices.empty()?rcaviar::synthetic_prices():rcaviar::read_prices(prices),reset_on_gap);
        input.description+=(shared?"\nDataset: "+dataset+"; ticker: "+ticker:"\nSource: "+(prices.empty()?std::string("seeded high-frequency simulation"):prices));
        const auto result=rcaviar::run_replay(input,cfg);
        rcaviar::write_results(input,cfg,result,output);
        std::cout << "Wrote " << output << ": " << result.summaries.size() << " models, " << result.forecasts.size()
            << " forecast times, " << result.calibrations.size() << " calibrations\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << "error: " << error.what() << '\n'; return 1; }
}
