#include "rcaviar/shared_data.hpp"
#include "garcht/math.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <stdexcept>

namespace rcaviar {
std::vector<Window> make_bar_windows(const garcht::BarPanel& panel, std::size_t asset) {
    if (asset>=panel.num_assets()) throw std::invalid_argument("asset index out of range");
    const auto session=garcht::detect_session(panel);
    if (session.bar_minutes!=15) throw std::invalid_argument("shared-data mode requires 15-minute bars");
    std::vector<double> returns;
    std::vector<int> slots;
    garcht::bar_returns(panel,session,returns,slots);
    auto valid_close=[&](std::size_t i) { const double p=panel.at(panel.close,i,asset); return std::isfinite(p) && p>0; };
    std::vector<Window> windows;
    for (std::size_t i=3; i<panel.num_bars(); ++i) {
        bool valid=true;
        double aggregate=0,rv=0;
        for (std::size_t j=i-3; j<=i; ++j) {
            const double r=returns[j*panel.num_assets()+asset];
            if (panel.date[j]!=panel.date[i] || slots[j]<0 || !std::isfinite(r) || !valid_close(j)) valid=false;
            if (j>i-3 && (slots[j]!=slots[j-1]+1 || panel.time_key[j]-panel.time_key[j-1]!=900000)) valid=false;
            // Exclude returns spanning a missing price, including one just
            // before the feature window. Slot zero uses the session open.
            if (slots[j]>0 && (j==0 || panel.date[j-1]!=panel.date[j] || slots[j-1]!=slots[j]-1 || !valid_close(j-1))) valid=false;
            if (j<=i) { aggregate+=r; rv+=r*r; }
        }
        if (!valid || slots[i]>=session.bars_per_day-1) continue;
        // Forecast existence depends on observed features, never future data.
        // A missing next bar has an unscorable target but still updates q.
        double target=std::nan("");
        if (i+1<panel.num_bars() && panel.date[i+1]==panel.date[i] && slots[i+1]==slots[i]+1
            && panel.time_key[i+1]-panel.time_key[i]==900000 && valid_close(i+1))
            target=returns[(i+1)*panel.num_assets()+asset];
        windows.push_back({panel.time_key[i]/1000,panel.time_key[i]/1000+900,{aggregate,rv,target}});
    }
    return windows;
}
SharedData read_shared_data(const std::string& folder, const std::string& ticker, int window_days, const std::string& prices_path) {
    if (window_days<0) throw std::invalid_argument("window-days must be positive");
    const auto csv=garcht::read_csv((std::filesystem::path(folder)/"assets.csv").string());
    const int column=csv.column("ticker");
    if (column<0) throw std::invalid_argument("assets.csv needs ticker column");
    auto upper=[](std::string s) { for (char& c:s) c=static_cast<char>(std::toupper(static_cast<unsigned char>(c))); return s; };
    std::vector<std::string> tickers;
    for (const auto& row:csv.rows) {
        if (static_cast<std::size_t>(column)>=row.size()) throw std::invalid_argument("incomplete asset row");
        tickers.push_back(upper(row[static_cast<std::size_t>(column)]));
    }
    const auto found=std::find(tickers.begin(),tickers.end(),upper(ticker));
    if (found==tickers.end()) throw std::invalid_argument("ticker absent from assets.csv: "+ticker);
    const auto asset=static_cast<std::size_t>(found-tickers.begin());
    const auto panel=garcht::load_bars_csv(prices_path.empty()?(std::filesystem::path(folder)/"prices_15m.csv").string():prices_path,tickers);
    std::vector<int> dates;
    for (int date:panel.date) if (dates.empty() || date!=dates.back()) dates.push_back(date);
    const auto days=window_days==0?std::min<std::size_t>(60,dates.size()/2):static_cast<std::size_t>(window_days);
    if (days==0 || days>=dates.size()) throw std::invalid_argument("training window must leave out-of-sample sessions");
    std::int64_t cutoff=0;
    for (std::size_t i=0;i<panel.num_bars();++i) if (panel.date[i]==dates[days-1]) cutoff=panel.time_key[i]/1000;
    auto windows=make_bar_windows(panel,asset);
    if (windows.empty()) throw std::invalid_argument("no complete intraday windows for ticker");
    SharedData result{std::move(windows),cutoff,dates.size(),days,panel.bad_rows,panel.duplicates,{},{},{}};
    std::vector<double> returns; std::vector<int> slots;
    garcht::bar_returns(panel,garcht::detect_session(panel),returns,slots);
    for (std::size_t i=0;i<panel.num_bars();++i) {
        const auto time=panel.time_key[i]/1000;
        if (i==0 || panel.date[i]!=panel.date[i-1]) result.session_starts.push_back(time);
        else for (auto missing=panel.time_key[i-1]/1000+900;missing<time;missing+=900) {
            result.return_times.push_back(missing); result.returns.push_back(std::nan(""));
        }
        double r=returns[i*panel.num_assets()+asset];
        // A return spanning an absent close is not a single-bar observation.
        if (slots[i]<0 || (slots[i]>0 && (i==0 || panel.date[i-1]!=panel.date[i] || slots[i-1]+1!=slots[i]
            || !std::isfinite(panel.at(panel.close,i-1,asset))))) r=std::nan("");
        result.return_times.push_back(time); result.returns.push_back(r);
    }
    return result;
}
garcht::CoverageTest coverage_with_boundaries(const std::vector<int>& hits,
                                            const std::vector<bool>& adjacent, double tau) {
    if (hits.size()!=adjacent.size() || !std::isfinite(tau) || tau<=0 || tau>=1)
        throw std::invalid_argument("invalid coverage inputs");
    auto result=garcht::coverage_test(hits,tau);
    double counts[2][2]{};
    for (std::size_t i=1;i<hits.size();++i) if (adjacent[i]) ++counts[hits[i-1]!=0][hits[i]!=0];
    const double n0=counts[0][0]+counts[0][1], n1=counts[1][0]+counts[1][1];
    if (n0+n1==0) { result.lr_ind=result.p_ind=result.lr_cc=result.p_cc=std::nan(""); return result; }
    const double p0=n0>0?counts[0][1]/n0:0, p1=n1>0?counts[1][1]/n1:0;
    const double p=(counts[0][1]+counts[1][1])/(n0+n1);
    auto xlog=[](double n,double probability) { return n==0?0:n*std::log(probability); };
    const double null=xlog(counts[0][0]+counts[1][0],1-p)+xlog(counts[0][1]+counts[1][1],p);
    const double alternate=xlog(counts[0][0],1-p0)+xlog(counts[0][1],p0)+xlog(counts[1][0],1-p1)+xlog(counts[1][1],p1);
    result.lr_ind=std::max(0.0,2*(alternate-null));
    result.p_ind=garcht::math::chi2_sf(result.lr_ind,1);
    result.lr_cc=result.lr_uc+result.lr_ind;
    result.p_cc=garcht::math::chi2_sf(result.lr_cc,2);
    return result;
}
}
