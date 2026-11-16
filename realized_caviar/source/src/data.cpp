#include "rcaviar/data.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
namespace rcaviar {
std::vector<Price> read_prices(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open prices: " + path);
    std::string line;
    if (!std::getline(file, line)) throw std::invalid_argument("empty price CSV");
    if (!line.empty() && line.back()=='\r') line.pop_back();
    if (line != "timestamp,price") throw std::invalid_argument("CSV header must be timestamp,price");
    std::vector<Price> prices;
    std::size_t row = 1;
    while (std::getline(file, line)) {
        ++row;
        if (!line.empty() && line.back()=='\r') line.pop_back();
        const auto comma = line.find(',');
        if (comma == std::string::npos || line.find(',', comma+1)!=std::string::npos) throw std::invalid_argument("invalid CSV row " + std::to_string(row));
        std::size_t used_time=0, used_price=0;
        const auto time_text=line.substr(0,comma), price_text=line.substr(comma+1);
        const auto timestamp = std::stoll(time_text, &used_time);
        const auto price = std::stod(price_text, &used_price);
        if (used_time!=time_text.size() || used_price!=price_text.size()) throw std::invalid_argument("invalid CSV value at row " + std::to_string(row));
        prices.push_back({timestamp,price});
    }
    return prices;
}
std::vector<Price> synthetic_prices(std::size_t samples, unsigned seed) {
    std::mt19937 generator(seed);
    std::student_t_distribution<double> shocks(7);
    std::vector<Price> prices{{0,100}};
    double variance = 1e-8, price = 100;
    for (std::size_t i=1; i<=samples; ++i) {
        const double r = std::sqrt(variance)*shocks(generator)*std::sqrt(5.0/7.0);
        price *= std::exp(r);
        variance = 2e-10 + .07*r*r + .91*variance;
        prices.push_back({static_cast<std::int64_t>(i)*5,price});
    }
    return prices;
}
void StreamingWindows::begin_session() {
    previous_.reset(); count_=cursor_=0; aggregate_=rv_=0; returns_.fill(0);
}
std::optional<Window> StreamingWindows::push(Price p) {
    if (p.timestamp<0 || p.timestamp>std::numeric_limits<std::int64_t>::max()-300 || !std::isfinite(p.price) || p.price<=0)
        throw std::invalid_argument("invalid timestamp or price");
    if (!previous_) { previous_=p; return std::nullopt; }
    if (p.timestamp<=previous_->timestamp) throw std::invalid_argument("duplicate or out-of-order timestamp");
    if (p.timestamp-previous_->timestamp!=5) throw std::invalid_argument("missing/stale sample: expected 5-second spacing");
    const double r=std::log(p.price)-std::log(previous_->price);
    const double old=returns_[cursor_];
    aggregate_+=r-old; rv_+=r*r-old*old;
    returns_[cursor_]=r; cursor_=(cursor_+1)%returns_.size(); ++count_; previous_=p;
    if (count_<60 || (count_-60)%12!=0) return std::nullopt;
    return Window{p.timestamp,p.timestamp+300,{aggregate_,std::max(0.0,rv_),std::nan("")}};
}
std::vector<Window> make_windows(const std::vector<Price>& prices, bool reset_on_gap) {
    if (prices.size()<61) throw std::invalid_argument("need at least 61 prices");
    StreamingWindows stream;
    std::vector<Window> windows;
    std::vector<std::size_t> sessions(prices.size());
    for (std::size_t i=0;i<prices.size();++i) {
        if (i) {
            if (prices[i].timestamp<=prices[i-1].timestamp) throw std::invalid_argument("duplicate or out-of-order timestamp");
            sessions[i]=sessions[i-1];
            if (reset_on_gap && prices[i].timestamp-prices[i-1].timestamp!=5) {
                stream.begin_session(); ++sessions[i];
            }
        }
        if (auto window=stream.push(prices[i])) windows.push_back(*window);
    }
    for (auto& w:windows) {
        auto before=std::lower_bound(prices.begin(),prices.end(),w.forecast_time,[](Price p,std::int64_t t){return p.timestamp<t;});
        auto after=std::lower_bound(before,prices.end(),w.target_time,[](Price p,std::int64_t t){return p.timestamp<t;});
        if (after!=prices.end() && after->timestamp==w.target_time && sessions[static_cast<std::size_t>(after-prices.begin())]==sessions[static_cast<std::size_t>(before-prices.begin())])
            w.observation.target_return=std::log(after->price)-std::log(before->price);
    }
    return windows;
}
}
