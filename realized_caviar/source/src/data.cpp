#include "rcaviar/data.hpp"
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
std::vector<Window> make_windows(const std::vector<Price>& prices) {
    if (prices.size()<121) throw std::invalid_argument("need at least 121 prices");
    std::vector<double> returns(prices.size(),0);
    for (std::size_t i=0; i<prices.size(); ++i) {
        const auto& p = prices[i];
        if (p.timestamp<0 || p.timestamp>std::numeric_limits<std::int64_t>::max()-300 || !std::isfinite(p.price) || p.price<=0)
            throw std::invalid_argument("invalid timestamp or price at sample " + std::to_string(i));
        if (i) {
            if (p.timestamp<=prices[i-1].timestamp) throw std::invalid_argument("duplicate or out-of-order timestamp");
            if (p.timestamp-prices[i-1].timestamp!=5) throw std::invalid_argument("missing/stale sample: expected 5-second spacing (split sessions into separate files)");
            returns[i] = std::log(p.price)-std::log(prices[i-1].price);
        }
    }
    std::vector<Window> windows;
    for (std::size_t i=60; i+60<prices.size(); i+=12) {
        double r=0,rv=0,target=0;
        for (std::size_t j=i-59; j<=i; ++j) { r+=returns[j]; rv+=returns[j]*returns[j]; }
        for (std::size_t j=i+1; j<=i+60; ++j) target+=returns[j];
        windows.push_back({prices[i].timestamp,prices[i+60].timestamp,{r,rv,target}});
    }
    return windows;
}
}
