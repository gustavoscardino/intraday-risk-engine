#include "rcaviar/model.hpp"
#include "garcht/market_data.hpp"
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc,char** argv) {
    try {
        if (argc<2 || argc>3) throw std::invalid_argument("usage: rcaviar_fit observations.csv [tau]");
        auto number=[](const std::string& value){
            std::size_t used=0; const double x=std::stod(value,&used);
            if (used!=value.size()) throw std::invalid_argument("invalid number");
            return x;
        };
        rcaviar::FitOptions options;
        if (argc==3) options.tau=number(argv[2]);
        const auto csv=garcht::read_csv(argv[1]);
        const int r=csv.column("aggregate_return"), rv=csv.column("realized_variance"), y=csv.column("target_return");
        if (r<0 || rv<0 || y<0) throw std::invalid_argument("expected aggregate_return,realized_variance,target_return");
        std::vector<rcaviar::Observation> observations;
        for (const auto& row:csv.rows) {
            if (row.size()<=static_cast<std::size_t>(std::max(r,std::max(rv,y)))) throw std::invalid_argument("incomplete observation");
            observations.push_back({number(row[static_cast<std::size_t>(r)]),number(row[static_cast<std::size_t>(rv)]),number(row[static_cast<std::size_t>(y)])});
        }
        const auto fitted=rcaviar::fit(observations,options);
        const auto& p=fitted.params;
        std::cout << std::setprecision(17) << "{\"beta\":[" << p.beta0 << ',' << p.beta1 << ',' << p.beta2 << ',' << p.beta3
            << "],\"initial_quantile\":" << fitted.initial_quantile << ",\"last_quantile\":" << fitted.last_quantile
            << ",\"mean_pinball\":" << fitted.mean_loss << ",\"constant_pinball\":" << fitted.constant_loss
            << ",\"evaluations\":" << fitted.evaluations << ",\"converged\":" << (fitted.converged?"true":"false") << "}\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << "error: " << error.what() << '\n'; return 1; }
}
