// Quasi-Newton (BFGS) minimizer with a strong-Wolfe line search.
//
// Designed for small, smooth problems (<= ~10 parameters) such as GARCH
// maximum likelihood. The objective may return +inf / NaN outside its domain;
// the line search treats such points as "too far" and backtracks.
#pragma once

#include <functional>
#include <vector>

namespace garcht {

// Evaluates f(x) and writes the gradient into g (same size as x).
using ObjectiveFn = std::function<double(const std::vector<double>& x, std::vector<double>& g)>;

struct BfgsOptions {
    int max_iter = 400;
    double grad_tol = 1e-8;   // stop when ||g||_inf <= grad_tol
    double f_rel_tol = 1e-14; // ... or relative decrease below this for 5 iterations
    double c1 = 1e-4;         // Armijo constant
    double c2 = 0.9;          // curvature constant
    int max_line_search = 40;
};

struct BfgsResult {
    std::vector<double> x;
    double f = 0.0;
    double grad_inf_norm = 0.0;
    int iterations = 0;
    int evaluations = 0;
    bool converged = false;
};

BfgsResult minimize_bfgs(const ObjectiveFn& fn, std::vector<double> x0, const BfgsOptions& opt = {});

}  // namespace garcht
