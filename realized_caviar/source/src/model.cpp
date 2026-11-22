#include "rcaviar/model.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace rcaviar {
namespace {
void check_tau(double tau) {
    if (!std::isfinite(tau) || tau <= 0 || tau >= .5) throw std::invalid_argument("tau must be in (0, .5)");
}
void check_features(double r, double rv) {
    if (!std::isfinite(r) || !std::isfinite(rv) || rv < 0) throw std::invalid_argument("features must be finite and RV nonnegative");
}
using Point = std::array<double, 4>;
Params decode(const Point& x, double scale) { return {x[0] * scale, x[1], x[2], x[3]}; }
}
bool Params::valid() const {
    return std::isfinite(beta0) && std::isfinite(beta1) && std::isfinite(beta2) && std::isfinite(beta3)
        && beta0 <= 0 && beta1 >= 0 && beta1 <= .995 && beta2 <= 0 && beta3 <= 0;
}
Filter::Filter(Params p, double q) : params_(p), q_(q) {
    if (!p.valid() || !std::isfinite(q) || q > 0) throw std::invalid_argument("invalid parameters or initial quantile");
}
double Filter::update(double r, double rv) {
    check_features(r, rv);
    const double next = params_.beta0 + params_.beta1*q_ + params_.beta2*std::abs(r) + params_.beta3*std::sqrt(rv);
    if (!std::isfinite(next)) throw std::overflow_error("quantile overflow");
    q_ = next;
    return q_;
}
double pinball(double y, double q, double tau) {
    check_tau(tau);
    if (!std::isfinite(y) || !std::isfinite(q)) throw std::invalid_argument("nonfinite loss input");
    const double error = y-q;
    return error * (tau - (error < 0 ? 1.0 : 0.0));
}
double long_var(double exposure, double q) {
    if (!std::isfinite(exposure) || exposure < 0 || !std::isfinite(q) || q > 0)
        throw std::invalid_argument("VaR requires nonnegative long exposure and lower-tail quantile");
    // Exact log-return to simple-return loss conversion.
    return -exposure * std::expm1(q);
}
FitResult fit(const std::vector<Observation>& rows, const FitOptions& opt) {
    check_tau(opt.tau);
    if (rows.size() < 100 || opt.max_iterations <= 0 || opt.starts <= 0 || !std::isfinite(opt.tolerance) || opt.tolerance <= 0)
        throw std::invalid_argument("fit needs >=100 training observations and positive optimizer settings");
    std::vector<double> ys;
    double squares = 0;
    for (const auto& row : rows) {
        check_features(row.aggregate_return, row.realized_variance);
        if (std::isinf(row.target_return)) throw std::invalid_argument("infinite target");
        if (std::isfinite(row.target_return)) { ys.push_back(row.target_return); squares += row.target_return*row.target_return; }
    }
    if (ys.size()<100) throw std::invalid_argument("fit needs >=100 observed training targets");
    std::sort(ys.begin(), ys.end());
    const double initial = ys[static_cast<std::size_t>(std::floor(opt.tau*static_cast<double>(ys.size()-1)))];
    const double scale = std::sqrt(squares/static_cast<double>(ys.size()));
    if (!(scale > 0) || !std::isfinite(scale) || !(initial < 0)) throw std::invalid_argument("training data need nonzero scale and negative empirical tail");
    FitResult result;
    result.initial_quantile = initial;
    for (double y : ys) result.constant_loss += pinball(y, initial, opt.tau)/static_cast<double>(ys.size());
    auto objective = [&](const Point& x) {
        ++result.evaluations;
        const auto p = decode(x, scale);
        if (!p.valid()) return std::numeric_limits<double>::infinity();
        Filter filter(p, initial);
        double loss = 0;
        try {
            for (const auto& row : rows) {
                const double q=filter.update(row.aggregate_return, row.realized_variance);
                if (std::isfinite(row.target_return)) loss += pinball(row.target_return,q,opt.tau);
            }
        } catch (const std::overflow_error&) { return std::numeric_limits<double>::infinity(); }
        return loss/static_cast<double>(ys.size())/scale;
    };
    // Always retain the empirical constant quantile as a valid candidate.
    Point best{initial/scale, 0, 0, 0};
    double best_loss = objective(best);
    bool any_converged = false;  // reported when no simplex beats the constant candidate
    bool from_simplex = false;
    for (int start = 0; start < opt.starts; ++start) {
        const double persistence = .15 + .8*static_cast<double>(start%6)/5.0;
        const double shock = .02 + .03*static_cast<double>(start%3);
        Point center{initial/scale*(1-persistence), persistence, -shock, -shock};
        struct Vertex { Point x; double loss; };
        std::array<Vertex, 5> simplex;
        simplex[0] = {center, objective(center)};
        for (std::size_t j=0; j<4; ++j) {
            Point x = center; x[j] += j==1 ? -.04 : -.08;
            simplex[j+1] = {x, objective(x)};
        }
        bool converged = false;
        for (int iteration=0; iteration<opt.max_iterations; ++iteration) {
            std::sort(simplex.begin(), simplex.end(), [](const Vertex& a, const Vertex& b) { return a.loss < b.loss; });
            double width = 0;
            for (std::size_t i=1; i<5; ++i) for (std::size_t j=0; j<4; ++j) width = std::max(width, std::abs(simplex[i].x[j]-simplex[0].x[j]));
            if (width < opt.tolerance) { converged = true; break; }
            Point centroid{};
            for (std::size_t i=0; i<4; ++i) for (std::size_t j=0; j<4; ++j) centroid[j] += simplex[i].x[j]/4;
            auto trial = [&](double factor) {
                Point x{};
                for (std::size_t j=0; j<4; ++j) x[j] = centroid[j]+factor*(centroid[j]-simplex[4].x[j]);
                return Vertex{x, objective(x)};
            };
            Vertex reflection = trial(1);
            if (reflection.loss < simplex[0].loss) {
                Vertex expansion = trial(2);
                simplex[4] = expansion.loss < reflection.loss ? expansion : reflection;
            } else if (reflection.loss < simplex[3].loss) simplex[4] = reflection;
            else {
                const bool outside = reflection.loss < simplex[4].loss;
                Vertex contraction = trial(outside ? .5 : -.5);
                if (contraction.loss < (outside ? reflection.loss : simplex[4].loss)) simplex[4] = contraction;
                else for (std::size_t i=1; i<5; ++i) {
                    for (std::size_t j=0; j<4; ++j) simplex[i].x[j] = (simplex[i].x[j]+simplex[0].x[j])/2;
                    simplex[i].loss = objective(simplex[i].x);
                }
            }
        }
        any_converged = any_converged || converged;
        for (const auto& vertex : simplex) if (vertex.loss < best_loss) {
            best = vertex.x; best_loss = vertex.loss; result.converged = converged; from_simplex = true;
        }
    }
    // The constant quantile kept as the best candidate is a valid optimum, not a failed search.
    if (!from_simplex) result.converged = any_converged;
    result.params = decode(best, scale);
    result.mean_loss = best_loss*scale;
    Filter filter(result.params, initial);
    for (const auto& row : rows) filter.update(row.aggregate_return, row.realized_variance);
    result.last_quantile = filter.quantile();
    return result;
}
}
