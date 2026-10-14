#include "garcht/optimize.hpp"

#include <algorithm>
#include <cmath>

namespace garcht {

namespace {

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

double inf_norm(const std::vector<double>& a) {
    double m = 0.0;
    for (double v : a) m = std::max(m, std::fabs(v));
    return m;
}

bool all_finite(const std::vector<double>& a) {
    for (double v : a)
        if (!std::isfinite(v)) return false;
    return true;
}

struct Trial {
    double a = 0.0;      // step length
    double f = 0.0;      // objective
    double slope = 0.0;  // directional derivative g . d
    std::vector<double> g;
    bool ok = false;
};

class LineSearch {
public:
    LineSearch(const ObjectiveFn& fn, const BfgsOptions& opt, const std::vector<double>& x,
               double f0, const std::vector<double>& g0, const std::vector<double>& d, int& evals)
        : fn_(fn), opt_(opt), x_(x), f0_(f0), d_(d), slope0_(dot(g0, d)), evals_(evals),
          x_trial_(x.size()) {
        origin_.a = 0.0;
        origin_.f = f0;
        origin_.slope = slope0_;
        origin_.g = g0;
        origin_.ok = true;
    }

    // Strong-Wolfe line search (Nocedal & Wright, Alg. 3.5). Returns false on failure.
    bool run(double a_init, Trial& out) {
        Trial prev = origin_;
        double a = a_init;
        for (int i = 0; i < opt_.max_line_search; ++i) {
            Trial cur = eval(a);
            if (!cur.ok || cur.f > f0_ + opt_.c1 * a * slope0_ || (i > 0 && cur.f >= prev.f))
                return zoom(prev, cur, out);
            if (std::fabs(cur.slope) <= -opt_.c2 * slope0_) {
                out = std::move(cur);
                return true;
            }
            if (cur.slope >= 0.0) return zoom(cur, prev, out);
            prev = std::move(cur);
            a *= 2.0;
        }
        if (prev.a > 0.0) {
            out = std::move(prev);
            return true;
        }
        return false;
    }

private:
    Trial eval(double a) {
        Trial t;
        t.a = a;
        for (size_t i = 0; i < x_.size(); ++i) x_trial_[i] = x_[i] + a * d_[i];
        t.g.assign(x_.size(), 0.0);
        t.f = fn_(x_trial_, t.g);
        ++evals_;
        t.ok = std::isfinite(t.f) && all_finite(t.g);
        t.slope = t.ok ? dot(t.g, d_) : 0.0;
        return t;
    }

    // Zoom phase (Nocedal & Wright, Alg. 3.6). `lo` satisfies Armijo with the lowest f.
    bool zoom(Trial lo, Trial hi, Trial& out) {
        for (int j = 0; j < opt_.max_line_search; ++j) {
            const double width = hi.a - lo.a;
            double t = 0.5;
            if (hi.ok) {  // minimizer of the quadratic through (lo.f, lo.slope, hi.f)
                const double denom = 2.0 * (hi.f - lo.f - lo.slope * width);
                if (denom > 0.0) t = -lo.slope * width / denom;
            }
            t = std::clamp(t, 0.1, 0.9);
            const double a = lo.a + t * width;
            Trial cur = eval(a);
            if (!cur.ok || cur.f > f0_ + opt_.c1 * a * slope0_ || cur.f >= lo.f) {
                hi = std::move(cur);
            } else {
                if (std::fabs(cur.slope) <= -opt_.c2 * slope0_) {
                    out = std::move(cur);
                    return true;
                }
                if (cur.slope * (hi.a - lo.a) >= 0.0) hi = lo;
                lo = std::move(cur);
            }
            if (std::fabs(hi.a - lo.a) <= 1e-16 * std::max(1.0, std::fabs(lo.a))) break;
        }
        if (lo.a > 0.0 && lo.f < f0_) {  // sufficient decrease holds by construction
            out = std::move(lo);
            return true;
        }
        return false;
    }

    const ObjectiveFn& fn_;
    const BfgsOptions& opt_;
    const std::vector<double>& x_;
    double f0_;
    const std::vector<double>& d_;
    double slope0_;
    int& evals_;
    std::vector<double> x_trial_;
    Trial origin_;
};

}  // namespace

BfgsResult minimize_bfgs(const ObjectiveFn& fn, std::vector<double> x, const BfgsOptions& opt) {
    const size_t n = x.size();
    BfgsResult res;
    std::vector<double> g(n, 0.0), d(n), s(n), y(n), Hy(n);
    std::vector<double> H(n * n, 0.0);
    auto set_identity = [&](double scale) {
        std::fill(H.begin(), H.end(), 0.0);
        for (size_t i = 0; i < n; ++i) H[i * n + i] = scale;
    };
    set_identity(1.0);

    double f = fn(x, g);
    res.evaluations = 1;
    if (!std::isfinite(f) || !all_finite(g)) {
        res.x = std::move(x);
        res.f = f;
        res.grad_inf_norm = HUGE_VAL;
        return res;
    }

    bool fresh_hessian = true;  // H is a (scaled) identity: next step is steepest descent
    int stall = 0;
    int iter = 0;
    for (; iter < opt.max_iter; ++iter) {
        if (inf_norm(g) <= opt.grad_tol) {
            res.converged = true;
            break;
        }
        for (size_t i = 0; i < n; ++i) {
            double acc = 0.0;
            for (size_t j = 0; j < n; ++j) acc -= H[i * n + j] * g[j];
            d[i] = acc;
        }
        if (!(dot(d, g) < 0.0)) {  // not a descent direction: restart
            set_identity(1.0);
            fresh_hessian = true;
            for (size_t i = 0; i < n; ++i) d[i] = -g[i];
        }

        const double a_init =
            (iter == 0) ? std::min(1.0, 1.0 / std::max(1e-12, std::sqrt(dot(g, g)))) : 1.0;
        LineSearch ls(fn, opt, x, f, g, d, res.evaluations);
        Trial step;
        if (!ls.run(a_init, step)) {
            if (fresh_hessian) break;  // even steepest descent failed: we are at a (numerical) minimum
            set_identity(1.0);
            fresh_hessian = true;
            continue;
        }

        for (size_t i = 0; i < n; ++i) {
            s[i] = step.a * d[i];
            y[i] = step.g[i] - g[i];
            x[i] += s[i];
        }
        const double rel_decrease = (f - step.f) / std::max(1.0, std::fabs(f));
        f = step.f;
        g = std::move(step.g);

        const double sy = dot(s, y);
        if (sy > 1e-12 * std::sqrt(dot(s, s) * dot(y, y))) {
            if (fresh_hessian) {  // Shanno-Phua scaling of the initial inverse Hessian
                set_identity(sy / dot(y, y));
                fresh_hessian = false;
            }
            const double rho = 1.0 / sy;
            for (size_t i = 0; i < n; ++i) {
                double acc = 0.0;
                for (size_t j = 0; j < n; ++j) acc += H[i * n + j] * y[j];
                Hy[i] = acc;
            }
            const double yHy = dot(y, Hy);
            const double coef = rho * rho * yHy + rho;
            for (size_t i = 0; i < n; ++i)
                for (size_t j = 0; j < n; ++j)
                    H[i * n + j] += coef * s[i] * s[j] - rho * (Hy[i] * s[j] + s[i] * Hy[j]);
        }

        stall = (rel_decrease < opt.f_rel_tol) ? stall + 1 : 0;
        if (stall >= 5) {
            res.converged = true;
            ++iter;
            break;
        }
    }
    res.iterations = iter;
    res.x = std::move(x);
    res.f = f;
    res.grad_inf_norm = inf_norm(g);
    if (res.grad_inf_norm <= opt.grad_tol) res.converged = true;
    return res;
}

}  // namespace garcht
