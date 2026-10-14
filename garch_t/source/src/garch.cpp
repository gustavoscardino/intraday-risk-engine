#include "garcht/garch.hpp"

#include <algorithm>
#include <cmath>

#include "garcht/math.hpp"
#include "garcht/optimize.hpp"

namespace garcht {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();
enum : int { kMu = 0, kOmega = 1, kAlpha = 2, kBeta = 3, kNu = 4, kNat = 5 };

double logistic(double x) {
    if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
    const double e = std::exp(x);
    return e / (1.0 + e);
}

double logit(double p) { return std::log(p / (1.0 - p)); }

// Core likelihood recursion. kT selects the Student-t density; kGrad enables the
// analytic gradient (recursive derivatives of h_t). If opg != nullptr the outer
// product of per-observation scores (5x5, natural parameters) is accumulated.
template <bool kT, bool kGrad>
double loglik_kernel(const double* y, size_t n, const GarchParams& p, double bc, double* grad,
                     double* h_out, double* opg) {
    const double mu = p.mu, om = p.omega, al = p.alpha, be = p.beta, nu = p.nu;
    double c = 0.0, cprime = 0.0, half_nu1 = 0.0, inv_nm2 = 0.0;
    if constexpr (kT) {
        c = math::log_gamma(0.5 * (nu + 1.0)) - math::log_gamma(0.5 * nu) -
            0.5 * std::log(math::kPi * (nu - 2.0));
        if constexpr (kGrad)
            cprime = 0.5 * (math::digamma(0.5 * (nu + 1.0)) - math::digamma(0.5 * nu)) -
                     0.5 / (nu - 2.0);
        half_nu1 = 0.5 * (nu + 1.0);
        inv_nm2 = 1.0 / (nu - 2.0);
    }

    double h = om + (al + be) * bc;
    double dh[4] = {0.0, 1.0, bc, bc};  // dh/d(mu, omega, alpha, beta)
    double g[kNat] = {0.0, 0.0, 0.0, 0.0, 0.0};
    double ll = 0.0;

    for (size_t t = 0; t < n; ++t) {
        if (!(h > 0.0) || !std::isfinite(h)) return -kInf;
        if (h_out) h_out[t] = h;
        const double yt = y[t];
        double e2;
        double de2_mu = 0.0;
        bool observed = std::isfinite(yt);
        if (observed) {
            const double e = yt - mu;
            e2 = e * e;
            const double logh = std::log(h);
            double gh = 0.0, ge = 0.0, gnu = 0.0;
            if constexpr (kT) {
                const double x = e2 * inv_nm2 / h;
                // log(1 + x) rather than log1p: same absolute accuracy (~1e-16 per term) and
                // several times faster with common math libraries.
                const double l1px = std::log(1.0 + x);
                ll += c - 0.5 * logh - half_nu1 * l1px;
                if constexpr (kGrad) {
                    const double r = x / (1.0 + x);
                    gh = 0.5 * ((nu + 1.0) * r - 1.0) / h;
                    ge = -(nu + 1.0) * e * inv_nm2 / (h * (1.0 + x));
                    gnu = cprime - 0.5 * l1px + half_nu1 * r * inv_nm2;
                }
            } else {
                ll += -0.5 * (math::kLn2Pi + logh + e2 / h);
                if constexpr (kGrad) {
                    gh = 0.5 * (e2 / h - 1.0) / h;
                    ge = -e / h;
                }
            }
            if constexpr (kGrad) {
                const double s[kNat] = {gh * dh[0] - ge, gh * dh[1], gh * dh[2], gh * dh[3], gnu};
                for (int i = 0; i < kNat; ++i) g[i] += s[i];
                if (opg)
                    for (int i = 0; i < kNat; ++i)
                        for (int j = 0; j < kNat; ++j) opg[i * kNat + j] += s[i] * s[j];
                de2_mu = -2.0 * e;
            }
        } else {
            e2 = h;  // E[eps_t^2 | F_{t-1}]
        }
        if constexpr (kGrad) {
            double n0, n1, n2, n3;
            if (observed) {
                n0 = al * de2_mu + be * dh[0];
                n1 = 1.0 + be * dh[1];
                n2 = e2 + be * dh[2];
                n3 = h + be * dh[3];
            } else {  // e2 = h, so de2 = dh
                const double ab = al + be;
                n0 = ab * dh[0];
                n1 = 1.0 + ab * dh[1];
                n2 = e2 + ab * dh[2];
                n3 = h + ab * dh[3];
            }
            dh[0] = n0; dh[1] = n1; dh[2] = n2; dh[3] = n3;
        }
        h = om + al * e2 + be * h;
    }
    if constexpr (kGrad) {
        if (grad)
            for (int i = 0; i < kNat; ++i) grad[i] = g[i];
    }
    return std::isfinite(ll) ? ll : -kInf;
}

double loglik_dispatch(const double* y, size_t n, Dist dist, const GarchParams& p, double bc,
                       double* grad, double* h_out, double* opg) {
    if (dist == Dist::StudentT) {
        if (!(p.nu > 2.0) || !std::isfinite(p.nu)) return -kInf;
        return grad ? loglik_kernel<true, true>(y, n, p, bc, grad, h_out, opg)
                    : loglik_kernel<true, false>(y, n, p, bc, nullptr, h_out, nullptr);
    }
    return grad ? loglik_kernel<false, true>(y, n, p, bc, grad, h_out, opg)
                : loglik_kernel<false, false>(y, n, p, bc, nullptr, h_out, nullptr);
}

// Maps unconstrained optimizer variables u <-> GARCH parameters (on standardized data).
//   u = [mu]? [log omega]? logit(P / Pmax) logit(alpha / P) [logit((nu - nu_min)/range)]?
// with P = alpha + beta. All constraints hold for every finite u.
struct Transform {
    bool est_mu, est_omega, est_nu;
    double vbar;  // sample variance for variance targeting
    double pmax, nu_min, nu_range;

    int size() const { return (est_mu ? 1 : 0) + (est_omega ? 1 : 0) + 2 + (est_nu ? 1 : 0); }

    GarchParams to_params(const std::vector<double>& u) const {
        GarchParams p;
        int k = 0;
        p.mu = est_mu ? u[k++] : 0.0;
        if (est_omega) p.omega = std::exp(u[k++]);
        const double P = pmax * logistic(u[k++]);
        const double a = logistic(u[k++]);
        p.alpha = a * P;
        p.beta = (1.0 - a) * P;
        if (!est_omega) p.omega = vbar * (1.0 - P);
        p.nu = est_nu ? nu_min + nu_range * logistic(u[k++]) : kInf;
        return p;
    }

    std::vector<double> from_params(const GarchParams& p) const {
        std::vector<double> u;
        u.reserve(size());
        if (est_mu) u.push_back(p.mu);
        if (est_omega) u.push_back(std::log(std::max(p.omega, 1e-300)));
        const double P = std::clamp(p.alpha + p.beta, 1e-6, pmax * (1.0 - 1e-9));
        u.push_back(logit(P / pmax));
        u.push_back(logit(std::clamp(p.alpha / P, 1e-6, 1.0 - 1e-6)));
        if (est_nu) u.push_back(logit(std::clamp((p.nu - nu_min) / nu_range, 1e-6, 1.0 - 1e-6)));
        return u;
    }

    // gu = dL/du given gnat = dL/d(mu, omega, alpha, beta, nu) with omega free.
    void chain(const std::vector<double>& u, const GarchParams& p, const double* gnat,
               std::vector<double>& gu) const {
        double ga = gnat[kAlpha], gb = gnat[kBeta];
        if (!est_omega) {  // omega = vbar * (1 - alpha - beta)
            ga -= vbar * gnat[kOmega];
            gb -= vbar * gnat[kOmega];
        }
        int k = 0;
        if (est_mu) gu[k++] = gnat[kMu];
        if (est_omega) gu[k++] = gnat[kOmega] * p.omega;
        const double sp = logistic(u[k]);
        const double dP = pmax * sp * (1.0 - sp);
        const double P = pmax * sp;
        const double a = logistic(u[k + 1]);
        gu[k] = (ga * a + gb * (1.0 - a)) * dP;
        gu[k + 1] = (ga - gb) * P * a * (1.0 - a);
        k += 2;
        if (est_nu) {
            const double s = logistic(u[k]);
            gu[k] = gnat[kNu] * nu_range * s * (1.0 - s);
        }
    }
};

// In-place inverse of a symmetric positive-definite k x k matrix (Cholesky).
bool invert_spd(std::vector<double>& A, int k) {
    std::vector<double> L(k * k, 0.0);
    for (int j = 0; j < k; ++j) {
        double d = A[j * k + j];
        for (int m = 0; m < j; ++m) d -= L[j * k + m] * L[j * k + m];
        if (!(d > 0.0) || !std::isfinite(d)) return false;
        L[j * k + j] = std::sqrt(d);
        for (int i = j + 1; i < k; ++i) {
            double s = A[i * k + j];
            for (int m = 0; m < j; ++m) s -= L[i * k + m] * L[j * k + m];
            L[i * k + j] = s / L[j * k + j];
        }
    }
    // inv(L) (lower triangular)
    std::vector<double> Li(k * k, 0.0);
    for (int i = 0; i < k; ++i) {
        Li[i * k + i] = 1.0 / L[i * k + i];
        for (int j = 0; j < i; ++j) {
            double s = 0.0;
            for (int m = j; m < i; ++m) s -= L[i * k + m] * Li[m * k + j];
            Li[i * k + j] = s / L[i * k + i];
        }
    }
    // A^-1 = Li^T Li
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < k; ++j) {
            double s = 0.0;
            for (int m = std::max(i, j); m < k; ++m) s += Li[m * k + i] * Li[m * k + j];
            A[i * k + j] = s;
        }
    return true;
}

// Classic and robust standard errors (on standardized data, natural parameters).
void compute_std_errors(const std::vector<double>& ys, Dist dist, const GarchSpec& spec,
                        const GarchParams& th, double bc, double vbar, double se[kNat],
                        double rse[kNat]) {
    for (int i = 0; i < kNat; ++i) se[i] = rse[i] = kNaN;
    // Free natural parameters phi and the linear map M from natural scores to phi-scores.
    std::vector<int> idx;
    if (spec.mean == MeanModel::Constant) idx.push_back(kMu);
    if (!spec.variance_targeting) idx.push_back(kOmega);
    idx.push_back(kAlpha);
    idx.push_back(kBeta);
    if (dist == Dist::StudentT) idx.push_back(kNu);
    const int k = static_cast<int>(idx.size());
    std::vector<double> M(k * kNat, 0.0);
    for (int r = 0; r < k; ++r) {
        M[r * kNat + idx[r]] = 1.0;
        if (spec.variance_targeting && (idx[r] == kAlpha || idx[r] == kBeta))
            M[r * kNat + kOmega] = -vbar;
    }

    auto set_phi = [&](GarchParams p, int j, double v) {
        double* f[kNat] = {&p.mu, &p.omega, &p.alpha, &p.beta, &p.nu};
        *f[idx[j]] = v;
        if (spec.variance_targeting) p.omega = vbar * (1.0 - p.alpha - p.beta);
        return p;
    };
    // The likelihood is well defined whenever h_t > 0 (stationarity is not needed for
    // finite-difference probes around the optimum).
    auto valid = [&](const GarchParams& p) {
        return p.omega > 0.0 && p.alpha >= 0.0 && p.beta >= 0.0 &&
               (dist != Dist::StudentT || p.nu > 2.0);
    };
    auto phi_grad = [&](const GarchParams& p, std::vector<double>& out) {
        double gn[kNat];
        const double ll = garch_loglik(ys.data(), ys.size(), dist, p, bc, gn, nullptr);
        if (!std::isfinite(ll)) return false;
        for (int r = 0; r < k; ++r) {
            double s = 0.0;
            for (int c = 0; c < kNat; ++c) s += M[r * kNat + c] * gn[c];
            out[r] = s;
        }
        return true;
    };

    const double phi0[kNat] = {th.mu, th.omega, th.alpha, th.beta, th.nu};
    std::vector<double> H(k * k), gp(k), gm(k), g0(k);
    if (!phi_grad(th, g0)) return;
    for (int j = 0; j < k; ++j) {
        const double v = phi0[idx[j]];
        const double d = 1e-5 * std::max(std::fabs(v), 1e-3);
        const GarchParams pp = set_phi(th, j, v + d), pm = set_phi(th, j, v - d);
        const bool okp = valid(pp) && phi_grad(pp, gp);
        const bool okm = valid(pm) && phi_grad(pm, gm);
        for (int i = 0; i < k; ++i) {
            if (okp && okm) H[i * k + j] = (gp[i] - gm[i]) / (2.0 * d);
            else if (okp) H[i * k + j] = (gp[i] - g0[i]) / d;
            else if (okm) H[i * k + j] = (g0[i] - gm[i]) / d;
            else return;
        }
    }
    std::vector<double> negH(k * k);
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < k; ++j) negH[i * k + j] = -0.5 * (H[i * k + j] + H[j * k + i]);
    if (!invert_spd(negH, k)) return;  // negH now holds the classic covariance
    for (int r = 0; r < k; ++r) se[idx[r]] = std::sqrt(negH[r * k + r]);

    // Robust: cov = H^-1 J H^-1, with J = outer product of per-observation scores.
    std::vector<double> opg(kNat * kNat, 0.0);
    double gn[kNat];
    if (dist == Dist::StudentT)
        loglik_kernel<true, true>(ys.data(), ys.size(), th, bc, gn, nullptr, opg.data());
    else
        loglik_kernel<false, true>(ys.data(), ys.size(), th, bc, gn, nullptr, opg.data());
    std::vector<double> J(k * k, 0.0), tmp(k * kNat, 0.0);
    for (int r = 0; r < k; ++r)
        for (int c = 0; c < kNat; ++c) {
            double s = 0.0;
            for (int m = 0; m < kNat; ++m) s += M[r * kNat + m] * opg[m * kNat + c];
            tmp[r * kNat + c] = s;
        }
    for (int r = 0; r < k; ++r)
        for (int c = 0; c < k; ++c) {
            double s = 0.0;
            for (int m = 0; m < kNat; ++m) s += tmp[r * kNat + m] * M[c * kNat + m];
            J[r * k + c] = s;
        }
    for (int r = 0; r < k; ++r) {
        double s = 0.0;  // diag of negH^-1 J negH^-1 (negH holds the inverse already)
        for (int a = 0; a < k; ++a)
            for (int b = 0; b < k; ++b) s += negH[r * k + a] * J[a * k + b] * negH[b * k + r];
        rse[idx[r]] = s >= 0.0 ? std::sqrt(s) : kNaN;
    }
}

}  // namespace

// ============================================================== GarchParams

double GarchParams::unconditional_variance() const {
    const double P = alpha + beta;
    return P < 1.0 ? omega / (1.0 - P) : kInf;
}

double GarchParams::half_life() const {
    const double P = alpha + beta;
    if (P <= 0.0) return 0.0;
    if (P >= 1.0) return kInf;
    return std::log(0.5) / std::log(P);
}

bool GarchParams::is_valid(Dist dist) const {
    if (!(std::isfinite(mu) && omega > 0.0 && std::isfinite(omega) && alpha >= 0.0 &&
          beta >= 0.0 && alpha + beta < 1.0))
        return false;
    return dist != Dist::StudentT || (nu > 2.0 && std::isfinite(nu));
}

int num_free_params(const GarchSpec& spec) {
    return (spec.mean == MeanModel::Constant ? 1 : 0) + (spec.variance_targeting ? 0 : 1) + 2 +
           (spec.dist == Dist::StudentT ? 1 : 0);
}

const char* to_string(FitStatus s) {
    switch (s) {
        case FitStatus::Ok: return "ok";
        case FitStatus::NotConverged: return "not_converged";
        case FitStatus::InsufficientData: return "insufficient_data";
        case FitStatus::DegenerateData: return "degenerate_data";
        case FitStatus::NumericalError: return "numerical_error";
    }
    return "unknown";
}

// ============================================================== likelihood

double garch_backcast(const double* resid, size_t n) {
    double num = 0.0, den = 0.0, w = 1.0;
    int used = 0;
    for (size_t i = 0; i < n && used < 75; ++i) {
        if (!std::isfinite(resid[i])) continue;
        num += w * resid[i] * resid[i];
        den += w;
        w *= 0.94;
        ++used;
    }
    return den > 0.0 ? num / den : kNaN;
}

double garch_loglik(const double* y, size_t n, Dist dist, const GarchParams& p, double backcast,
                    double* grad, double* h_out) {
    if (!(p.omega > 0.0) || p.alpha < 0.0 || p.beta < 0.0 || !(backcast >= 0.0) ||
        !std::isfinite(p.mu))
        return -kInf;
    return loglik_dispatch(y, n, dist, p, backcast, grad, h_out, nullptr);
}

double garch_variance_path(const double* y, size_t n, const GarchParams& p, double backcast,
                           double* h_out) {
    double h = p.omega + (p.alpha + p.beta) * backcast;
    for (size_t t = 0; t < n; ++t) {
        if (h_out) h_out[t] = h;
        const double e = y[t] - p.mu;
        const double e2 = std::isfinite(e) ? e * e : h;
        h = p.omega + p.alpha * e2 + p.beta * h;
    }
    return h;
}

// ============================================================== estimation

FitResult fit_garch(const double* y, size_t n, const GarchSpec& spec, const FitOptions& opt) {
    FitResult res;
    res.n_params = num_free_params(spec);
    const bool est_mu = spec.mean == MeanModel::Constant;
    const bool est_nu = spec.dist == Dist::StudentT;

    // ---- descriptive statistics on finite observations
    size_t nf = 0;
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i)
        if (std::isfinite(y[i])) {
            sum += y[i];
            ++nf;
        }
    res.n_obs = static_cast<int>(nf);
    if (nf < static_cast<size_t>(std::max(opt.min_obs, res.n_params + 2))) {
        res.status = FitStatus::InsufficientData;
        res.message = "not enough finite observations";
        return res;
    }
    const double dnf = static_cast<double>(nf);
    const double mean = sum / dnf;
    double ss_c = 0.0, ss_0 = 0.0;
    for (size_t i = 0; i < n; ++i)
        if (std::isfinite(y[i])) {
            ss_c += (y[i] - mean) * (y[i] - mean);
            ss_0 += y[i] * y[i];
        }
    const double sd = std::sqrt(ss_c / dnf);
    // A series without variation (e.g. a frozen price) carries no volatility information.
    if (!std::isfinite(sd) || !(sd > 1e-10 * std::fabs(mean)) || sd < 1e-150) {
        res.status = FitStatus::DegenerateData;
        res.message = "zero or non-finite variance";
        return res;
    }
    const double center = est_mu ? mean : 0.0;
    const double scale = est_mu ? sd : std::sqrt(ss_0 / dnf);

    // ---- standardize: the optimizer works with data of unit scale
    std::vector<double> ys(n);
    for (size_t i = 0; i < n; ++i) ys[i] = y[i] / scale;
    const double mu0 = center / scale;
    std::vector<double> r0(n);
    double vbar = 0.0;
    for (size_t i = 0; i < n; ++i) {
        r0[i] = ys[i] - mu0;
        if (std::isfinite(r0[i])) vbar += r0[i] * r0[i];
    }
    vbar /= dnf;  // == 1 up to rounding
    const double bc = garch_backcast(r0.data(), n);

    Transform tf{est_mu, !spec.variance_targeting, est_nu, vbar, opt.max_persistence, opt.nu_min,
                 opt.nu_max - opt.nu_min};
    const int k = tf.size();
    const double inv_nf = 1.0 / dnf;

    ObjectiveFn objective = [&](const std::vector<double>& u, std::vector<double>& gu) -> double {
        const GarchParams th = tf.to_params(u);
        double gn[kNat];
        const double ll = loglik_dispatch(ys.data(), n, spec.dist, th, bc, gn, nullptr, nullptr);
        if (!std::isfinite(ll)) return kInf;
        tf.chain(u, th, gn, gu);
        for (double& v : gu) v *= -inv_nf;
        return -ll * inv_nf;
    };
    auto value_only = [&](const GarchParams& th) {
        const double ll = loglik_dispatch(ys.data(), n, spec.dist, th, bc, nullptr, nullptr, nullptr);
        return std::isfinite(ll) ? -ll * inv_nf : kInf;
    };

    // ---- starting values: grid over (alpha, beta) with nu = 8, plus optional warm start
    struct Start {
        double f;
        std::vector<double> u;
    };
    std::vector<Start> grid;
    static constexpr double kAB[][2] = {{0.02, 0.97}, {0.05, 0.93}, {0.08, 0.90},
                                        {0.10, 0.85}, {0.15, 0.80}, {0.05, 0.80},
                                        {0.20, 0.60}, {0.03, 0.80}, {0.10, 0.60}};
    for (const auto& ab : kAB) {
        GarchParams p;
        p.mu = mu0;
        p.alpha = ab[0];
        p.beta = ab[1];
        p.omega = vbar * (1.0 - ab[0] - ab[1]);
        p.nu = est_nu ? 8.0 : kInf;
        std::vector<double> u = tf.from_params(p);
        grid.push_back({value_only(tf.to_params(u)), std::move(u)});
    }
    std::sort(grid.begin(), grid.end(), [](const Start& a, const Start& b) { return a.f < b.f; });

    // ---- BFGS runs; keep the best
    BfgsOptions bo;
    bo.max_iter = opt.max_iter;
    bo.grad_tol = opt.grad_tol;
    BfgsResult best;
    best.f = kInf;
    int total_evals = static_cast<int>(grid.size());
    auto run_from = [&](const std::vector<double>& u0) {
        BfgsResult r = minimize_bfgs(objective, u0, bo);
        total_evals += r.evaluations;
        if (std::isfinite(r.f) &&
            (r.f < best.f - 1e-12 || (!best.converged && r.converged && r.f <= best.f + 1e-12)))
            best = std::move(r);
    };
    bool warm_used = false;
    if (opt.warm_start && opt.warm_start->omega > 0.0 && opt.warm_start->alpha >= 0.0 &&
        opt.warm_start->beta >= 0.0) {
        GarchParams w = *opt.warm_start;
        w.mu = est_mu ? w.mu / scale : 0.0;
        w.omega = spec.variance_targeting ? vbar * (1.0 - w.alpha - w.beta) : w.omega / (scale * scale);
        w.nu = est_nu ? (std::isfinite(w.nu) ? w.nu : 8.0) : kInf;
        const std::vector<double> u = tf.from_params(w);
        if (std::isfinite(value_only(tf.to_params(u)))) {
            run_from(u);
            warm_used = best.converged;
        }
    }
    // Cold start, or safeguard: if an untouched grid point already beats the warm-started
    // optimum, the warm start got stuck in an inferior region, so search from the grid too.
    const int n_grid_runs = warm_used ? 1 : std::max(1, opt.n_starts);
    for (int s = 0; s < n_grid_runs && s < static_cast<int>(grid.size()); ++s) {
        const Start& g = grid[static_cast<size_t>(s)];
        if (!std::isfinite(g.f)) break;
        if (warm_used && !(g.f < best.f - 1e-9)) break;
        run_from(g.u);
    }
    res.evaluations = total_evals;
    if (!std::isfinite(best.f) || static_cast<int>(best.x.size()) != k) {
        res.status = FitStatus::NumericalError;
        res.message = "likelihood could not be evaluated at any starting point";
        return res;
    }

    // ---- back to original units
    const GarchParams ths = tf.to_params(best.x);
    res.params = ths;
    res.params.mu = ths.mu * scale;
    res.params.omega = ths.omega * scale * scale;
    res.loglik = -best.f * static_cast<double>(nf) - static_cast<double>(nf) * std::log(scale);
    res.aic = 2.0 * res.n_params - 2.0 * res.loglik;
    res.bic = res.n_params * std::log(static_cast<double>(nf)) - 2.0 * res.loglik;
    res.backcast = bc * scale * scale;
    res.iterations = best.iterations;
    res.grad_inf_norm = best.grad_inf_norm;
    res.status = best.converged ? FitStatus::Ok : FitStatus::NotConverged;
    res.message = best.converged ? "converged"
                  : best.iterations >= bo.max_iter ? "iteration limit reached"
                                                          : "line search failed before convergence";

    std::vector<double> h(n);
    res.next_variance = garch_variance_path(y, n, res.params, res.backcast, h.data());
    res.last_variance = n > 0 ? h[n - 1] : kNaN;

    if (opt.compute_std_errors) {
        double se[kNat], rse[kNat];
        compute_std_errors(ys, spec.dist, spec, ths, bc, vbar, se, rse);
        const double unit[kNat] = {scale, scale * scale, 1.0, 1.0, 1.0};
        double* out_se[kNat] = {&res.std_err.mu, &res.std_err.omega, &res.std_err.alpha,
                                &res.std_err.beta, &res.std_err.nu};
        double* out_rse[kNat] = {&res.robust_std_err.mu, &res.robust_std_err.omega,
                                 &res.robust_std_err.alpha, &res.robust_std_err.beta,
                                 &res.robust_std_err.nu};
        for (int i = 0; i < kNat; ++i) {
            *out_se[i] = se[i] * unit[i];
            *out_rse[i] = rse[i] * unit[i];
        }
    } else {
        res.std_err = res.robust_std_err = GarchParams{kNaN, kNaN, kNaN, kNaN, kNaN};
    }
    return res;
}

// ============================================================== online filter

void GarchFilter::reset(const GarchParams& p, double next_variance) {
    p_ = p;
    h_ = next_variance;
}

double GarchFilter::update(double y, double z_clip) {
    if (!std::isfinite(y)) {
        update_missing();
        return kNaN;
    }
    const double e = y - p_.mu;
    const double z = e / std::sqrt(h_);
    double e2 = e * e;
    if (z_clip > 0.0 && std::fabs(z) > z_clip) e2 = z_clip * z_clip * h_;
    h_ = p_.omega + p_.alpha * e2 + p_.beta * h_;
    return z;
}

void GarchFilter::update_missing() { h_ = p_.omega + (p_.alpha + p_.beta) * h_; }

double GarchFilter::forecast_variance(int k) const {
    if (k <= 1) return h_;
    const double P = p_.alpha + p_.beta;
    if (std::fabs(1.0 - P) < 1e-12) return h_ + (k - 1) * p_.omega;  // integrated (EWMA)
    const double v = p_.omega / (1.0 - P);
    return v + std::pow(P, k - 1) * (h_ - v);
}

double GarchFilter::cumulative_variance(int k) const {
    if (k <= 0) return 0.0;
    const double P = p_.alpha + p_.beta;
    if (std::fabs(1.0 - P) < 1e-12) return k * h_ + p_.omega * 0.5 * k * (k - 1.0);
    const double v = p_.omega / (1.0 - P);
    return k * v + (h_ - v) * (1.0 - std::pow(P, k)) / (1.0 - P);
}

}  // namespace garcht
