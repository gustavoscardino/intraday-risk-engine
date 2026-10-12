// Fits a GARCH(1,1) to one column of a CSV file and prints the result as JSON.
//
//   fit_series <file.csv> [--column NAME] [--dist t|normal] [--mean zero|constant]
//              [--targeting] [--var-confidence 0.99]
//
// Used by tests/reference/crosscheck_arch.py to compare against the Python `arch` package.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli.hpp"
#include "garcht/garcht.hpp"

using namespace garcht;

namespace {

void usage() {
    std::fprintf(stderr,
                 "usage: fit_series <file.csv> [--column NAME] [--dist t|normal] "
                 "[--mean zero|constant] [--targeting] [--var-confidence C]\n");
}

const char* num(double v, char* buf) {  // JSON has no NaN/Inf
    if (std::isfinite(v)) std::snprintf(buf, 32, "%.17g", v);
    else std::snprintf(buf, 32, "null");
    return buf;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 2;
    }
    std::string path = argv[1], column;
    GarchSpec spec;
    double conf = 0.99;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
            return argv[++i];
        };
        try {
            int k = 0;
            bool ok = true;
            if (a == "--column") column = next();
            else if (a == "--dist") {
                ok = (k = cli::choice(a, next(), {"t", "normal"})) >= 0;
                spec.dist = k == 1 ? Dist::Normal : Dist::StudentT;
            } else if (a == "--mean") {
                ok = (k = cli::choice(a, next(), {"zero", "constant"})) >= 0;
                spec.mean = k == 1 ? MeanModel::Constant : MeanModel::Zero;
            } else if (a == "--targeting") spec.variance_targeting = true;
            else if (a == "--var-confidence") ok = cli::to_double(a, next(), conf, 0.5, 0.99999);
            else ok = false;
            if (!ok) {
                usage();
                return 2;
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "%s\n", e.what());
            return 2;
        }
    }

    CsvTable csv;
    try {
        csv = read_csv(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    const int col = column.empty() ? 0 : csv.column(column);
    if (col < 0) {
        std::fprintf(stderr, "error: column '%s' not found\n", column.c_str());
        return 1;
    }
    std::vector<double> y;
    y.reserve(csv.rows.size());
    for (size_t r = 0; r < csv.rows.size(); ++r) {  // '.' or ',' decimals; empty / NA = missing
        const auto& row = csv.rows[r];
        const std::string s = static_cast<size_t>(col) < row.size() ? row[static_cast<size_t>(col)] : "";
        double v = 0.0;
        if (!parse_number(s, v)) {
            std::fprintf(stderr, "error: data row %zu: '%s' is not a number\n", r + 1, s.c_str());
            return 1;
        }
        y.push_back(v);
    }

    const auto t0 = std::chrono::steady_clock::now();
    const FitResult f = fit_garch(y, spec);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const GarchFilter flt(f.params, f.next_variance);
    const TailFactors tf = tail_factors(spec.dist, f.params.nu, conf);
    const RiskMeasure rm = risk_measure(f.params.mu, std::sqrt(f.next_variance), tf);

    char b[6][32];
    std::printf("{\n  \"status\": \"%s\",\n  \"n_obs\": %d,\n  \"iterations\": %d,\n  \"evaluations\": %d,\n",
                to_string(f.status), f.n_obs, f.iterations, f.evaluations);
    std::printf("  \"fit_ms\": %s,\n  \"loglik\": %s,\n  \"aic\": %s,\n  \"bic\": %s,\n", num(ms, b[0]),
                num(f.loglik, b[1]), num(f.aic, b[2]), num(f.bic, b[3]));
    const char* names[5] = {"mu", "omega", "alpha", "beta", "nu"};
    const double vals[5] = {f.params.mu, f.params.omega, f.params.alpha, f.params.beta, f.params.nu};
    const double se[5] = {f.std_err.mu, f.std_err.omega, f.std_err.alpha, f.std_err.beta, f.std_err.nu};
    const double rse[5] = {f.robust_std_err.mu, f.robust_std_err.omega, f.robust_std_err.alpha,
                           f.robust_std_err.beta, f.robust_std_err.nu};
    std::printf("  \"params\": {");
    for (int k = 0; k < 5; ++k) std::printf("%s\"%s\": %s", k ? ", " : "", names[k], num(vals[k], b[0]));
    std::printf("},\n  \"std_err\": {");
    for (int k = 0; k < 5; ++k) std::printf("%s\"%s\": %s", k ? ", " : "", names[k], num(se[k], b[0]));
    std::printf("},\n  \"robust_std_err\": {");
    for (int k = 0; k < 5; ++k) std::printf("%s\"%s\": %s", k ? ", " : "", names[k], num(rse[k], b[0]));
    std::printf("},\n  \"persistence\": %s,\n  \"half_life\": %s,\n", num(f.params.persistence(), b[0]),
                num(f.params.half_life(), b[1]));
    std::printf("  \"next_variance\": %s,\n  \"forecast_variance_10\": %s,\n", num(f.next_variance, b[0]),
                num(flt.forecast_variance(10), b[1]));
    std::printf("  \"var\": {\"confidence\": %s, \"var\": %s, \"es\": %s}\n}\n", num(conf, b[0]),
                num(rm.var, b[1]), num(rm.es, b[2]));
    return f.usable() ? 0 : 1;
}
