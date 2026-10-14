// Unit tests of the EWMA intraday risk library. Reference values of the special functions,
// the VaR/ES factors and the coverage tests come from SciPy (tests/reference/scipy_reference.py).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "ewmar/ewmar.hpp"

using namespace ewmar;

namespace {
constexpr double kInf = std::numeric_limits<double>::infinity();
#include "reference/scipy_reference.inc"

struct TestCase {
    const char* name;
    void (*fn)();
};
std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}
struct Registrar {
    Registrar(const char* n, void (*f)()) { registry().push_back({n, f}); }
};
int g_checks = 0, g_failures = 0;

bool near(double a, double b, double abs_tol, double rel_tol) {
    if (std::isnan(a) || std::isnan(b)) return false;
    if (a == b) return true;
    return std::fabs(a - b) <= abs_tol + rel_tol * std::fabs(b);
}
}  // namespace

#define TEST(name)                                        \
    static void name();                                   \
    static const Registrar registrar_##name(#name, name); \
    static void name()

#define CHECK(cond)                                                         \
    do {                                                                    \
        ++g_checks;                                                         \
        if (!(cond)) {                                                      \
            ++g_failures;                                                   \
            std::printf("    FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                   \
    } while (0)

#define CHECK_NEAR(a, b, abs_tol, rel_tol)                                                                    \
    do {                                                                                                      \
        ++g_checks;                                                                                           \
        const double va_ = (a), vb_ = (b);                                                                    \
        if (!near(va_, vb_, abs_tol, rel_tol)) {                                                              \
            ++g_failures;                                                                                     \
            std::printf("    FAIL %s:%d: %s = %.17g, expected %.17g\n", __FILE__, __LINE__, #a, va_, vb_); \
        }                                                                                                     \
    } while (0)

// ------------------------------------------------------------------ helpers

namespace {

RawRow row(int ymd, int minute, int asset, double open, double close, double volume = 1000.0, size_t line = 0) {
    RawRow r;
    r.time.ymd = ymd;
    r.time.minute = minute;
    r.asset = asset;
    r.open = open;
    r.high = std::max(open, close);
    r.low = std::min(open, close);
    r.close = close;
    r.volume = volume;
    r.line = line;
    return r;
}

// business days from a start date
std::vector<int> business_days(int start, int count) {
    std::vector<int> d;
    for (int x = start; static_cast<int>(d.size()) < count; x = next_day(x))
        if (is_business_day(x)) d.push_back(x);
    return d;
}

// Random-walk market on the B3 grid (10:00 + 15-minute bars), rows in time order.
struct SimMarket {
    FeedFile feed;
    std::vector<int> days;
};
SimMarket simulate_market(size_t n, int sessions, int K, std::uint64_t seed, double bar_vol = 0.002) {
    SimMarket m;
    m.days = business_days(20260302, sessions);
    for (size_t i = 0; i < n; ++i) m.feed.tickers.push_back("A" + std::to_string(i));
    math::Rng rng(seed);
    std::vector<double> px(n, 20.0);
    size_t line = 2;
    for (int d : m.days) {
        for (size_t i = 0; i < n; ++i) px[i] *= std::exp(0.004 * rng.normal());  // overnight gap
        for (int k = 0; k < K; ++k) {
            const double common = rng.normal();
            for (size_t i = 0; i < n; ++i) {
                const double o = px[i];
                px[i] *= std::exp(bar_vol * (0.5 * common + 0.85 * rng.normal()));
                m.feed.rows.push_back(row(d, 600 + 15 * (k + 1), static_cast<int>(i), o, px[i], 1000.0, line++));
            }
        }
    }
    m.feed.lines = m.feed.rows.size();
    return m;
}

}  // namespace

// ================================================================== 1. data

TEST(csv_reader) {
    const CsvTable t = parse_csv("\xEF\xBB\xBF# comment\nticker;price;name\r\nPETR4;38,50;\"Petro; bras\"\r\n\r\nVALE3;1.234,56;\"say \"\"hi\"\"\"\n");
    CHECK(t.header.size() == 3);
    CHECK(t.rows.size() == 2);
    CHECK(t.column("PRICE") == 1);
    CHECK(t.rows[0][2] == "Petro; bras");
    CHECK(t.rows[1][2] == "say \"hi\"");
    double v = 0.0;
    CHECK(parse_number(t.rows[0][1], v) && v == 38.5);
    CHECK(parse_number(t.rows[1][1], v) && v == 1234.56);
}

TEST(numbers) {
    double v = 0.0;
    CHECK(parse_number("1,234.56", v) && v == 1234.56);
    CHECK(parse_number(" -1.5e-3 ", v) && v == -1.5e-3);
    CHECK(parse_number("1.234.567", v) && v == 1234567.0);
    CHECK(parse_number("NA", v) && std::isnan(v));
    CHECK(parse_number("", v) && std::isnan(v));
    CHECK(!parse_number("abc", v));
    CHECK(!parse_number("12x", v));
}

TEST(timestamps) {
    TimePoint t;
    CHECK(parse_time("2026-03-02 10:15:00", t) && t.ymd == 20260302 && t.minute == 615 && t.second == 0);
    CHECK(parse_time("02/03/2026 10:15", t) && t.ymd == 20260302 && t.minute == 615);
    CHECK(parse_time("2026-03-02T13:15:00Z", t) && t.ymd == 20260302 && t.minute == 615);         // UTC -> B3
    CHECK(parse_time("2026-03-02T10:15:00-03:00", t) && t.minute == 615);
    CHECK(parse_time("2026-03-03T01:00:00+00:00", t) && t.ymd == 20260302 && t.minute == 22 * 60);  // day back
    CHECK(parse_time("2026-03-02 10:15:30.250", t) && t.second == 30);
    CHECK(!parse_time("2026-02-30 10:00", t));
    CHECK(!parse_time("2026-03-02 25:00", t));
    CHECK(t.str() == "2026-03-02 10:15:30");
    TimePoint a, b;
    parse_time("2026-03-02 17:00", a);
    parse_time("2026-03-03 10:15", b);
    CHECK(b.key() - a.key() == (24 * 60 - 17 * 60 + 615) * 60LL);
}

TEST(calendar) {
    CHECK(weekday(20260302) == 0);  // Monday
    CHECK(weekday(20261010) == 5);  // Saturday
    CHECK(next_day(20260228) == 20260301);
    CHECK(next_day(20241231) == 20250101);
    CHECK(is_b3_holiday(20260216) && is_b3_holiday(20261120) && !is_b3_holiday(20260302));
    CHECK(!is_business_day(20260217) && !is_business_day(20261011) && is_business_day(20261013));
}

TEST(session_detection) {
    SimMarket m = simulate_market(2, 5, 28, 1);
    const SessionSpec s = detect_session(m.feed.rows);
    CHECK(s.open_minute == 600 && s.bar_minutes == 15 && s.bars_per_day == 28);
    CHECK(s.slot_of(615) == 0 && s.slot_of(1020) == 27 && s.slot_of(620) == -1 && s.slot_of(600) == -1 && s.slot_of(1035) == -1);
}

TEST(return_conventions) {
    // asset 0: full session; asset 1: misses slots 1-2 and the first bar of day 2
    FeedFile f;
    f.tickers = {"A", "B"};
    const int d1 = 20260302, d2 = 20260303;
    f.rows = {row(d1, 615, 0, 10.0, 11.0), row(d1, 615, 1, 20.0, 20.0), row(d1, 630, 0, 11.0, 12.1),
              row(d1, 645, 0, 12.1, 12.1), row(d1, 660, 0, 12.1, 13.31), row(d1, 660, 1, 20.0, 22.0),
              row(d2, 615, 0, 14.0, 14.7), row(d2, 630, 0, 14.7, 14.0), row(d2, 630, 1, 23.0, 23.0)};
    SessionSpec s{600, 15, 4};
    const Panel p = sorted_panel(f, s);
    std::vector<double> r, g;
    panel_returns(p, r, g);
    CHECK(p.bars() == 6);
    CHECK_NEAR(r[0 * 2 + 0], std::log(11.0 / 10.0), 1e-15, 0);  // first bar: close / open
    CHECK_NEAR(r[1 * 2 + 0], std::log(1.1), 1e-15, 0);
    CHECK(std::isnan(r[1 * 2 + 1]));
    CHECK_NEAR(r[3 * 2 + 1], std::log(22.0 / 20.0) / std::sqrt(3.0), 1e-15, 0);  // spans 3 slots
    CHECK_NEAR(r[4 * 2 + 0], std::log(14.7 / 14.0), 1e-15, 0);  // day 2: gap excluded
    CHECK_NEAR(g[4 * 2 + 0], std::log(14.0 / 13.31), 1e-15, 0);  // the gap itself
    CHECK(std::isnan(g[4 * 2 + 1]));                            // B did not open at slot 0
    CHECK_NEAR(r[5 * 2 + 1], std::log(23.0 / 23.0), 1e-15, 0);  // B day 2: own open (no session ref)
}

TEST(sorted_panel_duplicates_and_order) {
    FeedFile f;
    f.tickers = {"A"};
    f.rows = {row(20260302, 630, 0, 10.0, 10.5), row(20260302, 615, 0, 10.0, 10.2), row(20260302, 615, 0, 10.0, 99.0),
              row(20260302, 620, 0, 10.0, 10.0)};
    PanelStats st;
    const Panel p = sorted_panel(f, SessionSpec{600, 15, 28}, &st);
    CHECK(p.bars() == 2 && st.duplicates == 1 && st.off_grid == 1);
    CHECK(p.close[0] == 10.2 && p.close[1] == 10.5);
}

// ================================================================== 2. math

TEST(normal_quantiles) {
    for (const auto& r : kNormPpf) CHECK_NEAR(math::norm_ppf(r.p), r.x, 1e-14, 1e-13);
    CHECK_NEAR(math::norm_cdf(-1.6448536269514729), 0.05, 0, 1e-14);
    CHECK(std::isinf(math::norm_ppf(0.0)) && std::isnan(math::norm_ppf(1.5)));
}

TEST(student_t_distribution) {
    for (const auto& r : kTCdf) {
        CHECK_NEAR(math::t_cdf(r.x, r.nu), r.cdf, 1e-300, 1e-11);
        CHECK_NEAR(math::t_pdf(r.x, r.nu), r.pdf, 1e-300, 1e-12);
    }
    for (const auto& r : kTPpf) CHECK_NEAR(math::t_ppf(r.p, r.nu), r.x, 1e-13, 1e-11);
    // unit-variance t: variance 1 by numerical integration of z^2 f(z)
    double s = 0.0;
    const double h = 0.001;
    for (double z = -200; z < 200; z += h) s += z * z * std::exp(math::std_t_logpdf(z, 5.0)) * h;
    CHECK_NEAR(s, 1.0, 2e-3, 0);
}

TEST(incomplete_functions) {
    for (const auto& r : kChi2) CHECK_NEAR(math::chi2_sf(r.x, r.k), r.sf, 1e-300, 1e-11);
    for (const auto& r : kIncBeta) CHECK_NEAR(math::inc_beta(r.a, r.b, r.x), r.v, 1e-15, 1e-11);
}

TEST(brent_minimizer) {
    const double x = math::brent_minimize([](double v) { return (v - 0.937) * (v - 0.937) + 1.0; }, 0.8, 0.999, 1e-9);
    CHECK_NEAR(x, 0.937, 1e-7, 0);
    double fm = 0.0;
    const double edge = math::brent_minimize([](double v) { return -v; }, 0.8, 0.999, 1e-9, &fm);
    CHECK(edge > 0.9989);  // optimum at the upper bound
}

TEST(random_numbers) {
    math::Rng rng(7);
    double s1 = 0, s2 = 0, t2 = 0;
    const int N = 400000;
    for (int i = 0; i < N; ++i) {
        const double z = rng.normal();
        s1 += z;
        s2 += z * z;
        const double t = rng.student_t(8.0);
        t2 += t * t;
    }
    CHECK_NEAR(s1 / N, 0.0, 0.01, 0);
    CHECK_NEAR(s2 / N, 1.0, 0.01, 0);
    CHECK_NEAR(t2 / N, 8.0 / 6.0, 0.03, 0);  // var of t(8) = nu / (nu - 2)
    math::Rng a(3), b(3);
    CHECK(a.next() == b.next());
}

// ================================================================== 4. backtest

TEST(binomial_and_wilson) {
    for (const auto& r : kBinom) CHECK_NEAR(binomial_cdf(r.k, r.n, r.p), r.cdf, 1e-14, 1e-10);
    std::vector<int> hits(200, 0);
    for (int i : kHitIdx) hits[static_cast<size_t>(i)] = 1;
    double lo, hi;
    wilson_interval(8, 200, 0.95, lo, hi);
    CHECK_NEAR(lo, kWilsonLo, 1e-12, 0);
    CHECK_NEAR(hi, kWilsonHi, 1e-12, 0);
}

TEST(coverage_tests) {
    std::vector<int> hits(200, 0);
    for (int i : kHitIdx) hits[static_cast<size_t>(i)] = 1;
    const CoverageTest c = coverage_test(hits, kHitP);
    CHECK(c.n == 200 && c.exceptions == 8);
    CHECK_NEAR(c.lr_uc, kLrUc, 1e-10, 0);
    CHECK_NEAR(c.p_uc, kPUc, 1e-12, 1e-9);
    CHECK_NEAR(c.lr_ind, kLrInd, 1e-10, 0);
    CHECK_NEAR(c.p_ind, kPInd, 1e-14, 1e-9);
    CHECK_NEAR(c.p_cc, kPCc, 1e-14, 1e-9);
    // traffic light: 4 exceptions in 250 at 99% is green, 7 yellow, 12 red (Basel zones)
    auto zone = [](int k) {
        std::vector<int> h(250, 0);
        for (int i = 0; i < k; ++i) h[static_cast<size_t>(i * 20)] = 1;
        return coverage_test(h, 0.01).zone;
    };
    CHECK(zone(4) == TrafficLight::Green && zone(5) == TrafficLight::Yellow && zone(9) == TrafficLight::Yellow &&
          zone(10) == TrafficLight::Red);
}

TEST(berkowitz_and_es_tests) {
    math::Rng rng(11);
    std::vector<double> pit_ok, pit_bad, x_ok, x_fat;
    std::vector<DistForecast> f;
    for (int t = 0; t < 4000; ++t) {
        const double sigma = 1.0 + 0.5 * std::sin(t * 0.01);
        const double z = rng.normal(), zt = rng.student_t(3.0) / std::sqrt(3.0);  // unit variance
        f.push_back({sigma, kInf});
        x_ok.push_back(sigma * z);
        x_fat.push_back(sigma * zt);
        pit_ok.push_back(forecast_cdf(f.back(), x_ok.back()));
        pit_bad.push_back(forecast_cdf(f.back(), 1.3 * x_ok.back()));
    }
    CHECK(berkowitz_test(pit_ok).p > 0.01);
    CHECK(berkowitz_test(pit_bad).p < 1e-6);
    const EsTest good = acerbi_szekely_z2(x_ok, f, 0.99, 500, 1);
    const EsTest fat = acerbi_szekely_z2(x_fat, f, 0.99, 500, 1);
    CHECK(good.p2 > 0.01);
    CHECK(fat.z2 < good.z2 && fat.p2 < 0.01);  // Normal ES under heavy tails: rejected
}

// ================================================================== 5. EWMA-N

TEST(diurnal_profile_recovery) {
    const int K = 28;
    const size_t n = 5, days = 80;
    std::vector<double> s_true(K);
    for (int k = 0; k < K; ++k) s_true[static_cast<size_t>(k)] = 1.0 + 1.2 * std::pow((k - 13.5) / 13.5, 2);
    double ms = 0;
    for (double v : s_true) ms += v * v;
    for (double& v : s_true) v /= std::sqrt(ms / K);
    math::Rng rng(5);
    std::vector<double> rows;
    std::vector<int> slots;
    for (size_t d = 0; d < days; ++d)
        for (int k = 0; k < K; ++k) {
            slots.push_back(k);
            for (size_t i = 0; i < n; ++i) rows.push_back(0.002 * (1 + 0.3 * i) * s_true[static_cast<size_t>(k)] * rng.normal());
        }
    DiurnalProfile p(K, n, DiurnalMode::Pooled);
    p.estimate(rows, slots);
    double worst = 0;
    for (int k = 0; k < K; ++k) worst = std::max(worst, std::fabs(p.factor(k, 2) / s_true[static_cast<size_t>(k)] - 1));
    CHECK(worst < 0.08);
    double m2 = 0;
    for (double v : p.pooled()) m2 += v * v;
    CHECK_NEAR(m2 / K, 1.0, 1e-12, 0);
    DiurnalProfile off(K, n, DiurnalMode::Off);
    off.estimate(rows, slots);
    CHECK(off.factor(0, 0) == 1.0);
}

TEST(lambda_recovery) {
    // returns whose variance follows exactly an EWMA recursion with lambda = 0.96
    const double lam = 0.96;
    const size_t n = 10, T = 3000;
    math::Rng rng(9);
    std::vector<double> rows(T * n), v(n, 1e-6);
    for (size_t t = 0; t < T; ++t)
        for (size_t i = 0; i < n; ++i) {
            const double y = std::sqrt(v[i]) * rng.normal();
            rows[t * n + i] = y;
            v[i] = lam * v[i] + (1 - lam) * y * y;
            v[i] = std::clamp(v[i], 1e-8, 1e-4);
        }
    std::vector<double> v0(n);
    for (size_t i = 0; i < n; ++i) v0[i] = 1e-6;
    const double est = estimate_lambda(rows, n, 0, T, v0);
    CHECK_NEAR(est, lam, 0.006, 0);
}

TEST(ewma_covariance_psd_with_missing) {
    const size_t n = 6;
    math::Rng rng(21);
    EwmaCovariance S(n);
    std::vector<double> I(n * n, 0.0);
    for (size_t i = 0; i < n; ++i) I[i * n + i] = 1e-4;
    S.set(I);
    std::vector<double> y(n);
    bool psd = true;
    for (int t = 0; t < 3000; ++t) {
        const double m = rng.normal();
        for (size_t i = 0; i < n; ++i) y[i] = rng.uniform() < 0.15 ? kNaN : 0.01 * (0.8 * m + 0.6 * rng.normal());
        S.update(y.data(), 0.94);
        // Cholesky of S: fails if S is not positive (semi)definite
        std::vector<double> L(n * n, 0.0);
        for (size_t i = 0; i < n && psd; ++i)
            for (size_t j = 0; j <= i; ++j) {
                double s = S(i, j);
                for (size_t k = 0; k < j; ++k) s -= L[i * n + k] * L[j * n + k];
                if (i == j) {
                    if (s <= -1e-18) psd = false;
                    L[i * n + i] = std::sqrt(std::max(s, 1e-30));
                } else {
                    L[i * n + j] = s / L[j * n + j];
                }
            }
    }
    CHECK(psd);
    // a fully observed row reproduces the textbook recursion
    EwmaCovariance A(2);
    A.set({1.0, 0.5, 0.5, 2.0});
    const double z[2] = {2.0, -1.0};
    A.update(z, 0.9);
    CHECK_NEAR(A(0, 0), 0.9 + 0.1 * 4.0, 1e-15, 0);
    CHECK_NEAR(A(0, 1), 0.45 + 0.1 * -2.0, 1e-15, 0);
    const double zm[2] = {2.0, kNaN};
    A.set({1.0, 0.5, 0.5, 2.0});
    A.update(zm, 0.9);
    CHECK_NEAR(A(1, 1), 2.0, 0, 0);
    CHECK_NEAR(A(0, 1), 0.5 * std::sqrt(0.9), 1e-15, 0);
}

TEST(engine_forecast_and_undo) {
    // iid returns with a known sigma: after a refit the EWMA-N sigma is close to it
    const size_t n = 3;
    const int K = 28;
    EngineConfig cfg;
    cfg.bars_per_day = K;
    cfg.window_bars = 30 * K;
    cfg.diurnal = DiurnalMode::Off;
    EwmaEngine eng(n, cfg);
    math::Rng rng(4);
    std::vector<double> r(n);
    for (int t = 0; t < 30 * K; ++t) {
        for (size_t i = 0; i < n; ++i) r[i] = 0.001 * rng.normal();
        eng.on_bar(t % K, r.data());
    }
    const double x[3] = {1e6, 0.0, 0.0};
    const RefitReport rep = eng.refit(x);
    CHECK(rep.ready && eng.ready());
    PortfolioRisk pr;
    eng.compute(0, x, pr);
    CHECK_NEAR(pr.normal.sigma, 1e3, 0, 0.25);
    CHECK_NEAR(pr.ref.sigma, 1e3, 0, 0.35);
    CHECK_NEAR(pr.normal.m[2].var / pr.normal.sigma, 2.3263478740408408, 1e-12, 0);
    CHECK(pr.student.m[2].es / pr.student.m[2].var > pr.normal.m[2].es / pr.normal.m[2].var - 1e-9);
    CHECK_NEAR(pr.contrib[0], pr.normal.sigma, 1e-9, 1e-9);
    // undo restores the filters exactly
    const double before = pr.normal.sigma;
    const double shock[3] = {0.05, -0.05, 0.02};
    eng.on_bar(0, shock);
    eng.compute(1, x, pr);
    CHECK(pr.normal.sigma > 2 * before);
    CHECK(eng.undo_last_bar());
    eng.compute(0, x, pr);
    CHECK_NEAR(pr.normal.sigma, before, 0, 1e-14);
    CHECK(!eng.undo_last_bar());
}

// ================================================================== 6. EWMA-t

TEST(student_factors_against_scipy) {
    for (const auto& r : kFactors) {
        const TailFactors f = student_factors(r.nu, r.c);
        CHECK_NEAR(f.var, r.var, 1e-12, 1e-11);
        CHECK_NEAR(f.es, r.es, 1e-10, 1e-9);
    }
}

TEST(nu_recovery) {
    math::Rng rng(13);
    for (const double nu : {4.0, 8.0}) {
        std::vector<double> u;
        for (int i = 0; i < 30000; ++i) u.push_back(rng.student_t(nu) * std::sqrt((nu - 2) / nu));
        CHECK_NEAR(fit_nu(u), nu, 0, 0.12);
    }
    std::vector<double> g;
    for (int i = 0; i < 30000; ++i) g.push_back(rng.normal());
    CHECK(fit_nu(g) > 30.0);  // Normal data: nu goes to the upper end
}

// ================================================================== 7. options

TEST(black_scholes) {
    CHECK_NEAR(bs_price(true, 100, 100, 1, 0.2, 0.05), 10.450583572185565, 1e-10, 0);
    CHECK_NEAR(bs_price(false, 100, 100, 1, 0.2, 0.05), 5.573526022256971, 1e-10, 0);
    CHECK_NEAR(bs_delta(true, 100, 100, 1, 0.2, 0.05), 0.6368306511756191, 1e-12, 0);
    CHECK_NEAR(bs_delta(false, 100, 100, 1, 0.2, 0.05), 0.6368306511756191 - 1, 1e-12, 0);
    // put-call parity
    const double c = bs_price(true, 37, 40, 0.3, 0.35, 0.14), p = bs_price(false, 37, 40, 0.3, 0.35, 0.14);
    CHECK_NEAR(c - p, 37 - 40 * std::exp(-0.14 * 0.3), 1e-12, 0);
    const SessionSpec s{600, 15, 28};
    // Friday 2026-11-13 at the open, expiry Friday 2026-11-20 (holiday): Mon-Thu = 4 days + today
    CHECK_NEAR(years_to_expiry(20261113, 600, 20261120, s) * 252, 1.0 + 4.0, 1e-12, 0);
    CHECK(years_to_expiry(20261121, 600, 20261120, s) == 0.0);
}

// ================================================================== 8. data quality

TEST(dq_arrival_rules) {
    FeedFile f;
    f.tickers = {"A", "B"};
    const int d = 20260302;
    size_t L = 2;
    f.rows = {row(d, 615, 0, 10, 10.0, 1e3, L++), row(d, 615, 1, 20, 20.0, 1e3, L++),
              row(d, 615, 0, 10, 10.0, 1e3, L++),  // duplicate: discarded
              row(d, 630, 0, 10, 10.1, 1e3, L++), row(d, 645, 1, 20, 20.2, 1e3, L++),
              row(d, 630, 1, 20, 20.1, 1e3, L++),  // older than B's watermark: out of order, accepted
              row(d, 645, 0, 10, 10.2, 1e3, L++), row(d, 660, 0, 10, 10.3, 1e3, L++), row(d, 675, 0, 10, 10.4, 1e3, L++),
              row(d, 660, 1, 20, 20.3, 1e3, L++),  // one bar behind the feed: in time
              row(d, 690, 0, 10, 10.5, 1e3, L++), row(d, 705, 0, 10, 10.6, 1e3, L++),
              row(d, 675, 1, 20, 20.4, 1e3, L++),  // two bars behind: the bar has closed, discarded
              row(d, 690, 1, 20, 20.5, 1e3, L++), row(d, 705, 1, 20, 20.6, 1e3, L++)};
    const SessionSpec s{600, 15, 28};
    const IngestResult g = ingest_feed(f, s, DqConfig{});
    CHECK(g.duplicates == 1 && g.out_of_order == 1 && g.delayed == 1);
    CHECK(g.panel.bars() == 7);
    CHECK(g.flags[1 * 2 + 1] & kOutOfOrder);
    CHECK(std::isfinite(g.panel.close[1 * 2 + 1]));
    CHECK(!(g.flags[3 * 2 + 1] & (kOutOfOrder | kDelay | kGap)));
    CHECK((g.flags[4 * 2 + 1] & kGap) && (g.flags[4 * 2 + 1] & kDelay));
    CHECK(std::isnan(g.panel.close[4 * 2 + 1]));
    CHECK(g.flags[0] & kDuplicate);
}

TEST(dq_stale_rule) {
    FeedFile f;
    f.tickers = {"FROZEN", "CHEAP"};
    const int d = 20260302;
    for (int k = 0; k < 8; ++k) {
        const bool frozen = k >= 2 && k <= 6;
        f.rows.push_back(row(d, 615 + 15 * k, 0, 10, frozen ? 10.5 : 10.0 + 0.5 * (k == 1) + 0.01 * k, frozen ? 0.0 : 1e3));
        f.rows.push_back(row(d, 615 + 15 * k, 1, 1.0, 1.0, 5e4));  // price does not move, but it trades
    }
    // bar 1 closes at 10.51; bars 2-6 frozen at 10.5 with volume 0
    f.rows[2].close = 10.5;
    const IngestResult g = ingest_feed(f, SessionSpec{600, 15, 28}, DqConfig{});
    // run of unchanged zero-volume closes starts at bar 2 (equal to bar 1's close)
    int stale = 0;
    for (int k = 0; k < 8; ++k) stale += (g.flags[static_cast<size_t>(k) * 2] & kStale) != 0;
    CHECK(stale == 5);
    CHECK(std::isfinite(g.panel.close[2 * 2]) && std::isfinite(g.panel.close[3 * 2]));  // retroactive: already used
    CHECK(std::isnan(g.panel.close[4 * 2]) && std::isnan(g.panel.close[6 * 2]));       // masked from detection on
    for (int k = 0; k < 8; ++k) CHECK(!(g.flags[static_cast<size_t>(k) * 2 + 1] & kStale));
}

TEST(dq_score_and_modes) {
    DqConfig cfg;
    const std::uint16_t fl[3] = {0, kGap, kOutOfOrder};
    const double x[3] = {600.0, 300.0, -100.0};
    CHECK_NEAR(dq_score(fl, x, 3, cfg), 0.6 + 0.0 + 0.1 * 0.9, 1e-12, 0);
    const double zero[3] = {0, 0, 0};
    CHECK_NEAR(dq_score(fl, zero, 3, cfg), (1.0 + 0.0 + 0.9) / 3.0, 1e-12, 0);
    ModeController m(cfg);
    CHECK(m.update(0.95) == DqMode::Normal);
    CHECK(m.update(0.5) == DqMode::Freeze);  // worse: immediately
    CHECK(m.update(0.95) == DqMode::Freeze);
    CHECK(m.update(0.95) == DqMode::Freeze);
    CHECK(m.update(0.7) == DqMode::Degraded);  // 3 better bars, the worst of them DEGRADED
    CHECK(m.update(0.95) == DqMode::Degraded);
    CHECK(m.update(0.95) == DqMode::Degraded);
    CHECK(m.update(0.95) == DqMode::Normal);
}

TEST(dq_spike_rule_in_replay) {
    // bad tick on A0 (reverted next bar) and a real -10% jump on A1 (not reverted)
    SimMarket m = simulate_market(3, 26, 28, 77);
    const int bad_day = m.days[23], jump_day = m.days[24];
    double jump_mult = 1.0;
    for (RawRow& r : m.feed.rows) {
        if (r.time.ymd == bad_day && r.time.minute == 600 + 15 * 11 && r.asset == 0) r.close *= 1.08;
        if (r.asset == 1 && r.time.ymd == jump_day && r.time.minute == 600 + 15 * 6) jump_mult = 0.9;
        if (r.asset == 1 && jump_mult != 1.0) {
            if (!(r.time.ymd == jump_day && r.time.minute == 600 + 15 * 6)) r.open *= jump_mult;
            r.close *= jump_mult;
        }
    }
    ReplayInput in;
    in.tickers = m.feed.tickers;
    in.feed = m.feed;
    in.initial_quantity = {1000, 1000, 1000};
    ReplayConfig cfg;
    cfg.window_days = 20;
    const ReplayResult res = run_replay(in, cfg);
    CHECK(res.spikes == 1);
    CHECK(res.shocks_kept >= 1);
    TimePoint tb, tj;
    tb.ymd = bad_day;
    tb.minute = 600 + 15 * 11;
    tj.ymd = jump_day;
    tj.minute = 600 + 15 * 6;
    const auto bad = res.ingest.cell_flags.find(cell_key(tb.key(), 0));
    const auto jump = res.ingest.cell_flags.find(cell_key(tj.key(), 1));
    CHECK(bad != res.ingest.cell_flags.end() && (bad->second & kSpike));
    CHECK(jump != res.ingest.cell_flags.end() && (jump->second & kShockKept) && !(jump->second & kSpike));
    // the bad tick did not reach the realized P&L of its bar
    for (const ForecastRecord& f : res.forecasts)
        if (f.target_time == tb.str()) CHECK(std::fabs(f.pnl) < 0.05 * 1000 * 25);

    // without the DQ layer nothing is flagged, and the forecasts start at the same bar
    cfg.dq = false;
    const ReplayResult raw = run_replay(in, cfg);
    CHECK(raw.spikes == 0 && raw.suspects == 0);
    CHECK(!raw.forecasts.empty() && raw.forecasts.front().forecast_time == res.forecasts.front().forecast_time);
}

TEST(replay_first_forecast_and_window) {
    SimMarket m = simulate_market(2, 25, 28, 3);
    ReplayInput in;
    in.tickers = m.feed.tickers;
    in.feed = m.feed;
    in.initial_quantity = {100, -50};
    ReplayConfig cfg;
    cfg.window_days = 20;
    const ReplayResult res = run_replay(in, cfg);
    // first forecast: the close of session 20 (end of the window); 5 sessions of 28 bars scored
    CHECK(res.sessions == 25);
    TimePoint end_of_window;
    end_of_window.ymd = m.days[19];
    CHECK(res.forecasts.front().forecast_time.substr(0, 10) == end_of_window.str().substr(0, 10));
    CHECK(res.forecasts.size() == 5 * 28);
    CHECK(res.overnight.size() == 5);
    CHECK(res.daily.size() == 6);
    cfg.window_days = 25;
    bool threw = false;
    try {
        run_replay(in, cfg);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
}

// ------------------------------------------------------------------ runner

int main() {
    int failed_tests = 0;
    for (const TestCase& t : registry()) {
        const int before = g_failures;
        std::printf("[ RUN  ] %s\n", t.name);
        try {
            t.fn();
        } catch (const std::exception& e) {
            ++g_failures;
            std::printf("    FAIL exception: %s\n", e.what());
        }
        const bool ok = g_failures == before;
        failed_tests += !ok;
        std::printf("[ %s ] %s\n", ok ? " OK " : "FAIL", t.name);
    }
    std::printf("\n%zu tests, %d checks, %d failed checks, %d failed tests\n", registry().size(), g_checks, g_failures,
                failed_tests);
    return failed_tests == 0 ? 0 : 1;
}
