#include "garcht/synthetic.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <stdexcept>

#include "garcht/random.hpp"

namespace garcht {

namespace {

constexpr int kBarsPerDay = 28;  // 10:00 -> 17:00
constexpr int kOpenMinute = 600;
constexpr int kBarMinutes = 15;
constexpr int kSteps = 15;  // 1-minute steps inside a bar (high/low via Brownian bridge)

struct AssetDef {
    const char* ticker;
    const char* name;
    const char* sector;
    int sector_id;
    double price0;
    double beta;        // loading on the market factor
    double gamma;       // loading on the sector factor
    double idio_daily;  // idiosyncratic daily volatility
    double alpha, persistence, nu;
    double base_volume;
    long qty0;  // initial position (shares, negative = short)
};

// Fictitious companies (names and tickers invented for this project). XVAR3, XTEL3 and
// XIMB3 trade at low prices, where the R$ 0.01 tick is a large fraction of a 15-min move.
const AssetDef kAssets[] = {
    {"XPET3", "Austral Petroleum", "Energy", 0, 38.50, 1.15, 1.00, 0.0130, 0.07, 0.975, 5.0, 420000, 40000},
    {"XELE3", "Blue Ridge Power", "Energy", 0, 42.10, 0.60, 0.70, 0.0090, 0.05, 0.980, 8.0, 150000, 20000},
    {"XBAN4", "Horizon Bank", "Financials", 1, 31.20, 1.05, 1.00, 0.0100, 0.06, 0.980, 6.0, 380000, 50000},
    {"XSEG3", "Watchtower Insurance", "Financials", 1, 27.80, 0.80, 0.80, 0.0110, 0.05, 0.970, 9.0, 120000, 15000},
    {"XMIN3", "Golden Valley Mining", "Materials", 2, 64.30, 1.20, 1.00, 0.0140, 0.08, 0.975, 5.0, 300000, 25000},
    {"XSID3", "North Iron Steel", "Materials", 2, 18.90, 1.35, 1.10, 0.0180, 0.09, 0.970, 4.5, 260000, 30000},
    {"XVAR3", "Good Price Retail", "Consumer", 3, 5.90, 1.40, 0.90, 0.0200, 0.10, 0.965, 4.0, 900000, 156000},
    {"XCON3", "Southern Foods", "Consumer", 3, 22.60, 0.75, 0.80, 0.0095, 0.05, 0.985, 8.0, 140000, 25000},
    {"XTEC3", "Tecnova Systems", "Technology", 4, 48.70, 1.30, 1.10, 0.0170, 0.08, 0.970, 5.0, 220000, -15000},
    {"XTEL3", "TeleSouth Communications", "Technology", 4, 3.20, 0.65, 0.60, 0.0090, 0.04, 0.985, 10.0, 960000, 180000},
    {"XAGR3", "AgroFields", "Materials", 2, 25.90, 0.85, 0.70, 0.0120, 0.06, 0.975, 6.0, 110000, 20000},
    {"XPAP3", "Ipe Pulp and Paper", "Materials", 2, 52.40, 0.90, 0.80, 0.0110, 0.05, 0.980, 7.0, 130000, 10000},
    {"XSAU3", "Integral Health", "Consumer", 3, 33.60, 0.95, 0.60, 0.0130, 0.06, 0.975, 6.0, 170000, 0},
    {"XIMB3", "Concrete Realty", "Consumer", 3, 4.80, 1.25, 0.90, 0.0160, 0.08, 0.970, 5.0, 640000, 118000},
    {"XLOG3", "LogSouth Transport", "Technology", 4, 21.50, 1.10, 0.80, 0.0140, 0.07, 0.975, 6.0, 150000, 20000},
};
constexpr int kN = sizeof(kAssets) / sizeof(kAssets[0]);
constexpr int kSectors = 5;

// GJR-GARCH(1,1)-t on the deseasonalized scale (gamma = 0: plain GARCH).
struct Vol {
    double omega, alpha, gamma, beta, nu, h;
    static Vol make(double daily_vol, double alpha, double gamma, double persistence, double nu) {
        const double v = daily_vol * daily_vol / kBarsPerDay;  // unconditional bar variance
        return {v * (1.0 - persistence), alpha, gamma, persistence - alpha - 0.5 * gamma, nu, v};
    }
    double draw(Rng& rng) const { return std::sqrt(h) * rng.std_t(nu); }
    void update(double e, double omega_mult = 1.0) {
        h = omega * omega_mult + (alpha + (e < 0.0 ? gamma : 0.0)) * e * e + beta * h;
    }
};

struct Date {
    int y, m, d;
    int ymd() const { return y * 10000 + m * 100 + d; }
};
bool is_leap(int y) { return (y % 4 == 0 && y % 100 != 0) || y % 400 == 0; }
int days_in_month(int y, int m) {
    static const int dm[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    return m == 2 && is_leap(y) ? 29 : dm[m - 1];
}
Date next_day(Date x) {
    if (++x.d > days_in_month(x.y, x.m)) {
        x.d = 1;
        if (++x.m > 12) {
            x.m = 1;
            ++x.y;
        }
    }
    return x;
}
int weekday(Date x) {  // 0 = Monday
    static const int t[] = {0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4};
    const int y = x.y - (x.m < 3);
    const int w = (y + y / 4 - y / 100 + y / 400 + t[x.m - 1] + x.d) % 7;
    return (w + 6) % 7;
}
bool is_holiday(Date x) { return is_b3_holiday(x.ymd()); }
TimePoint tp(Date x, int minute, int second = 0) {
    TimePoint t;
    t.ymd = x.ymd();
    t.minute = minute;
    t.millis = second * 1000;
    return t;
}
double round2(double v) { return std::round(v * 100.0) / 100.0; }

struct FillSpec {
    int day;
    int hh, mm, ss;
    const char* ticker;
    long qty;  // signed
};

// Client trades during the out-of-sample period (day index is 0-based).
const FillSpec kFills[] = {
    {64, 11, 7, 23, "XSAU3", 20000},   {64, 11, 12, 40, "XSAU3", 10000},  {68, 14, 31, 5, "XVAR3", -26000},
    {72, 10, 47, 51, "XTEC3", 15000},  {75, 15, 20, 12, "XTEC3", 8000},   {80, 13, 2, 33, "XBAN4", 20000},
    {86, 16, 41, 9, "XELE3", -5000},   {95, 10, 52, 44, "XMIN3", -15000}, {95, 11, 5, 10, "XSID3", -20000},
    {95, 11, 15, 0, "XIMB3", -50000},  {97, 15, 55, 38, "XPET3", -10000}, {103, 12, 20, 27, "XMIN3", 10000},
    {106, 10, 30, 55, "XSID3", 10000}, {112, 14, 14, 14, "XAGR3", 15000}, {118, 11, 48, 2, "XCON3", 10000},
    {120, 13, 33, 47, "XVAR3", -130000}, {126, 16, 5, 31, "XTEL3", -180000}, {131, 10, 21, 19, "XLOG3", 10000},
    {138, 15, 44, 50, "XTEC3", -23000}, {144, 12, 57, 3, "XPAP3", 10000}, {147, 14, 9, 26, "XBAN4", -30000},
};

// Scenario parameters
constexpr int kStressStart = 95, kStressEnd = 104;  // elevated market volatility regime
constexpr double kStressOmegaMult = 4.0;
constexpr double kCrashZ = -9.0;                    // market shock at the open of kStressStart
constexpr int kJumpDay = 120, kJumpBar = 12;        // earnings shock on XVAR3
constexpr double kJumpReturn = -0.09;
constexpr int kOptionsDay = 60;                     // option book is set up at this day's open

int find_ticker(const char* t) {
    for (int i = 0; i < kN; ++i)
        if (std::string(kAssets[i].ticker) == t) return i;
    return -1;
}

FILE* open_out(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + path);
    return f;
}

std::string upper(std::string s) {
    for (char& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

}  // namespace

std::vector<std::string> SyntheticMarket::tickers() const {
    std::vector<std::string> t;
    for (const AssetInfo& a : assets) t.push_back(a.ticker);
    return t;
}

SyntheticMarket generate_market(const SyntheticConfig& cfg) {
    if (cfg.days < 2) throw std::invalid_argument("generate_market: days must be >= 2");
    const bool real = cfg.realistic;
    SyntheticMarket mk;
    mk.session = SessionSpec{kOpenMinute, kBarMinutes, kBarsPerDay};
    for (const AssetDef& a : kAssets) {
        mk.assets.push_back({a.ticker, a.name, a.sector, a.beta});
        mk.initial_quantity.push_back(static_cast<double>(a.qty0));
    }
    Rng rng(cfg.seed);

    // diurnal U-shape (mean(s^2) = 1); realistic: each asset bends it differently
    double base[kBarsPerDay];
    for (int k = 0; k < kBarsPerDay; ++k)
        base[k] = 1.0 + 0.9 * std::exp(-k / 2.2) + 0.4 * std::exp(-(kBarsPerDay - 1 - k) / 2.5);
    std::vector<double> s(static_cast<size_t>(kN * kBarsPerDay));
    auto normalize = [](double* v) {
        double ms = 0.0;
        for (int k = 0; k < kBarsPerDay; ++k) ms += v[k] * v[k];
        for (int k = 0; k < kBarsPerDay; ++k) v[k] /= std::sqrt(ms / kBarsPerDay);
    };
    std::vector<double> bend(kN, 1.0);
    for (int i = 0; i < kN; ++i) {
        bend[static_cast<size_t>(i)] = real ? 0.75 + 0.5 * rng.uniform() : 1.0;
        for (int k = 0; k < kBarsPerDay; ++k)
            s[static_cast<size_t>(i * kBarsPerDay + k)] = std::pow(base[k], bend[static_cast<size_t>(i)]);
        normalize(&s[static_cast<size_t>(i * kBarsPerDay)]);
    }

    Vol mkt = Vol::make(0.013, real ? 0.025 : 0.05, real ? 0.05 : 0.0, 0.985, 5.0);
    Vol sect[kSectors];
    for (auto& g : sect) g = Vol::make(0.007, real ? 0.03 : 0.05, real ? 0.04 : 0.0, 0.980, 7.0);
    Vol idio[kN];
    for (int i = 0; i < kN; ++i) {
        const AssetDef& a = kAssets[i];
        idio[i] = Vol::make(a.idio_daily, real ? 0.5 * a.alpha : a.alpha, real ? a.alpha : 0.0, a.persistence, a.nu);
    }
    std::vector<int> regime(kN, 0);  // 0 calm, 1 agitated (realistic only)
    constexpr double kAgitatedMult = 3.0, kP01 = 0.03, kP10 = 0.15;

    // ---- trading calendar
    std::vector<Date> days;
    for (Date d{2026, 3, 2}; static_cast<int>(days.size()) < cfg.days; d = next_day(d))
        if (weekday(d) < 5 && !is_holiday(d)) days.push_back(d);

    double logp[kN];
    for (int i = 0; i < kN; ++i) logp[i] = std::log(kAssets[i].price0);
    mk.rows.reserve(static_cast<size_t>(cfg.days) * kBarsPerDay * kN);
    std::vector<int> row_day, row_bar;  // for the degraded copy
    double path[kN][kSteps + 1];
    std::vector<double> option_spot(kN, 0.0);

    for (int d = 0; d < cfg.days; ++d) {
        const Date day = days[static_cast<size_t>(d)];
        if (d > 0) {  // overnight gap (market + idiosyncratic), not part of the intraday returns
            const double gm = 0.006 * rng.std_t(5.0);
            for (int i = 0; i < kN; ++i) logp[i] += kAssets[i].beta * gm + 0.005 * rng.std_t(5.0);
        }
        if (real)
            for (int i = 0; i < kN; ++i) {
                const double u = rng.uniform();
                if (regime[i] == 0 && u < kP01) regime[i] = 1;
                else if (regime[i] == 1 && u < kP10) regime[i] = 0;
            }
        if (d == kOptionsDay)
            for (int i = 0; i < kN; ++i) option_spot[i] = std::exp(logp[i]);
        const bool stress = d >= kStressStart && d <= kStressEnd;
        for (int k = 0; k < kBarsPerDay; ++k) {
            const int close_min = kOpenMinute + (k + 1) * kBarMinutes;
            const TimePoint ts = tp(day, close_min);
            const bool halted = real && d == kStressStart && (k == 1 || k == 2);  // circuit breaker

            double f = mkt.draw(rng);
            if (d == kStressStart && k == 0) {
                f = kCrashZ * std::sqrt(mkt.h);
                mk.events.push_back({d, ts.str(), "market crash: factor shock of -9 conditional std devs; elevated "
                                                   "market volatility regime until day " + std::to_string(kStressEnd)});
                if (real)
                    mk.events.push_back({d, tp(day, close_min + 2 * kBarMinutes).str(),
                                         "circuit breaker: trading halted for two bars (no prints at 10:30 and 10:45)"});
            }
            double g[kSectors];
            for (int j = 0; j < kSectors; ++j) g[j] = sect[j].draw(rng);
            for (int i = 0; i < kN; ++i) {
                const AssetDef& a = kAssets[i];
                const double sk = s[static_cast<size_t>(i * kBarsPerDay + k)];
                double e = idio[i].draw(rng);
                double jump = 0.0;
                if (d == kJumpDay && k == kJumpBar && std::string(a.ticker) == "XVAR3") {
                    e = kJumpReturn / sk;
                    mk.events.push_back({d, ts.str(), "XVAR3 earnings shock: -9% idiosyncratic move in one bar"});
                }
                if (real && k == 0 && rng.uniform() < 0.2) jump = 0.005 * rng.normal();  // opening auction
                if (real && k == kBarsPerDay - 1 && rng.uniform() < 0.3) jump = 0.003 * rng.normal();  // closing call
                const double bar_sd =
                    sk * std::sqrt(a.beta * a.beta * mkt.h + a.gamma * a.gamma * sect[a.sector_id].h + idio[i].h);
                const double r = sk * (a.beta * f + a.gamma * g[a.sector_id] + e) + jump;

                // Brownian bridge for the 1-minute path from open to close
                const double step = bar_sd / std::sqrt(static_cast<double>(kSteps));
                double w[kSteps + 1];
                w[0] = 0.0;
                for (int m = 1; m <= kSteps; ++m) w[m] = w[m - 1] + step * rng.normal();
                const double o = logp[i];
                for (int m = 0; m <= kSteps; ++m)
                    path[i][m] = o + w[m] - (static_cast<double>(m) / kSteps) * (w[kSteps] - r);
                logp[i] = o + r;

                const double po = round2(std::exp(o)), pc = round2(std::exp(logp[i]));
                double hi = std::max(po, pc), lo = std::min(po, pc);
                for (int m = 1; m < kSteps; ++m) {
                    hi = std::max(hi, round2(std::exp(path[i][m])));
                    lo = std::min(lo, round2(std::exp(path[i][m])));
                }
                const double activity = std::pow(sk, 1.5) * std::exp(0.35 * rng.normal()) *
                                        (1.0 + 1.5 * std::fabs(r) / std::max(bar_sd, 1e-12));
                const double vol = std::max(100.0, std::floor(a.base_volume * activity / 100.0) * 100.0);
                if (!halted) {
                    mk.rows.push_back({ts, a.ticker, po, hi, lo, pc, vol});
                    row_day.push_back(d);
                    row_bar.push_back(k);
                }
                idio[i].update(e, real && regime[i] == 1 ? kAgitatedMult : 1.0);
            }
            mkt.update(f, stress ? kStressOmegaMult : 1.0);
            for (int j = 0; j < kSectors; ++j) sect[j].update(g[j]);

            // fills executed inside this bar, priced on the 1-minute path
            const int bar_open_min = close_min - kBarMinutes;
            for (const FillSpec& fl : kFills) {
                if (fl.day != d) continue;
                const double t_min = fl.hh * 60 + fl.mm + fl.ss / 60.0;
                if (!(t_min > bar_open_min && t_min <= close_min)) continue;
                const int i = find_ticker(fl.ticker);
                const double pos = t_min - bar_open_min;  // minutes into the bar, (0, 15]
                const int m0 = std::min(static_cast<int>(pos), kSteps - 1);
                const double frac = pos - m0;
                const double lp = path[i][m0] + frac * (path[i][m0 + 1] - path[i][m0]);
                const double slip = fl.qty > 0 ? 0.01 : -0.01;
                Fill out;
                char id[16];
                std::snprintf(id, sizeof id, "F%04zu", mk.fills.size() + 1);
                out.id = id;
                out.time = tp(day, fl.hh * 60 + fl.mm, fl.ss);
                out.ticker = fl.ticker;
                out.quantity = static_cast<double>(fl.qty);
                out.price = round2(std::exp(lp)) + slip;
                mk.fills.push_back(out);
            }
        }
    }
    if (cfg.days > kStressEnd + 1)
        mk.events.push_back({kStressEnd + 1, tp(days[kStressEnd + 1], kOpenMinute).str(),
                             "market volatility regime back to normal"});

    // ---- option book (set up at the open of kOptionsDay, expiring after the sample)
    if (cfg.days > kOptionsDay + 1) {
        Date e = days.back();
        for (int k = 0; k < 45; ++k) e = next_day(e);  // 3rd Friday of that month (or the next)
        auto third_friday = [](int y, int m) {
            Date x{y, m, 1};
            int fridays = 0;
            for (;; x = next_day(x))
                if (weekday(x) == 4 && ++fridays == 3) return x;
        };
        Date exp = third_friday(e.y, e.m);
        if (exp.ymd() < e.ymd()) exp = third_friday(e.m == 12 ? e.y + 1 : e.y, e.m == 12 ? 1 : e.m + 1);
        struct OptSpec {
            const char* id;
            const char* und;
            OptionType type;
            double moneyness, vol, qty;
        };
        const OptSpec specs[] = {{"OPT001", "XPET3", OptionType::Call, 1.00, 0.32, -300},
                                 {"OPT002", "XMIN3", OptionType::Put, 0.95, 0.38, 400},
                                 {"OPT003", "XBAN4", OptionType::Put, 0.95, 0.30, -500},
                                 {"OPT004", "XTEC3", OptionType::Call, 1.05, 0.45, 200}};
        for (const OptSpec& sp : specs) {
            const double spot = option_spot[static_cast<size_t>(find_ticker(sp.und))];
            const double tick = spot < 10 ? 0.1 : (spot < 50 ? 0.5 : 1.0);
            mk.options.push_back({sp.id, sp.und, sp.type, std::round(spot * sp.moneyness / tick) * tick,
                                  exp.ymd(), sp.vol, 0.14, 100.0, sp.qty});
        }
        mk.events.push_back({kOptionsDay, tp(days[kOptionsDay], kOpenMinute).str(),
                             "option book set up (options.csv); expiry " + tp(exp, 1020).str().substr(0, 10)});
    }

    // ---- degraded feed: same market, typical data problems
    if (cfg.degraded_copy) {
        Rng drng(cfg.seed ^ 0xDE6EADEDULL);
        const int stale_asset = find_ticker("XCON3"), stale_day = std::min(130, cfg.days - 1);
        struct Spike {
            int day, bar;
            const char* ticker;
            double mult;
        };
        const Spike spikes[] = {{88, 14, "XPET3", 1.07}, {102, 5, "XBAN4", 0.93}, {140, 20, "XTEC3", 1.10}};
        double stale_price = 0.0;
        auto& deg = mk.degraded_rows;
        deg.reserve(mk.rows.size() + mk.rows.size() / 100);
        for (size_t r = 0; r < mk.rows.size(); ++r) {
            BarRow row = mk.rows[r];
            const int d = row_day[r], k = row_bar[r];
            const int asset = find_ticker(row.ticker.c_str());
            if (asset == stale_asset && d == stale_day) {  // frozen quote between bars 8 and 17
                if (k == 7) stale_price = row.close;
                if (k >= 8 && k <= 17) {
                    row.open = row.high = row.low = row.close = stale_price;
                    row.volume = 0;
                }
            }
            for (const Spike& sp : spikes)
                if (d == sp.day && k == sp.bar && asset == find_ticker(sp.ticker)) {
                    row.close = round2(row.close * sp.mult);  // bad tick on the close only
                    row.high = std::max(row.high, row.close);
                    row.low = std::min(row.low, row.close);
                }
            if (drng.uniform() < 0.004) continue;  // missing row
            deg.push_back(row);
            if (drng.uniform() < 0.003) deg.push_back(row);  // duplicate
        }
        for (size_t r = 1; r + kN < deg.size(); ++r)  // local out-of-order delivery
            if (drng.uniform() < 0.002) std::swap(deg[r], deg[r + kN]);
        mk.events.push_back({stale_day, tp(days[static_cast<size_t>(stale_day)], kOpenMinute + 8 * kBarMinutes).str(),
                             "[degraded feed only] XCON3 price frozen for bars 8-17"});
        for (const Spike& sp : spikes)
            if (sp.day < cfg.days)
                mk.events.push_back({sp.day, tp(days[static_cast<size_t>(sp.day)], kOpenMinute + (sp.bar + 1) * kBarMinutes).str(),
                                     std::string("[degraded feed only] bad tick on ") + sp.ticker + " close"});
    }
    std::sort(mk.events.begin(), mk.events.end(),
              [](const ScenarioEvent& a, const ScenarioEvent& b) { return a.timestamp < b.timestamp; });

    // ---- true parameters
    char buf[256];
    mk.truth_lines.push_back("process,market_beta,sector_loading,daily_vol,alpha,gamma_leverage,beta,nu,diurnal_bend");
    std::snprintf(buf, sizeof buf, "market,,,0.0130,%.3f,%.3f,%.3f,5,", mkt.alpha, mkt.gamma, mkt.beta);
    mk.truth_lines.push_back(buf);
    for (int j = 0; j < kSectors; ++j) {
        std::snprintf(buf, sizeof buf, "sector_%d,,,0.0070,%.3f,%.3f,%.3f,7,", j, sect[j].alpha, sect[j].gamma, sect[j].beta);
        mk.truth_lines.push_back(buf);
    }
    for (int i = 0; i < kN; ++i) {
        const AssetDef& a = kAssets[i];
        std::snprintf(buf, sizeof buf, "idio_%s,%.2f,%.2f,%.4f,%.3f,%.3f,%.3f,%.1f,%.3f", a.ticker, a.beta, a.gamma,
                      a.idio_daily, idio[i].alpha, idio[i].gamma, idio[i].beta, a.nu, bend[static_cast<size_t>(i)]);
        mk.truth_lines.push_back(buf);
    }
    if (real) {
        mk.truth_lines.push_back("# idiosyncratic regimes: agitated state multiplies omega by 3; daily switching "
                                 "probabilities calm->agitated 0.03, agitated->calm 0.15");
        mk.truth_lines.push_back("# auctions: opening jump N(0, 0.5%) with prob. 0.2; closing jump N(0, 0.3%) with prob. 0.3");
    }
    return mk;
}

void write_market(const SyntheticMarket& m, const std::string& dir) {
    std::filesystem::create_directories(dir);
    FILE* f = open_out(dir + "/assets.csv");
    std::fprintf(f, "ticker,name,sector,market_beta\n");
    for (const AssetInfo& a : m.assets)
        std::fprintf(f, "%s,%s,%s,%.2f\n", a.ticker.c_str(), a.name.c_str(), a.sector.c_str(), a.market_beta);
    std::fclose(f);

    f = open_out(dir + "/portfolio_initial.csv");
    std::fprintf(f, "ticker,quantity\n");
    for (size_t i = 0; i < m.assets.size(); ++i)
        std::fprintf(f, "%s,%.0f\n", m.assets[i].ticker.c_str(), m.initial_quantity[i]);
    std::fclose(f);

    f = open_out(dir + "/fills.csv");
    std::fprintf(f, "fill_id,timestamp,ticker,side,quantity,price\n");
    for (const Fill& x : m.fills)
        std::fprintf(f, "%s,%s,%s,%s,%.0f,%.2f\n", x.id.c_str(), x.time.str().c_str(), x.ticker.c_str(),
                     x.quantity >= 0 ? "BUY" : "SELL", std::fabs(x.quantity), x.price);
    std::fclose(f);

    f = open_out(dir + "/options.csv");
    std::fprintf(f, "id,underlying,type,strike,expiry,implied_vol,rate,multiplier,quantity\n");
    for (const OptionContract& o : m.options)
        std::fprintf(f, "%s,%s,%s,%.2f,%04d-%02d-%02d,%.4f,%.4f,%.0f,%.0f\n", o.id.c_str(), o.underlying.c_str(),
                     o.type == OptionType::Call ? "CALL" : "PUT", o.strike, o.expiry_ymd / 10000,
                     (o.expiry_ymd / 100) % 100, o.expiry_ymd % 100, o.implied_vol, o.rate, o.multiplier, o.quantity);
    std::fclose(f);

    auto write_prices = [&](const std::string& path, const std::vector<BarRow>& rows) {
        FILE* p = open_out(path);
        std::fprintf(p, "timestamp,ticker,open,high,low,close,volume\n");
        for (const BarRow& r : rows)
            std::fprintf(p, "%s,%s,%.2f,%.2f,%.2f,%.2f,%.0f\n", r.time.str().c_str(), r.ticker.c_str(), r.open,
                         r.high, r.low, r.close, r.volume);
        std::fclose(p);
    };
    write_prices(dir + "/prices_15m.csv", m.rows);
    if (!m.degraded_rows.empty()) write_prices(dir + "/prices_15m_degraded.csv", m.degraded_rows);

    f = open_out(dir + "/scenario.csv");
    std::fprintf(f, "day,timestamp,event\n");
    for (const ScenarioEvent& e : m.events)
        std::fprintf(f, "%d,%s,\"%s\"\n", e.day, e.timestamp.c_str(), e.text.c_str());
    std::fclose(f);

    f = open_out(dir + "/simulation_truth.csv");
    for (const std::string& line : m.truth_lines) std::fprintf(f, "%s\n", line.c_str());
    std::fclose(f);
}

std::vector<Fill> load_fills_csv(const std::string& path, int offset_min) {
    const CsvTable t = read_csv(path);
    const int cid = t.column("fill_id"), cts = t.column("timestamp"), ctk = t.column("ticker"),
              cs = t.column("side"), cq = t.column("quantity"), cp = t.column("price");
    if (cts < 0 || ctk < 0 || cq < 0) throw std::runtime_error(path + ": needs timestamp, ticker and quantity");
    std::vector<Fill> out;
    for (size_t r = 0; r < t.rows.size(); ++r) {
        const auto& row = t.rows[r];
        auto field = [&](int c) { return c >= 0 && static_cast<size_t>(c) < row.size() ? row[static_cast<size_t>(c)] : std::string(); };
        Fill f;
        double q = 0.0, px = 0.0;
        if (!parse_time(field(cts), f.time, offset_min) || !parse_number(field(cq), q) || !std::isfinite(q))
            throw std::runtime_error(path + ": invalid fill on line " + std::to_string(r + 2));
        parse_number(field(cp), px);
        const std::string side = upper(field(cs));
        if (side == "SELL" || side == "S" || side == "V" || side == "VENDA") q = -std::fabs(q);
        else if (side == "BUY" || side == "B" || side == "C" || side == "COMPRA") q = std::fabs(q);
        else if (!side.empty())  // e.g. SHORT: guessing a direction would silently flip positions
            throw std::runtime_error(path + ": unknown side '" + field(cs) + "' on line " + std::to_string(r + 2) +
                                     " (BUY/SELL, B/S, the Portuguese C/V and COMPRA/VENDA, or empty for a signed quantity)");
        f.id = cid >= 0 ? field(cid) : std::to_string(r + 1);
        f.ticker = field(ctk);
        f.quantity = q;
        f.price = px;
        out.push_back(std::move(f));
    }
    std::stable_sort(out.begin(), out.end(), [](const Fill& a, const Fill& b) { return a.time.key() < b.time.key(); });
    return out;
}

std::vector<OptionContract> load_options_csv(const std::string& path) {
    const CsvTable t = read_csv(path);
    const int cid = t.column("id"), cu = t.column("underlying"), cty = t.column("type"), ck = t.column("strike"),
              ce = t.column("expiry"), cv = t.column("implied_vol"), cr = t.column("rate"),
              cm = t.column("multiplier"), cq = t.column("quantity");
    if (cu < 0 || cty < 0 || ck < 0 || ce < 0 || cv < 0 || cq < 0)
        throw std::runtime_error(path + ": needs underlying, type, strike, expiry, implied_vol, quantity");
    std::vector<OptionContract> out;
    for (size_t r = 0; r < t.rows.size(); ++r) {
        const auto& row = t.rows[r];
        auto field = [&](int c) { return c >= 0 && static_cast<size_t>(c) < row.size() ? row[static_cast<size_t>(c)] : std::string(); };
        OptionContract o;
        TimePoint e;
        double rate = 0.0, mult = 100.0;
        const std::string type = upper(field(cty));
        if (!parse_number(field(ck), o.strike) || !parse_number(field(cv), o.implied_vol) ||
            !parse_number(field(cq), o.quantity) || !parse_time(field(ce) + " 17:00", e) ||
            (type != "CALL" && type != "PUT" && type != "C" && type != "P"))
            throw std::runtime_error(path + ": invalid option on line " + std::to_string(r + 2));
        if (!(o.strike > 0.0) || !(o.implied_vol > 0.0) || !std::isfinite(o.quantity))
            throw std::runtime_error(path + ": option on line " + std::to_string(r + 2) +
                                     " needs a positive strike and implied_vol and a quantity");
        if (cr >= 0) parse_number(field(cr), rate);
        if (cm >= 0) parse_number(field(cm), mult);
        o.id = cid >= 0 ? field(cid) : "OPT" + std::to_string(r + 1);
        o.underlying = field(cu);
        o.type = (type == "CALL" || type == "C") ? OptionType::Call : OptionType::Put;
        o.expiry_ymd = e.ymd;
        o.rate = std::isfinite(rate) ? rate : 0.0;
        o.multiplier = std::isfinite(mult) ? mult : 100.0;
        out.push_back(o);
    }
    return out;
}

}  // namespace garcht
