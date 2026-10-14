// Writes the synthetic intraday dataset (see include/garcht/synthetic.hpp).
//
//   generate_data [--out DIR] [--days N] [--seed S] [--simple] [--no-degraded]
//
// With no arguments it rewrites data/synthetic (next to garch_t): the generator is
// deterministic, so the files come out identical to the distributed ones. A variant
// (--days, --seed, --simple, --no-degraded) needs its own --out folder.
// --simple turns off the realistic features (leverage, regimes, auctions, circuit breaker,
// asset-specific intraday patterns), leaving a market close to the model's assumptions.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

#include "cli.hpp"
#include "garcht/synthetic.hpp"
#include "paths.hpp"

using namespace garcht;

int main(int argc, char** argv) {
    const paths::Utf8Console console;
    std::string out;
    bool variant = false;
    SyntheticConfig cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool has = i + 1 < argc;
        bool ok = true;
        int seed = 0;
        if (a == "--out" && has) out = argv[++i];
        else if (a == "--days" && has) ok = cli::to_int(a, argv[++i], cfg.days, 2, 2000);
        else if (a == "--seed" && has) {
            ok = cli::to_int(a, argv[++i], seed, 0, 2147483647);
            cfg.seed = static_cast<std::uint64_t>(seed);
        } else if (a == "--simple") cfg.realistic = false;
        else if (a == "--no-degraded") cfg.degraded_copy = false;
        else ok = false;
        if (!ok) {
            std::fprintf(stderr, "usage: generate_data [--out DIR] [--days N] [--seed S] [--simple] [--no-degraded]\n");
            return 2;
        }
        variant = variant || a != "--out";
    }
    if (out.empty()) {
        if (variant) {  // never overwrite the distributed dataset with a different market
            std::fprintf(stderr, "error: a variant of the dataset needs its own folder: add --out FOLDER\n");
            return 2;
        }
        const paths::fs::path found = paths::find_dataset("synthetic");
        out = !found.empty() ? found.string()
                             : (paths::project_root().parent_path() / "data" / "synthetic").string();
    }
    try {
        const SyntheticMarket m = generate_market(cfg);
        write_market(m, out);
        std::printf("generated %d trading days (%s .. %s), %d bars/day, %zu assets, %s profile\n", cfg.days,
                    m.rows.front().time.str().substr(0, 10).c_str(), m.rows.back().time.str().substr(0, 10).c_str(),
                    m.session.bars_per_day, m.assets.size(), cfg.realistic ? "realistic" : "simple");
        std::printf("  prices_15m.csv           %zu rows\n", m.rows.size());
        if (cfg.degraded_copy)
            std::printf("  prices_15m_degraded.csv  %zu rows (gaps, duplicates, out-of-order rows, a stale price, bad ticks)\n",
                        m.degraded_rows.size());
        std::printf("  fills.csv                %zu fills\n", m.fills.size());
        std::printf("  options.csv              %zu options\n", m.options.size());
        std::printf("  scenario.csv             %zu events\n", m.events.size());
        std::printf("written to %s\n", out.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
