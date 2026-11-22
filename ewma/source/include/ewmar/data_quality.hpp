// Data-quality (DQ) layer between the price feed and the risk engine.
//
// The rows of the price file are taken in file order, as the order in which they arrived from
// the feed. Objective rules, per asset:
//   DUPLICATE     same (ticker, bar) already received                      -> row discarded
//   OUT_OF_ORDER  bar older than the asset's watermark, within the allowed
//                 lateness (default 1 bar); delay = watermark - bar         -> row accepted
//   DELAY         bar already closed when the row arrives                   -> row discarded
//   GAP           bar printed by other assets but not by this one           -> missing return
//   STALE         close unchanged with volume 0 for >= 3 bars in a row (volume 0 tells a frozen
//                 quote from a low-priced stock that just did not move)     -> missing return
//   SPIKE         |z| > 6 against the forecast sigma: the EWMA is updated with the winsorized
//                 value. If the next bar reverts the move (less than 25% of it left), it was a
//                 bad tick: the price is excluded and the bar re-fed. If not, the shock is real
//                 and is kept (SHOCK_KEPT). Several assets jumping together = a market move, kept
//
// The bar's score is dq_score = sum_i w_i q_i, w_i = |exposure_i| / sum |exposure|, q_i = 1 minus
// the penalties of the asset's active flags. Modes, with hysteresis (a better mode only after
// `recover_bars` bars that qualify for it):
//   NORMAL   score >= 0.9
//   DEGRADED 0.6 <= score < 0.9   forecasts published with a mark, warning alert
//   FREEZE   score < 0.6          EWMA not updated, last valid VaR/ES kept, critical alert
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "ewmar/market_data.hpp"

namespace ewmar {

enum DqFlag : std::uint16_t {
    kGap = 1u << 0,
    kDuplicate = 1u << 1,
    kOutOfOrder = 1u << 2,
    kDelay = 1u << 3,
    kStale = 1u << 4,
    kSpike = 1u << 5,      // confirmed bad tick (excluded)
    kSuspect = 1u << 6,    // |z| above the limit, decision pending (winsorized meanwhile)
    kShockKept = 1u << 7,  // suspect released: the move was real and was kept
};
const char* flag_name(std::uint16_t single_flag);
std::string flags_to_string(std::uint16_t flags);  // "GAP|STALE"

enum class DqMode { Normal = 0, Degraded = 1, Freeze = 2 };
const char* to_string(DqMode m);

struct DqConfig {
    double normal_threshold = 0.9, freeze_threshold = 0.6;
    int recover_bars = 3;   // hysteresis: clean bars needed to move to a better mode
    int lateness_bars = 1;  // a bar closes when the feed is this many bars past it
    int stale_bars = 3;
    double spike_z = 6.0;
    double revert_fraction = 0.25;  // bad tick if less than this fraction of the move is left next bar
    // A bad tick is one asset's problem: when at least max(market_wide_assets, market_wide_share * n)
    // assets jump at the same bar, the move is the market's and is kept at once.
    int market_wide_assets = 3;
    double market_wide_share = 0.2;
    // penalties subtracted from an asset's quality q_i
    double pen_gap = 1.0, pen_stale = 1.0, pen_suspect = 1.0, pen_spike = 1.0;
    double pen_out_of_order = 0.1, pen_duplicate = 0.05;
};

struct DqEvent {
    std::string time;  // bar close of the affected cell
    std::string ticker, flag, detail;
};

struct DqAlert {
    std::string time, severity, type, message;
};

// Cell key: (bar close key, asset).
inline long long cell_key(long long time_key, int asset) { return time_key * 4096 + asset; }

struct IngestResult {
    Panel panel;                       // accepted rows on the session grid (STALE cells masked)
    std::vector<std::uint16_t> flags;  // [bar][asset]
    std::unordered_map<long long, std::uint16_t> cell_flags;  // every flagged cell, also off-panel
    std::vector<DqEvent> events;
    size_t accepted = 0, duplicates = 0, out_of_order = 0, delayed = 0, off_grid = 0, gaps = 0, stale = 0;
};

// Streams the rows of the feed in file order through the arrival rules (DUPLICATE, OUT_OF_ORDER,
// DELAY), builds the panel, marks GAP cells and runs the STALE rule bar by bar.
IngestResult ingest_feed(const FeedFile& feed, const SessionSpec& s, const DqConfig& cfg);

// Exposure-weighted score of one bar (flags of each asset); equal weights if all exposures are 0.
double dq_score(const std::uint16_t* flags, const double* exposures, size_t n, const DqConfig& cfg);

class ModeController {
public:
    explicit ModeController(const DqConfig& cfg) : cfg_(cfg) {}
    // Feeds the score of one bar; returns the mode in force for it.
    DqMode update(double score);
    DqMode mode() const { return mode_; }
    DqMode level(double score) const;

private:
    DqConfig cfg_;
    DqMode mode_ = DqMode::Normal;
    DqMode run_worst_ = DqMode::Normal;  // worst level seen in the current run of better bars
    int better_run_ = 0;
};

// Detection against the known truth: the degraded feed compared with the clean file.
struct DetectionRow {
    std::string flag;
    size_t injected = 0, detected = 0, other = 0;  // other = flags on cells without that fault
    double recall = kNaN;
};
// DELAY / OUT_OF_ORDER truth comes from the arrival order of the received file: a row that arrives
// after the feed moved more than lateness_bars past its bar is late; one that arrives after a newer
// row of its stock (and is not late) is out of order.
std::vector<DetectionRow> evaluate_detection(const FeedFile& clean, const FeedFile& received, const SessionSpec& s,
                                             const std::unordered_map<long long, std::uint16_t>& cell_flags,
                                             int lateness_bars = 1);

}  // namespace ewmar
