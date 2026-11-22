// Market data: CSV reading, timestamps, the B3 calendar, the trading-session grid, the bar
// panel and the per-bar returns. The return conventions are the same as the GARCH-t engine's,
// so that the forecasts of both can be compared bar by bar:
//   * first bar of a session: ln(close / open) -- the overnight gap is excluded;
//   * otherwise ln(close / previous close of the same session);
//   * a return spanning k grid slots (missing bars) is divided by sqrt(k);
//   * the overnight gap ln(open of slot 0 / last close of the previous session) is kept apart.
#pragma once

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace ewmar {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

inline bool valid_price(double v) { return v == v && v > 0.0 && v < std::numeric_limits<double>::infinity(); }

// ---------------------------------------------------------------- CSV

struct CsvTable {
    std::vector<std::string> header;
    std::vector<std::vector<std::string>> rows;
    std::vector<size_t> line;  // 1-based line number of each row in the file
    int column(const std::string& name) const;  // case-insensitive; -1 if absent
};

// Delimiter detected from the header among ',', ';' and tab. Quoted fields, CRLF, a UTF-8 BOM,
// blank lines and '#' comment lines are handled. Throws std::runtime_error on I/O errors.
CsvTable read_csv(const std::string& path);
CsvTable parse_csv(const std::string& text);

// "38.50", "38,50", "1.234,56", "1,234.56", "-1e-3". Empty / NA / NaN / null give NaN (true);
// anything else that is not a number gives false.
bool parse_number(const std::string& s, double& out);

// ---------------------------------------------------------------- time and calendar

struct TimePoint {
    int ymd = 0;     // YYYYMMDD
    int minute = 0;  // minutes since midnight
    int second = 0;
    long long key() const;    // seconds since 1970-01-01 (exchange local time); increasing with time
    std::string str() const;  // "YYYY-MM-DD HH:MM:SS"
};

// "YYYY-MM-DD HH:MM[:SS[.fff]]" ('T' or ' ' separator, '/' allowed in the date), "DD/MM/YYYY HH:MM[:SS]"
// and a date alone ("YYYY-MM-DD", midnight). A zone suffix (Z, +HH:MM, -HHMM, -HH) converts the
// instant to the exchange's local time, given by its UTC offset in minutes (B3: -180).
bool parse_time(const std::string& s, TimePoint& out, int utc_offset_min = -180);
bool parse_date(const std::string& s, int& ymd);

int weekday(int ymd);  // 0 = Monday ... 6 = Sunday
int next_day(int ymd);
bool is_b3_holiday(int ymd);  // weekday holidays of B3, 2025-2027
bool is_business_day(int ymd);

// ---------------------------------------------------------------- session grid

struct SessionSpec {
    int open_minute = 10 * 60;  // session open (the first bar closes at open + bar_minutes)
    int bar_minutes = 15;
    int bars_per_day = 28;
    int slot_of(int minute_of_day) const;  // slot of a bar closing at that minute; -1 if off-grid
    int close_minute(int slot) const { return open_minute + (slot + 1) * bar_minutes; }
};

// ---------------------------------------------------------------- feed rows

// One row of the price file, in the order it appears in the file (the order it arrived from the
// feed). `asset` is the index of the ticker, -1 if the ticker is not in the portfolio.
struct RawRow {
    TimePoint time;  // bar close
    int asset = -1;
    double open = kNaN, high = kNaN, low = kNaN, close = kNaN, volume = kNaN;
    size_t line = 0;
};

struct FeedFile {
    std::vector<std::string> tickers;
    std::vector<RawRow> rows;  // parsed rows of known tickers, file order
    size_t lines = 0, bad_rows = 0, unknown_rows = 0;
};

// Columns timestamp (or datetime, or date + time), ticker (or symbol), open, high, low, close,
// volume. Rows that cannot be parsed are counted in bad_rows and skipped.
FeedFile load_feed(const std::string& path, const std::vector<std::string>& tickers, int utc_offset_min = -180);

// Grid inferred from the data: most common spacing between bars and most common first / last bar.
SessionSpec detect_session(const std::vector<RawRow>& rows);

// Samples finer than the bar grid (e.g. 5-second prices) aggregated into the bars of `grid`: a row
// stamped at instant tau belongs to the bar closing at the first grid time >= tau. Per (bar, asset):
// open of the earliest sample, highest high, lowest low, close of the latest sample, summed volume,
// stamped at the bar close. The aggregated row takes the file position of the last sample of its
// bar (when the bar was complete), so the arrival rules of the data-quality layer still apply. Rows
// outside the session keep their stamp. With it the engine runs on any grid: for a 5-minute VaR
// every minute, five runs on 5-minute grids opening one minute apart (see the README).
FeedFile resample_feed(const FeedFile& feed, const SessionSpec& grid);

// ---------------------------------------------------------------- panel

// Bars on a common time grid, row-major [bar][asset], NaN = no row.
struct Panel {
    std::vector<std::string> tickers;
    std::vector<std::string> stamp;  // bar close "YYYY-MM-DD HH:MM:SS"
    std::vector<long long> key;
    std::vector<int> ymd, minute, slot;
    std::vector<double> open, high, low, close, volume;
    size_t bars() const { return stamp.size(); }
    size_t assets() const { return tickers.size(); }
    double at(const std::vector<double>& m, size_t t, size_t i) const { return m[t * tickers.size() + i]; }
    double& at(std::vector<double>& m, size_t t, size_t i) { return m[t * tickers.size() + i]; }
};

struct PanelStats {
    size_t duplicates = 0, off_grid = 0;
};

// Panel without any data-quality layer (the --no-dq path, like a plain loader): rows sorted by
// time, a repeated (time, ticker) keeps its first occurrence, off-grid rows are dropped.
Panel sorted_panel(const FeedFile& feed, const SessionSpec& s, PanelStats* stats = nullptr);

// ---------------------------------------------------------------- returns

// Per-asset state of the return conventions; step() is called for every asset at every bar of
// the panel (also when the asset has no row), in time order.
struct ReturnState {
    int day = -1;
    double ref = kNaN;   // last valid price of the session (the session open at its start)
    int ref_slot = -1;   // slot whose close `ref` is (slot - 1 for the session open)
    double last_close = kNaN;  // last valid close of any session (for the overnight gap)
    // Returns the bar return (NaN when there is no valid close) and writes the overnight gap
    // (NaN unless this is the session's first bar, at slot 0, with a previous close).
    double step(int ymd, int slot, double open, double close, double* gap = nullptr);
};

// Batch version over a whole panel: ret and gaps are row-major [bar][asset].
void panel_returns(const Panel& p, std::vector<double>& ret, std::vector<double>& gaps);

// ---------------------------------------------------------------- portfolio inputs

struct Fill {
    TimePoint time;
    std::string ticker;
    double quantity = 0.0;  // signed: positive = buy
};
std::vector<Fill> load_fills(const std::string& path, int utc_offset_min = -180);

struct OptionContract {
    std::string id, underlying;
    bool call = true;
    double strike = 0.0;
    int expiry_ymd = 0;  // expires at the close of that day
    double implied_vol = 0.0, rate = 0.0, multiplier = 1.0, quantity = 0.0;
};
std::vector<OptionContract> load_options(const std::string& path);

}  // namespace ewmar
