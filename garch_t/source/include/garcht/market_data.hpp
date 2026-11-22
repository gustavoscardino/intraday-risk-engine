// Loading of intraday bar data (CSV) and construction of per-bar log-returns.
#pragma once

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace garcht {

// ---------------------------------------------------------------- CSV (RFC 4180)

struct CsvTable {
    char delimiter = ',';
    std::vector<std::string> header;
    std::vector<std::vector<std::string>> rows;
    int column(const std::string& name) const;  // case-insensitive; -1 if absent
};

// RFC 4180 reader: quoted fields (embedded delimiters, line breaks and "" escapes), LF or CRLF,
// UTF-8 BOM, lines starting with '#' are comments, lines with only delimiters/blanks are
// skipped. If `delimiter` is 0 it is detected from the header line among ',', ';' and tab
// (Brazilian spreadsheets usually export ';'); a header with none of them is one column.
CsvTable read_csv(const std::string& path, char delimiter = 0);  // throws std::runtime_error
CsvTable parse_csv(const std::string& text, char delimiter = 0);

// Number with '.' or ',' as decimal separator and optional thousands separators:
// "38.50", "38,50", "1.234,56", "1,234.56", "-1.5e-3". A single ',' without '.' is read as a
// decimal comma (so "1,234" is 1.234). Empty, "NA", "NaN", "null" and "-" give NaN (true).
bool parse_number(const std::string& s, double& v);

// ---------------------------------------------------------------- time

// Wall-clock instant in the exchange's local time, millisecond resolution.
struct TimePoint {
    int ymd = 0;     // YYYYMMDD
    int minute = 0;  // minutes since midnight
    int millis = 0;  // milliseconds within the minute
    long long key() const;  // strictly increasing with time (ms since 1970-01-01, local)
    std::string str() const;  // "YYYY-MM-DD HH:MM:SS"
};

// Accepts "YYYY-MM-DD HH:MM[:SS[.fff]]" ('T', 't' or ' ' between date and time; '/' also allowed
// in the date), Brazilian "DD/MM/YYYY HH:MM[:SS]", and an optional zone suffix "Z", "+HH:MM",
// "-HHMM", "-HH". Zoned instants are converted to the exchange's time zone, given by its UTC
// offset in minutes (B3: -180). Unzoned instants are taken as exchange local time.
bool parse_time(const std::string& s, TimePoint& out, int exchange_utc_offset_min = -180);

// B3 holidays falling on weekdays, 2025-2027 (extend the table for other years).
bool is_b3_holiday(int yyyymmdd);

// Convenience wrapper over parse_time (B3 time; zoned inputs converted; seconds truncated).
bool parse_timestamp(const std::string& s, int& yyyymmdd, int& minute_of_day, int* second = nullptr);

// ---------------------------------------------------------------- bars

struct SessionSpec {
    int open_minute = 10 * 60;  // 10:00
    int bar_minutes = 15;
    int bars_per_day = 28;      // last bar closes at 17:00
};

// Slot of a bar that *closes* at `minute_of_day` (0 = first bar), or -1 if off-grid.
int bar_slot(int minute_of_day, const SessionSpec& s);

// Bars aligned on a common time grid; matrices are row-major [bar][asset], NaN = missing.
struct BarPanel {
    std::vector<std::string> tickers;
    std::vector<std::string> timestamps;  // bar close time "YYYY-MM-DD HH:MM:SS", ascending
    std::vector<long long> time_key;      // TimePoint::key() of each bar
    std::vector<int> date;                // YYYYMMDD
    std::vector<int> minute;              // minutes since midnight
    std::vector<double> open, high, low, close, volume;

    size_t num_bars() const { return timestamps.size(); }
    size_t num_assets() const { return tickers.size(); }
    double at(const std::vector<double>& m, size_t bar, size_t asset) const {
        return m[bar * tickers.size() + asset];
    }

    // Data-quality counters from loading.
    size_t rows_read = 0, duplicates = 0, bad_rows = 0, unknown_tickers = 0;
};

// One bar of one asset, as delivered by a feed (any order; duplicates allowed). Bars are
// identified by their closing *minute*: rows stamped 10:15:00 and 10:15:30 are the same bar.
struct BarRow {
    TimePoint time;  // bar close
    std::string ticker;
    double open = 0.0, high = 0.0, low = 0.0, close = 0.0, volume = 0.0;
};

// Aligns rows on a common time grid (sorted by time; duplicated (time, ticker) pairs keep
// the first occurrence; unknown tickers are counted and skipped).
BarPanel build_panel(const std::vector<BarRow>& rows, const std::vector<std::string>& tickers);

// Long format with columns timestamp,ticker,open,high,low,close,volume (aliases accepted:
// datetime/time for timestamp, separate date + time columns, symbol/asset for ticker; open,
// high, low and volume are optional). Tickers match case-insensitively. Rows are ordered by
// their parsed time, so the file order does not matter. Duplicated (time, ticker) rows keep
// the first occurrence; unparseable rows are counted and skipped.
BarPanel load_bars_csv(const std::string& path, const std::vector<std::string>& tickers,
                       int exchange_utc_offset_min = -180, const SessionSpec* resample = nullptr);

// Samples finer than the bar grid (e.g. 5-second prices) aggregated into the bars of `grid`: a row
// stamped at instant tau belongs to the bar closing at the first grid time >= tau (bars are stamped
// at their close). Per (bar, ticker): open of the first sample, highest high, lowest low, close of
// the last sample, summed volume; the aggregated row is stamped at the bar close. Rows outside the
// session keep their own minute (off-grid rows are ignored downstream). With load_bars_csv(...,
// &grid) the 15-minute engines run on any grid: for a 5-minute VaR every minute, five runs on
// 5-minute grids opening one minute apart (see the README).
std::vector<BarRow> resample_rows(const std::vector<BarRow>& rows, const SessionSpec& grid);

// Session grid inferred from the data: the most common spacing between consecutive bars of
// a day and the most common first / last bar times. Falls back to SessionSpec{} if unclear.
SessionSpec detect_session(const BarPanel& p);

// Per-bar log-returns for the risk engine (row-major [bar][asset]).
//  * first bar of each session: ln(close / open) -- the overnight gap is excluded;
//  * otherwise ln(close_t / previous on-grid close of the session);
//  * a return spanning k grid slots (missing bars, halts) is scaled by 1/sqrt(k) so that its
//    variance stays comparable to a single bar;
//  * off-grid rows (slot -1) are ignored; non-finite or non-positive prices give NaN.
// `slot` receives the session slot of each bar (-1 if off-grid).
void bar_returns(const BarPanel& panel, const SessionSpec& session, std::vector<double>& ret,
                 std::vector<int>& slot);

// The same returns and overnight gaps one on-grid bar at a time (bar_returns and
// overnight_returns run on it, so the numbers are the same by construction): for a live feed or a
// replay that prepares each bar's input inside its own cycle.
class ReturnStream {
public:
    explicit ReturnStream(size_t assets);
    // One on-grid bar (slot >= 0), in time order: open / close of the n assets (NaN = missing).
    // Writes the n bar returns and the n overnight gaps (NaN unless this is the first bar of a
    // session opening at slot 0).
    void step(int date, int slot, const double* open, const double* close, double* ret, double* gap);

private:
    struct Asset {
        double ref = std::numeric_limits<double>::quiet_NaN();  // last valid price of the session
        int ref_slot = -1;  // slot whose close `ref` is (slot - 1 = the session open)
        double last_close = std::numeric_limits<double>::quiet_NaN();  // of any session (gaps)
    };
    std::vector<Asset> a_;
    int day_ = -1;
};

// Overnight log-returns ln(open of the session's first bar / last on-grid close of the
// previous session), stored on the row of each session's first on-grid bar (row-major
// [bar][asset]). NaN on other rows, on the first session, when the asset has no opening
// print, and on sessions that do not open at slot 0 (late openings). `slot` as returned by
// bar_returns; off-grid rows are ignored.
void overnight_returns(const BarPanel& panel, const std::vector<int>& slot, std::vector<double>& gaps);

}  // namespace garcht
