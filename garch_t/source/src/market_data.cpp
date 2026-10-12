#include "garcht/market_data.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace garcht {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

std::string trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

std::string upper(std::string s) {
    for (char& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

// ---- calendar arithmetic (H. Hinnant's algorithms, proleptic Gregorian)
long long days_from_civil(int y, int m, int d) {
    y -= m <= 2;
    const long long era = (y >= 0 ? y : y - 399) / 400;
    const long long yoe = y - era * 400;
    const long long doy = (153 * (m > 2 ? m - 3 : m + 9) + 2) / 5 + d - 1;
    const long long doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + doe - 719468;
}

void civil_from_days(long long z, int& y, int& m, int& d) {
    z += 719468;
    const long long era = (z >= 0 ? z : z - 146096) / 146097;
    const long long doe = z - era * 146097;
    const long long yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    const long long doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const long long mp = (5 * doy + 2) / 153;
    d = static_cast<int>(doy - (153 * mp + 2) / 5 + 1);
    m = static_cast<int>(mp < 10 ? mp + 3 : mp - 9);
    y = static_cast<int>(yoe + era * 400 + (m <= 2));
}

int days_in_month(int y, int m) {
    static const int dm[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    const bool leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    return (m == 2 && leap) ? 29 : dm[m - 1];
}

// Cursor-based helpers for parse_time.
struct Cursor {
    const std::string& s;
    size_t i = 0;
    bool eof() const { return i >= s.size(); }
    char peek() const { return eof() ? '\0' : s[i]; }
    bool take(char c) {
        if (peek() != c) return false;
        ++i;
        return true;
    }
    // reads between min_d and max_d digits; returns -1 if fewer than min_d
    long long digits(int min_d, int max_d, int* count = nullptr) {
        long long v = 0;
        int k = 0;
        while (k < max_d && !eof() && std::isdigit(static_cast<unsigned char>(s[i]))) {
            v = v * 10 + (s[i] - '0');
            ++i;
            ++k;
        }
        if (count) *count = k;
        return k >= min_d ? v : -1;
    }
    void skip_spaces() {
        while (!eof() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    }
};

}  // namespace

// ============================================================== CSV

int CsvTable::column(const std::string& name) const {
    const std::string target = upper(trim(name));
    for (size_t i = 0; i < header.size(); ++i)
        if (upper(trim(header[i])) == target) return static_cast<int>(i);
    return -1;
}

CsvTable parse_csv(const std::string& text_in, char delimiter) {
    size_t start = 0;
    if (text_in.size() >= 3 && static_cast<unsigned char>(text_in[0]) == 0xEF &&
        static_cast<unsigned char>(text_in[1]) == 0xBB && static_cast<unsigned char>(text_in[2]) == 0xBF)
        start = 3;  // UTF-8 BOM
    const std::string& text = text_in;

    if (delimiter == 0) {  // detect from the first non-comment line, outside quotes
        size_t p = start;
        while (p < text.size() && (text[p] == '#' || text[p] == '\n' || text[p] == '\r')) {
            if (text[p] == '#')
                while (p < text.size() && text[p] != '\n') ++p;
            ++p;
        }
        int comma = 0, semi = 0, tab = 0;
        bool q = false;
        for (; p < text.size() && (q || (text[p] != '\n' && text[p] != '\r')); ++p) {
            if (text[p] == '"') q = !q;
            else if (!q && text[p] == ',') ++comma;
            else if (!q && text[p] == ';') ++semi;
            else if (!q && text[p] == '\t') ++tab;
        }
        delimiter = (semi > comma && semi >= tab) ? ';' : (tab > comma && tab > semi) ? '\t' : ',';
        // a single-column header: no separator at all, so do not split "0,0012" (decimal comma)
        if (comma == 0 && semi == 0 && tab == 0) delimiter = '\t';
    }

    CsvTable t;
    t.delimiter = delimiter;
    std::vector<std::string> record;
    std::string field;
    bool quoted_field = false;  // current field started with a quote
    bool any_char = false;      // current record has content
    auto end_field = [&]() {
        record.push_back(quoted_field ? field : trim(field));
        field.clear();
        quoted_field = false;
    };
    auto end_record = [&]() {
        end_field();
        const bool blank = !any_char;  // only whitespace and delimiters (e.g. ";;" from a spreadsheet)
        const bool comment = !record.empty() && !record[0].empty() && record[0][0] == '#' && !blank;
        if (!blank && !comment) {
            if (t.header.empty()) t.header = std::move(record);
            else t.rows.push_back(std::move(record));
        }
        record.clear();
        any_char = false;
    };

    size_t i = start;
    const size_t n = text.size();
    while (i < n) {
        const char c = text[i];
        if (c == '"' && trim(field).empty() && !quoted_field) {  // opening quote
            field.clear();
            quoted_field = true;
            any_char = true;
            ++i;
            while (i < n) {
                if (text[i] == '"') {
                    if (i + 1 < n && text[i + 1] == '"') {  // escaped quote
                        field.push_back('"');
                        i += 2;
                        continue;
                    }
                    ++i;  // closing quote
                    break;
                }
                field.push_back(text[i++]);
            }
            while (i < n && (text[i] == ' ' || text[i] == '\t') && text[i] != delimiter) ++i;
            continue;
        }
        if (c == delimiter) {
            end_field();
            ++i;
        } else if (c == '\r' || c == '\n') {
            end_record();
            if (c == '\r' && i + 1 < n && text[i + 1] == '\n') ++i;
            ++i;
        } else {
            if (!quoted_field) field.push_back(c);
            any_char = any_char || !std::isspace(static_cast<unsigned char>(c));
            ++i;
        }
    }
    if (!field.empty() || !record.empty() || quoted_field) end_record();
    return t;
}

CsvTable read_csv(const std::string& path, char delimiter) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::stringstream buf;
    buf << in.rdbuf();
    CsvTable t = parse_csv(buf.str(), delimiter);
    if (t.header.empty()) throw std::runtime_error("empty csv: " + path);
    return t;
}

bool parse_number(const std::string& raw, double& v) {
    const std::string s = trim(raw);
    if (s.empty() || s == "-") {
        v = kNaN;
        return true;
    }
    const std::string u = upper(s);
    if (u == "NA" || u == "NAN" || u == "NULL" || u == "N/A") {
        v = kNaN;
        return true;
    }
    const size_t last_dot = s.rfind('.'), last_comma = s.rfind(',');
    const size_t n_dot = static_cast<size_t>(std::count(s.begin(), s.end(), '.'));
    const size_t n_comma = static_cast<size_t>(std::count(s.begin(), s.end(), ','));
    char decimal = '.';
    char thousands = '\0';
    if (n_dot && n_comma) {
        decimal = last_dot > last_comma ? '.' : ',';
        thousands = decimal == '.' ? ',' : '.';
    } else if (n_comma) {
        if (n_comma == 1) decimal = ',';
        else thousands = ',';
    } else if (n_dot > 1) {
        thousands = '.';
    }
    std::string clean;
    clean.reserve(s.size());
    for (char c : s) {
        if (c == thousands) continue;
        clean.push_back(c == decimal ? '.' : c);
    }
    if (std::count(clean.begin(), clean.end(), '.') > 1) return false;
    char* end = nullptr;
    v = std::strtod(clean.c_str(), &end);
    return end != clean.c_str() && *end == '\0';
}

// ============================================================== time

long long TimePoint::key() const {
    const int y = ymd / 10000, m = (ymd / 100) % 100, d = ymd % 100;
    return (days_from_civil(y, m, d) * 1440LL + minute) * 60000LL + millis;
}

std::string TimePoint::str() const {
    char buf[24];
    std::snprintf(buf, sizeof buf, "%04d-%02d-%02d %02d:%02d:%02d", ymd / 10000, (ymd / 100) % 100,
                  ymd % 100, minute / 60, minute % 60, millis / 1000);
    return buf;
}

bool parse_time(const std::string& raw, TimePoint& out, int exchange_utc_offset_min) {
    const std::string s = trim(raw);
    Cursor c{s};
    long long y, mo, d;
    int k1 = 0;
    const long long a = c.digits(1, 4, &k1);
    if (a < 0) return false;
    if (k1 == 4) {  // YYYY-MM-DD or YYYY/MM/DD
        if (!(c.take('-') || c.take('/'))) return false;
        y = a;
        mo = c.digits(1, 2);
        if (!(c.take('-') || c.take('/'))) return false;
        d = c.digits(1, 2);
    } else {  // DD/MM/YYYY
        if (!c.take('/')) return false;
        d = a;
        mo = c.digits(1, 2);
        if (!c.take('/')) return false;
        int k3 = 0;
        y = c.digits(4, 4, &k3);
    }
    if (y < 1 || mo < 1 || mo > 12 || d < 1 || d > days_in_month(static_cast<int>(y), static_cast<int>(mo)))
        return false;
    if (!(c.take('T') || c.take('t') || c.take(' '))) return false;
    c.skip_spaces();
    const long long hh = c.digits(1, 2);
    if (hh < 0 || !c.take(':')) return false;
    const long long mm = c.digits(2, 2);
    long long ss = 0, ms = 0;
    if (c.take(':')) {
        ss = c.digits(2, 2);
        if (c.take('.') || c.take(',')) {
            int nd = 0;
            const long long frac = c.digits(1, 9, &nd);
            if (frac < 0) return false;
            ms = frac;
            for (int k = nd; k < 3; ++k) ms *= 10;
            for (int k = 3; k < nd; ++k) ms /= 10;
        }
    }
    if (hh > 23 || mm < 0 || mm > 59 || ss < 0 || ss > 60) return false;
    if (ss == 60) ss = 59;  // leap second

    c.skip_spaces();
    bool zoned = false;
    long long zone_min = 0;
    if (c.take('Z') || c.take('z')) {
        zoned = true;
    } else if (c.peek() == '+' || c.peek() == '-') {
        const int sign = c.peek() == '-' ? -1 : 1;
        ++c.i;
        const long long zh = c.digits(2, 2);
        long long zm = 0;  // +hh:mm, +hhmm or +hh (PostgreSQL "...-03")
        if (c.take(':') || std::isdigit(static_cast<unsigned char>(c.peek()))) zm = c.digits(2, 2);
        if (zh < 0 || zm < 0 || zh > 14 || zm > 59) return false;
        zoned = true;
        zone_min = sign * (zh * 60 + zm);
    }
    c.skip_spaces();
    if (!c.eof()) return false;

    long long days = days_from_civil(static_cast<int>(y), static_cast<int>(mo), static_cast<int>(d));
    long long minute = hh * 60 + mm;
    if (zoned) {  // local(given zone) -> UTC -> exchange local
        minute += exchange_utc_offset_min - zone_min;
        days += (minute >= 0 ? minute : minute - 1439) / 1440;
        minute = ((minute % 1440) + 1440) % 1440;
    }
    int yy, mmo, dd;
    civil_from_days(days, yy, mmo, dd);
    out.ymd = yy * 10000 + mmo * 100 + dd;
    out.minute = static_cast<int>(minute);
    out.millis = static_cast<int>(ss * 1000 + ms);
    return true;
}

bool is_b3_holiday(int ymd) {
    static const int h[] = {
        20250101, 20250303, 20250304, 20250418, 20250421, 20250501, 20250619, 20251120, 20251224, 20251225,
        20251231, 20260101, 20260216, 20260217, 20260403, 20260421, 20260501, 20260604, 20260907, 20261012,
        20261102, 20261120, 20261224, 20261225, 20261231, 20270101, 20270208, 20270209, 20270326, 20270421,
        20270527, 20270907, 20271012, 20271102, 20271115, 20271224, 20271231};
    return std::binary_search(std::begin(h), std::end(h), ymd);
}

bool parse_timestamp(const std::string& s, int& ymd, int& minute, int* second) {
    TimePoint tp;
    if (!parse_time(s, tp)) return false;  // zoned inputs are converted to B3 time (UTC-3)
    ymd = tp.ymd;
    minute = tp.minute;
    if (second) *second = tp.millis / 1000;
    return true;
}

// ============================================================== bars

int bar_slot(int minute_of_day, const SessionSpec& s) {
    const int since_open = minute_of_day - s.open_minute;
    if (since_open <= 0 || s.bar_minutes <= 0 || since_open % s.bar_minutes != 0) return -1;
    const int k = since_open / s.bar_minutes - 1;
    return (k >= 0 && k < s.bars_per_day) ? k : -1;
}

BarPanel build_panel(const std::vector<BarRow>& rows, const std::vector<std::string>& tickers) {
    BarPanel p;
    p.tickers = tickers;
    std::unordered_map<std::string, size_t> tk_index;
    for (size_t i = 0; i < tickers.size(); ++i) tk_index[upper(trim(tickers[i]))] = i;

    std::map<long long, TimePoint> grid;  // built from rows of known tickers only
    std::vector<long long> keys(rows.size());
    std::vector<long> asset(rows.size(), -1);
    for (size_t r = 0; r < rows.size(); ++r) {
        const auto ki = tk_index.find(upper(trim(rows[r].ticker)));
        if (ki == tk_index.end()) continue;
        asset[r] = static_cast<long>(ki->second);
        TimePoint tp = rows[r].time;
        tp.millis = 0;  // bars are identified by their closing minute (as their slot is)
        keys[r] = tp.key();
        grid.emplace(keys[r], tp);
    }
    std::unordered_map<long long, size_t> key_index;
    key_index.reserve(grid.size());
    for (const auto& kv : grid) {
        key_index[kv.first] = p.timestamps.size();
        p.timestamps.push_back(kv.second.str());
        p.time_key.push_back(kv.first);
        p.date.push_back(kv.second.ymd);
        p.minute.push_back(kv.second.minute);
    }
    const size_t T = p.timestamps.size(), n = tickers.size();
    for (auto* m : {&p.open, &p.high, &p.low, &p.close, &p.volume}) m->assign(T * n, kNaN);
    std::vector<char> seen(T * n, 0);
    for (size_t r = 0; r < rows.size(); ++r) {
        const BarRow& row = rows[r];
        ++p.rows_read;
        if (asset[r] < 0) {
            ++p.unknown_tickers;
            continue;
        }
        const size_t cell = key_index[keys[r]] * n + static_cast<size_t>(asset[r]);
        if (seen[cell]) {
            ++p.duplicates;
            continue;
        }
        seen[cell] = 1;
        p.open[cell] = std::isfinite(row.open) ? row.open : row.close;  // missing open -> close
        p.high[cell] = row.high;
        p.low[cell] = row.low;
        p.close[cell] = row.close;
        p.volume[cell] = row.volume;
    }
    return p;
}

BarPanel load_bars_csv(const std::string& path, const std::vector<std::string>& tickers,
                       int exchange_utc_offset_min) {
    const CsvTable csv = read_csv(path);
    int c_ts = csv.column("timestamp");
    if (c_ts < 0) c_ts = csv.column("datetime");
    if (c_ts < 0) c_ts = csv.column("date_time");
    const int c_date = csv.column("date"), c_time = csv.column("time");
    if (c_ts < 0 && c_date < 0 && c_time >= 0) c_ts = c_time;  // "time" holding a full timestamp
    int c_tk = csv.column("ticker");
    if (c_tk < 0) c_tk = csv.column("symbol");
    if (c_tk < 0) c_tk = csv.column("asset");
    const int c_o = csv.column("open"), c_h = csv.column("high"), c_l = csv.column("low");
    const int c_c = csv.column("close"), c_v = csv.column("volume");
    const bool split_dt = c_ts < 0 && c_date >= 0 && c_time >= 0;
    if ((c_ts < 0 && !split_dt) || c_tk < 0 || c_c < 0)
        throw std::runtime_error(path + ": needs timestamp (or date + time), ticker and close columns");

    auto field = [](const std::vector<std::string>& row, int c) -> std::string {
        return (c >= 0 && static_cast<size_t>(c) < row.size()) ? row[static_cast<size_t>(c)] : std::string();
    };
    std::vector<BarRow> rows;
    rows.reserve(csv.rows.size());
    size_t bad = 0;
    for (const auto& row : csv.rows) {
        BarRow b;
        const std::string ts = split_dt ? field(row, c_date) + " " + field(row, c_time) : field(row, c_ts);
        if (!parse_time(ts, b.time, exchange_utc_offset_min) || !parse_number(field(row, c_c), b.close) ||
            !parse_number(field(row, c_o), b.open) || !parse_number(field(row, c_h), b.high) ||
            !parse_number(field(row, c_l), b.low) || !parse_number(field(row, c_v), b.volume)) {
            ++bad;
            continue;
        }
        b.ticker = field(row, c_tk);
        rows.push_back(std::move(b));
    }
    BarPanel p = build_panel(rows, tickers);
    p.rows_read += bad;
    p.bad_rows += bad;
    return p;
}

SessionSpec detect_session(const BarPanel& p) {
    SessionSpec spec;
    const size_t T = p.num_bars();
    if (T < 2) return spec;
    std::map<int, int> spacing, first, last;
    size_t day_start = 0;
    for (size_t t = 1; t <= T; ++t) {
        if (t == T || p.date[t] != p.date[day_start]) {
            ++first[p.minute[day_start]];
            ++last[p.minute[t - 1]];
            for (size_t u = day_start + 1; u < t; ++u) ++spacing[p.minute[u] - p.minute[u - 1]];
            day_start = t;
        }
    }
    auto mode = [](const std::map<int, int>& m) {
        int best = -1, cnt = 0;
        for (const auto& kv : m)
            if (kv.second > cnt) {
                best = kv.first;
                cnt = kv.second;
            }
        return best;
    };
    const int bar = mode(spacing), f = mode(first), l = mode(last);
    if (bar <= 0 || f < 0 || l < f) return spec;
    spec.bar_minutes = bar;
    spec.open_minute = f - bar;
    spec.bars_per_day = (l - spec.open_minute) / bar;
    if (spec.bars_per_day < 1) return SessionSpec{};
    return spec;
}

void bar_returns(const BarPanel& p, const SessionSpec& session, std::vector<double>& ret,
                 std::vector<int>& slot) {
    const size_t T = p.num_bars(), n = p.num_assets();
    ret.assign(T * n, kNaN);
    slot.assign(T, -1);
    for (size_t t = 0; t < T; ++t) slot[t] = bar_slot(p.minute[t], session);

    auto valid = [](double v) { return std::isfinite(v) && v > 0.0; };
    for (size_t i = 0; i < n; ++i) {
        double ref = kNaN;  // last valid on-grid price of the current session
        int ref_slot = -1;  // slot whose close `ref` is (-1 = the session open)
        int day = -1;
        for (size_t t = 0; t < T; ++t) {
            if (slot[t] < 0) continue;  // off-grid rows carry no bar return
            if (p.date[t] != day) {     // new session: start from this bar's opening price
                day = p.date[t];
                ref = p.at(p.open, t, i);
                ref_slot = slot[t] - 1;
            }
            const double c = p.at(p.close, t, i);
            if (!valid(c)) continue;
            if (valid(ref)) {
                // spans counted in grid slots: a return over k bars is scaled by 1/sqrt(k)
                const int span = std::max(1, slot[t] - ref_slot);
                ret[t * n + i] = std::log(c / ref) / std::sqrt(static_cast<double>(span));
            } else {
                const double o = p.at(p.open, t, i);  // fall back to the bar's own open
                if (valid(o)) ret[t * n + i] = std::log(c / o);
            }
            ref = c;
            ref_slot = slot[t];
        }
    }
}

void overnight_returns(const BarPanel& p, const std::vector<int>& slot, std::vector<double>& gaps) {
    const size_t T = p.num_bars(), n = p.num_assets();
    gaps.assign(T * n, kNaN);
    std::vector<double> last_close(n, kNaN);
    auto valid = [](double v) { return std::isfinite(v) && v > 0.0; };
    int day = -1;
    for (size_t t = 0; t < T; ++t) {
        if (slot[t] < 0) continue;  // off-grid rows are ignored (as in bar_returns)
        const bool first_bar = p.date[t] != day;
        if (first_bar) day = p.date[t];
        for (size_t i = 0; i < n; ++i) {
            // a session that does not open at slot 0 (late opening) has no clean gap: the
            // move from the previous close would include unobserved intraday trading
            if (first_bar && slot[t] == 0) {
                const double o = p.at(p.open, t, i);
                if (valid(o) && valid(last_close[i])) gaps[t * n + i] = std::log(o / last_close[i]);
            }
            const double c = p.at(p.close, t, i);
            if (valid(c)) last_close[i] = c;
        }
    }
}

}  // namespace garcht
