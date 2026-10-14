#include "ewmar/market_data.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace ewmar {

namespace {

std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

std::string upper(std::string s) {
    for (char& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

// days since 1970-01-01 of a proleptic Gregorian date
long long days_from_civil(int y, int m, int d) {
    y -= m <= 2;
    const long long era = (y >= 0 ? y : y - 399) / 400;
    const long long yoe = y - era * 400;
    const long long doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const long long doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + doe - 719468;
}

int civil_from_days(long long z) {  // -> YYYYMMDD
    z += 719468;
    const long long era = (z >= 0 ? z : z - 146096) / 146097;
    const long long doe = z - era * 146097;
    const long long yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    const long long doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const long long mp = (5 * doy + 2) / 153;
    const int d = static_cast<int>(doy - (153 * mp + 2) / 5 + 1);
    const int m = static_cast<int>(mp < 10 ? mp + 3 : mp - 9);
    const int y = static_cast<int>(yoe + era * 400 + (m <= 2));
    return y * 10000 + m * 100 + d;
}

bool valid_date(int y, int m, int d) {
    static const int dm[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (y < 1900 || y > 2200 || m < 1 || m > 12 || d < 1) return false;
    const bool leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    return d <= (m == 2 && leap ? 29 : dm[m - 1]);
}

long long ymd_days(int ymd) { return days_from_civil(ymd / 10000, (ymd / 100) % 100, ymd % 100); }

// reads exactly `n` digits at s[i] (advancing i)
bool digits(const std::string& s, size_t& i, int n, int& out) {
    out = 0;
    for (int k = 0; k < n; ++k, ++i) {
        if (i >= s.size() || !std::isdigit(static_cast<unsigned char>(s[i]))) return false;
        out = out * 10 + (s[i] - '0');
    }
    return true;
}

// date at the start of s: YYYY-MM-DD, YYYY/MM/DD or DD/MM/YYYY
bool read_date(const std::string& s, size_t& i, int& ymd) {
    int a = 0, b = 0, c = 0;
    const size_t start = i;
    if (digits(s, i, 4, a)) {
        if (i >= s.size() || (s[i] != '-' && s[i] != '/')) return false;
        const char sep = s[i++];
        if (!digits(s, i, 2, b) || i >= s.size() || s[i] != sep) return false;
        ++i;
        if (!digits(s, i, 2, c) || !valid_date(a, b, c)) return false;
        ymd = a * 10000 + b * 100 + c;
        return true;
    }
    i = start;
    if (!digits(s, i, 2, a) || i >= s.size() || s[i] != '/') return false;
    ++i;
    if (!digits(s, i, 2, b) || i >= s.size() || s[i] != '/') return false;
    ++i;
    if (!digits(s, i, 4, c) || !valid_date(c, b, a)) return false;
    ymd = c * 10000 + b * 100 + a;
    return true;
}

}  // namespace

// ============================================================== CSV

int CsvTable::column(const std::string& name) const {
    const std::string u = upper(name);
    for (size_t k = 0; k < header.size(); ++k)
        if (upper(trim(header[k])) == u) return static_cast<int>(k);
    return -1;
}

CsvTable parse_csv(const std::string& text_in) {
    std::string text = text_in;
    if (text.size() >= 3 && static_cast<unsigned char>(text[0]) == 0xEF && static_cast<unsigned char>(text[1]) == 0xBB &&
        static_cast<unsigned char>(text[2]) == 0xBF)
        text.erase(0, 3);
    // delimiter: the most frequent of , ; tab on the first non-comment line (outside quotes)
    char delim = ',';
    {
        size_t p = 0;
        while (p < text.size()) {
            const size_t e = text.find('\n', p);
            const std::string line = text.substr(p, e == std::string::npos ? std::string::npos : e - p);
            if (!trim(line).empty() && trim(line)[0] != '#') {
                int cnt[3] = {0, 0, 0};
                bool q = false;
                for (char ch : line) {
                    if (ch == '"') q = !q;
                    else if (!q && ch == ',') ++cnt[0];
                    else if (!q && ch == ';') ++cnt[1];
                    else if (!q && ch == '\t') ++cnt[2];
                }
                if (cnt[1] > cnt[0] && cnt[1] >= cnt[2]) delim = ';';
                else if (cnt[2] > cnt[0] && cnt[2] > cnt[1]) delim = '\t';
                break;
            }
            if (e == std::string::npos) break;
            p = e + 1;
        }
    }
    CsvTable t;
    std::vector<std::string> rec;
    std::string field;
    bool quoted = false, at_record_start = true, comment = false, header_done = false;
    size_t line = 1, rec_line = 1;
    auto end_record = [&]() {
        rec.push_back(field);
        field.clear();
        bool blank = true;
        for (const auto& f : rec)
            if (!trim(f).empty()) blank = false;
        if (!blank) {
            if (!header_done) {
                t.header = rec;
                header_done = true;
            } else {
                t.rows.push_back(rec);
                t.line.push_back(rec_line);
            }
        }
        rec.clear();
        at_record_start = true;
    };
    for (size_t i = 0; i < text.size(); ++i) {
        const char ch = text[i];
        if (at_record_start) {
            rec_line = line;
            at_record_start = false;
            comment = ch == '#';
        }
        if (comment) {
            if (ch == '\n') {
                ++line;
                comment = false;
                at_record_start = true;
                rec.clear();
                field.clear();
            }
            continue;
        }
        if (quoted) {
            if (ch == '"') {
                if (i + 1 < text.size() && text[i + 1] == '"') {
                    field += '"';
                    ++i;
                } else {
                    quoted = false;
                }
            } else {
                if (ch == '\n') ++line;
                field += ch;
            }
            continue;
        }
        if (ch == '"') quoted = true;
        else if (ch == delim) {
            rec.push_back(field);
            field.clear();
        } else if (ch == '\r') {
            // CRLF: the '\n' ends the record
        } else if (ch == '\n') {
            end_record();
            ++line;
        } else {
            field += ch;
        }
    }
    if (!at_record_start && !comment) end_record();
    return t;
}

CsvTable read_csv(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path + ": " + std::strerror(errno));
    std::ostringstream ss;
    ss << in.rdbuf();
    return parse_csv(ss.str());
}

bool parse_number(const std::string& s_in, double& out) {
    std::string s;
    for (char ch : s_in)
        if (!std::isspace(static_cast<unsigned char>(ch))) s += ch;
    const std::string u = upper(s);
    if (s.empty() || u == "NA" || u == "NAN" || u == "NULL" || u == "-" || u == "N/A") {
        out = kNaN;
        return true;
    }
    const size_t ndot = static_cast<size_t>(std::count(s.begin(), s.end(), '.'));
    const size_t ncom = static_cast<size_t>(std::count(s.begin(), s.end(), ','));
    if (ndot > 0 && ncom > 0) {  // the last separator is the decimal one
        const char dec = s.find_last_of(".,") == s.rfind('.') ? '.' : ',';
        const char thou = dec == '.' ? ',' : '.';
        s.erase(std::remove(s.begin(), s.end(), thou), s.end());
        std::replace(s.begin(), s.end(), ',', '.');
    } else if (ncom == 1) {
        std::replace(s.begin(), s.end(), ',', '.');
    } else if (ncom > 1) {
        s.erase(std::remove(s.begin(), s.end(), ','), s.end());
    } else if (ndot > 1 && u.find('E') == std::string::npos) {
        s.erase(std::remove(s.begin(), s.end(), '.'), s.end());
    }
    char* end = nullptr;
    errno = 0;
    const double v = std::strtod(s.c_str(), &end);
    if (end == s.c_str() || *end != '\0' || errno == ERANGE) return false;
    out = v;
    return true;
}

// ============================================================== time

long long TimePoint::key() const { return ymd_days(ymd) * 86400LL + minute * 60LL + second; }

std::string TimePoint::str() const {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%04d-%02d-%02d %02d:%02d:%02d", ymd / 10000, (ymd / 100) % 100, ymd % 100,
                  minute / 60, minute % 60, second);
    return buf;
}

bool parse_date(const std::string& s_in, int& ymd) {
    const std::string s = trim(s_in);
    size_t i = 0;
    return read_date(s, i, ymd) && i == s.size();
}

bool parse_time(const std::string& s_in, TimePoint& out, int utc_offset_min) {
    const std::string s = trim(s_in);
    size_t i = 0;
    int ymd = 0;
    if (!read_date(s, i, ymd)) return false;
    int hh = 0, mm = 0, ss = 0;
    if (i < s.size() && (s[i] == ' ' || s[i] == 'T' || s[i] == 't')) {
        while (i < s.size() && (s[i] == ' ' || s[i] == 'T' || s[i] == 't')) ++i;
        if (!digits(s, i, 2, hh) || i >= s.size() || s[i] != ':') return false;
        ++i;
        if (!digits(s, i, 2, mm)) return false;
        if (i < s.size() && s[i] == ':') {
            ++i;
            if (!digits(s, i, 2, ss)) return false;
            if (i < s.size() && s[i] == '.') {  // fraction of a second: ignored
                ++i;
                while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) ++i;
            }
        }
        if (hh > 23 || mm > 59 || ss > 60) return false;
    }
    while (i < s.size() && s[i] == ' ') ++i;
    long long t = ymd_days(ymd) * 86400LL + hh * 3600LL + mm * 60LL + std::min(ss, 59);
    if (i < s.size()) {  // zone suffix
        int zone_min = 0;
        if (s[i] == 'Z' || s[i] == 'z') {
            ++i;
        } else if (s[i] == '+' || s[i] == '-') {
            const int sign = s[i] == '-' ? -1 : 1;
            ++i;
            int zh = 0, zm = 0;
            if (!digits(s, i, 2, zh)) return false;
            if (i < s.size() && s[i] == ':') ++i;
            if (i < s.size() && !digits(s, i, 2, zm)) return false;
            zone_min = sign * (zh * 60 + zm);
        } else {
            return false;
        }
        if (i != s.size()) return false;
        t += (utc_offset_min - zone_min) * 60LL;  // to UTC, then to the exchange's local time
    }
    long long days = t / 86400, rem = t % 86400;
    if (rem < 0) {
        rem += 86400;
        --days;
    }
    out.ymd = civil_from_days(days);
    out.minute = static_cast<int>(rem / 60);
    out.second = static_cast<int>(rem % 60);
    return true;
}

int weekday(int ymd) {
    const long long d = ymd_days(ymd);  // 1970-01-01 was a Thursday (3)
    return static_cast<int>(((d + 3) % 7 + 7) % 7);
}

int next_day(int ymd) { return civil_from_days(ymd_days(ymd) + 1); }

bool is_b3_holiday(int ymd) {
    // national holidays and exchange closures on weekdays (B3 calendar)
    static const int days[] = {
        20250101, 20250303, 20250304, 20250418, 20250421, 20250501, 20250619, 20251120, 20251224, 20251225,
        20251231, 20260101, 20260216, 20260217, 20260403, 20260421, 20260501, 20260604, 20260907, 20261012,
        20261102, 20261120, 20261224, 20261225, 20261231, 20270101, 20270208, 20270209, 20270326, 20270421,
        20270527, 20270907, 20271012, 20271102, 20271115, 20271224, 20271231};
    return std::binary_search(std::begin(days), std::end(days), ymd);
}

bool is_business_day(int ymd) { return weekday(ymd) < 5 && !is_b3_holiday(ymd); }

// ============================================================== session

int SessionSpec::slot_of(int minute_of_day) const {
    const int since = minute_of_day - open_minute;
    if (bar_minutes <= 0 || since <= 0 || since % bar_minutes != 0) return -1;
    const int k = since / bar_minutes - 1;
    return k < bars_per_day ? k : -1;
}

SessionSpec detect_session(const std::vector<RawRow>& rows) {
    std::map<int, std::vector<int>> by_day;  // day -> minutes with prints
    for (const RawRow& r : rows) by_day[r.time.ymd].push_back(r.time.minute);
    std::map<int, int> spacing, first, last;
    for (auto& kv : by_day) {
        std::vector<int>& m = kv.second;
        std::sort(m.begin(), m.end());
        m.erase(std::unique(m.begin(), m.end()), m.end());
        ++first[m.front()];
        ++last[m.back()];
        for (size_t k = 1; k < m.size(); ++k) ++spacing[m[k] - m[k - 1]];
    }
    auto mode = [](const std::map<int, int>& h) {
        int best = -1, cnt = 0;
        for (const auto& kv : h)
            if (kv.second > cnt) {
                best = kv.first;
                cnt = kv.second;
            }
        return best;
    };
    SessionSpec s;
    const int bar = mode(spacing), f = mode(first), l = mode(last);
    if (bar <= 0 || f < 0 || l < f) return s;
    s.bar_minutes = bar;
    s.open_minute = f - bar;
    s.bars_per_day = (l - s.open_minute) / bar;
    if (s.bars_per_day < 1) return SessionSpec{};
    return s;
}

// ============================================================== feed

FeedFile load_feed(const std::string& path, const std::vector<std::string>& tickers, int utc_offset_min) {
    const CsvTable csv = read_csv(path);
    int c_ts = csv.column("timestamp");
    if (c_ts < 0) c_ts = csv.column("datetime");
    const int c_date = csv.column("date"), c_time = csv.column("time");
    int c_tk = csv.column("ticker");
    if (c_tk < 0) c_tk = csv.column("symbol");
    const int c_o = csv.column("open"), c_h = csv.column("high"), c_l = csv.column("low");
    const int c_c = csv.column("close"), c_v = csv.column("volume");
    const bool split = c_ts < 0 && c_date >= 0 && c_time >= 0;
    if ((c_ts < 0 && !split) || c_tk < 0 || c_c < 0)
        throw std::runtime_error(path + ": needs timestamp (or date + time), ticker and close columns");

    std::unordered_map<std::string, int> index;
    for (size_t i = 0; i < tickers.size(); ++i) index[upper(trim(tickers[i]))] = static_cast<int>(i);
    auto field = [](const std::vector<std::string>& row, int c) {
        return c >= 0 && static_cast<size_t>(c) < row.size() ? row[static_cast<size_t>(c)] : std::string();
    };
    FeedFile f;
    f.tickers = tickers;
    f.lines = csv.rows.size();
    f.rows.reserve(csv.rows.size());
    for (size_t r = 0; r < csv.rows.size(); ++r) {
        const auto& row = csv.rows[r];
        RawRow b;
        b.line = csv.line[r];
        const std::string ts = split ? field(row, c_date) + " " + field(row, c_time) : field(row, c_ts);
        if (!parse_time(ts, b.time, utc_offset_min) || !parse_number(field(row, c_c), b.close) ||
            !parse_number(field(row, c_o), b.open) || !parse_number(field(row, c_h), b.high) ||
            !parse_number(field(row, c_l), b.low) || !parse_number(field(row, c_v), b.volume)) {
            ++f.bad_rows;
            continue;
        }
        const auto it = index.find(upper(trim(field(row, c_tk))));
        if (it == index.end()) {
            ++f.unknown_rows;
            continue;
        }
        b.asset = it->second;
        b.time.second = 0;  // a bar is identified by its closing minute
        if (!valid_price(b.open)) b.open = b.close;
        f.rows.push_back(b);
    }
    return f;
}

Panel sorted_panel(const FeedFile& feed, const SessionSpec& s, PanelStats* stats) {
    PanelStats st;
    Panel p;
    p.tickers = feed.tickers;
    const size_t n = feed.tickers.size();
    std::map<long long, TimePoint> grid;
    for (const RawRow& r : feed.rows) {
        if (s.slot_of(r.time.minute) < 0) {
            ++st.off_grid;
            continue;
        }
        grid.emplace(r.time.key(), r.time);
    }
    std::unordered_map<long long, size_t> row_of;
    for (const auto& kv : grid) {
        row_of[kv.first] = p.stamp.size();
        p.stamp.push_back(kv.second.str());
        p.key.push_back(kv.first);
        p.ymd.push_back(kv.second.ymd);
        p.minute.push_back(kv.second.minute);
        p.slot.push_back(s.slot_of(kv.second.minute));
    }
    const size_t T = p.stamp.size();
    for (auto* m : {&p.open, &p.high, &p.low, &p.close, &p.volume}) m->assign(T * n, kNaN);
    std::vector<char> seen(T * n, 0);
    for (const RawRow& r : feed.rows) {
        const auto it = row_of.find(r.time.key());
        if (it == row_of.end()) continue;
        const size_t cell = it->second * n + static_cast<size_t>(r.asset);
        if (seen[cell]) {
            ++st.duplicates;
            continue;
        }
        seen[cell] = 1;
        p.open[cell] = r.open;
        p.high[cell] = r.high;
        p.low[cell] = r.low;
        p.close[cell] = r.close;
        p.volume[cell] = r.volume;
    }
    if (stats) *stats = st;
    return p;
}

// ============================================================== returns

double ReturnState::step(int ymd, int slot, double open, double close, double* gap) {
    if (gap) *gap = kNaN;
    if (ymd != day) {  // the session's first bar: start from this bar's open
        day = ymd;
        if (gap && slot == 0 && valid_price(open) && valid_price(last_close)) *gap = std::log(open / last_close);
        ref = open;
        ref_slot = slot - 1;
    }
    if (!valid_price(close)) return kNaN;
    double r = kNaN;
    if (valid_price(ref)) r = std::log(close / ref) / std::sqrt(static_cast<double>(std::max(1, slot - ref_slot)));
    else if (valid_price(open)) r = std::log(close / open);
    ref = close;
    ref_slot = slot;
    last_close = close;
    return r;
}

void panel_returns(const Panel& p, std::vector<double>& ret, std::vector<double>& gaps) {
    const size_t T = p.bars(), n = p.assets();
    ret.assign(T * n, kNaN);
    gaps.assign(T * n, kNaN);
    std::vector<ReturnState> st(n);
    for (size_t t = 0; t < T; ++t)
        for (size_t i = 0; i < n; ++i)
            ret[t * n + i] = st[i].step(p.ymd[t], p.slot[t], p.at(p.open, t, i), p.at(p.close, t, i), &gaps[t * n + i]);
}

// ============================================================== portfolio inputs

std::vector<Fill> load_fills(const std::string& path, int utc_offset_min) {
    const CsvTable csv = read_csv(path);
    const int c_ts = csv.column("timestamp"), c_tk = csv.column("ticker"), c_side = csv.column("side"),
              c_q = csv.column("quantity");
    if (c_ts < 0 || c_tk < 0 || c_q < 0) throw std::runtime_error(path + ": needs timestamp, ticker and quantity columns");
    std::vector<Fill> out;
    for (size_t r = 0; r < csv.rows.size(); ++r) {
        const auto& row = csv.rows[r];
        auto field = [&](int c) { return c >= 0 && static_cast<size_t>(c) < row.size() ? row[static_cast<size_t>(c)] : std::string(); };
        Fill f;
        double q = 0.0;
        if (!parse_time(field(c_ts), f.time, utc_offset_min) || !parse_number(field(c_q), q) || !std::isfinite(q))
            throw std::runtime_error(path + ": invalid row at line " + std::to_string(csv.line[r]));
        const std::string side = upper(trim(field(c_side)));
        if (side == "SELL" || side == "S" || side == "V" || side == "VENDA") q = -std::fabs(q);
        else if (side == "BUY" || side == "B" || side == "C" || side == "COMPRA") q = std::fabs(q);
        else if (!side.empty()) throw std::runtime_error(path + ": unknown side '" + side + "' at line " + std::to_string(csv.line[r]));
        f.ticker = trim(field(c_tk));
        f.quantity = q;
        out.push_back(f);
    }
    std::stable_sort(out.begin(), out.end(), [](const Fill& a, const Fill& b) { return a.time.key() < b.time.key(); });
    return out;
}

std::vector<OptionContract> load_options(const std::string& path) {
    const CsvTable csv = read_csv(path);
    const int c_id = csv.column("id"), c_u = csv.column("underlying"), c_t = csv.column("type"), c_k = csv.column("strike"),
              c_e = csv.column("expiry"), c_v = csv.column("implied_vol"), c_r = csv.column("rate"),
              c_m = csv.column("multiplier"), c_q = csv.column("quantity");
    if (c_u < 0 || c_t < 0 || c_k < 0 || c_e < 0 || c_v < 0 || c_q < 0)
        throw std::runtime_error(path + ": needs underlying, type, strike, expiry, implied_vol and quantity columns");
    std::vector<OptionContract> out;
    for (size_t r = 0; r < csv.rows.size(); ++r) {
        const auto& row = csv.rows[r];
        auto field = [&](int c) { return c >= 0 && static_cast<size_t>(c) < row.size() ? row[static_cast<size_t>(c)] : std::string(); };
        OptionContract o;
        o.id = c_id >= 0 ? trim(field(c_id)) : "OPT" + std::to_string(r + 1);
        o.underlying = trim(field(c_u));
        const std::string type = upper(trim(field(c_t)));
        o.call = type == "CALL" || type == "C";
        const bool ok_type = o.call || type == "PUT" || type == "P";
        double rate = 0.0, mult = 1.0;
        const bool ok = ok_type && parse_number(field(c_k), o.strike) && parse_date(field(c_e), o.expiry_ymd) &&
                        parse_number(field(c_v), o.implied_vol) && parse_number(field(c_q), o.quantity) &&
                        (c_r < 0 || parse_number(field(c_r), rate)) && (c_m < 0 || parse_number(field(c_m), mult));
        if (!ok || !(o.strike > 0.0) || !(o.implied_vol > 0.0) || !std::isfinite(o.quantity))
            throw std::runtime_error(path + ": invalid option at line " + std::to_string(csv.line[r]));
        o.rate = std::isfinite(rate) ? rate : 0.0;
        o.multiplier = std::isfinite(mult) && mult > 0.0 ? mult : 1.0;
        out.push_back(o);
    }
    return out;
}

}  // namespace ewmar
