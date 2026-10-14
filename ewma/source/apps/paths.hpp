// Default folders, so that the programs run without path arguments:
//
//   <project folder>/
//   ├── ewma/      this module: bin/ (programs), results/ (outputs), source/
//   └── data/      the datasets: synthetic/, b3_real/
//
// A dataset given by name is looked up as data/<name> from the current folder and from the
// program's folder upwards; results go to ewma/results/<name>. Also switches the Windows console
// to UTF-8 while the program runs (paths with accents).
#pragma once

#include <filesystem>
#include <string>
#include <system_error>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace paths {

namespace fs = std::filesystem;

inline fs::path program_dir() {
    std::error_code ec;
#if defined(_WIN32)
    wchar_t buf[32768];
    const DWORD len = GetModuleFileNameW(nullptr, buf, 32768);
    if (len > 0 && len < 32768) return fs::path(std::wstring(buf, len)).parent_path();
#elif defined(__linux__)
    const fs::path p = fs::read_symlink("/proc/self/exe", ec);
    if (!ec) return p.parent_path();
#endif
    return fs::current_path(ec);
}

// The ewma folder: the first ancestor of the program with source/CMakeLists.txt, or bin/ + README.md.
inline fs::path module_root() {
    std::error_code ec;
    fs::path d = program_dir();
    for (int k = 0; k < 6 && !d.empty(); ++k) {
        if (fs::exists(d / "source" / "CMakeLists.txt", ec) || (fs::is_directory(d / "bin", ec) && fs::exists(d / "README.md", ec)))
            return d;
        if (d.parent_path() == d) break;
        d = d.parent_path();
    }
    return fs::current_path(ec);
}

inline std::string resolve_dataset(const std::string& arg) {
    std::error_code ec;
    if (fs::is_directory(arg, ec)) return arg;
    for (const fs::path& start : {fs::current_path(ec), program_dir()}) {
        fs::path d = start;
        for (int k = 0; k < 7 && !d.empty(); ++k) {
            if (fs::is_directory(d / "data" / arg, ec)) return (d / "data" / arg).string();
            if (d.parent_path() == d) break;
            d = d.parent_path();
        }
    }
    return arg;
}

inline std::string results_dir(const std::string& name) { return (module_root() / "results" / name).string(); }

// Path as shown in reports: relative to the project folder (data/..., ewma/results/...).
inline std::string display(const std::string& p) {
    std::error_code ec;
    const fs::path rel = fs::relative(fs::absolute(p, ec), module_root().parent_path(), ec);
    if (ec || rel.empty() || *rel.begin() == "..") return p;
    return rel.generic_string();
}

struct Utf8Console {
#if defined(_WIN32)
    UINT saved = GetConsoleOutputCP();
    Utf8Console() { SetConsoleOutputCP(CP_UTF8); }
    ~Utf8Console() {
        if (saved) SetConsoleOutputCP(saved);
    }
#else
    Utf8Console() {}
    ~Utf8Console() {}
#endif
    Utf8Console(const Utf8Console&) = delete;
    Utf8Console& operator=(const Utf8Console&) = delete;
};

}  // namespace paths
