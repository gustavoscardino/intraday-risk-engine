// Default folders, so that the programs run with no path arguments, and UTF-8 console output.
//
// Expected layout (the dataset folder sits next to the garch_t folder):
//
//   <parent folder>/
//   ├── garch_t/          this project: bin/ (programs), results/ (outputs), source/ (sources)
//   └── data/             the dataset: synthetic/, b3_real/
//
// A dataset given by name ("synthetic", "b3_real") is searched as "data/<name>" in the
// current folder, the executable's folder and their parents. Outputs go to
// "garch_t/results/<name>".
#pragma once

#include <filesystem>
#include <initializer_list>
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

// Folder of the running executable (falls back to the current folder).
inline fs::path exe_dir() {
    std::error_code ec;
#if defined(_WIN32)
    wchar_t buf[32768];
    const DWORD len = GetModuleFileNameW(nullptr, buf, static_cast<DWORD>(sizeof buf / sizeof buf[0]));
    if (len > 0 && len < sizeof buf / sizeof buf[0]) return fs::path(std::wstring(buf, len)).parent_path();
#elif defined(__linux__)
    const fs::path p = fs::read_symlink("/proc/self/exe", ec);
    if (!ec) return p.parent_path();
#endif
    return fs::current_path(ec);
}

// garch_t folder: the first ancestor of the executable holding source/ or bin/ next to README.md.
inline fs::path project_root() {
    std::error_code ec;
    fs::path d = exe_dir();
    for (int up = 0; up < 5 && !d.empty(); ++up) {
        if (fs::exists(d / "source" / "CMakeLists.txt", ec) ||
            (fs::is_directory(d / "bin", ec) && fs::exists(d / "README.md", ec)))
            return d;
        const fs::path parent = d.parent_path();
        if (parent == d) break;
        d = parent;
    }
    return fs::current_path(ec);
}

// "data/<name>" found from the current folder or the executable's folder upwards; empty if none.
inline fs::path find_dataset(const std::string& name) {
    std::error_code ec;
    for (const fs::path& start : {fs::current_path(ec), exe_dir()}) {
        fs::path d = start;
        for (int up = 0; up < 6 && !d.empty(); ++up) {
            if (fs::is_directory(d / "data" / name, ec)) return d / "data" / name;
            if (d.filename() == "data" && fs::is_directory(d / name, ec)) return d / name;
            const fs::path parent = d.parent_path();
            if (parent == d) break;
            d = parent;
        }
    }
    return {};
}

// A --data argument: an existing folder as given, else a dataset name searched as above.
inline std::string resolve_dataset(const std::string& arg) {
    std::error_code ec;
    if (fs::is_directory(arg, ec)) return arg;
    const fs::path p = find_dataset(arg);
    return p.empty() ? arg : p.string();
}

// Default output folder: garch_t/results/<name>.
inline std::string results_dir(const std::string& name) { return (project_root() / "results" / name).string(); }

// While alive, the Windows console shows UTF-8 text (paths with accents); restored afterwards.
struct Utf8Console {
#if defined(_WIN32)
    UINT saved = GetConsoleOutputCP();
    Utf8Console() { SetConsoleOutputCP(CP_UTF8); }
    ~Utf8Console() {
        if (saved != 0) SetConsoleOutputCP(saved);
    }
#else
    Utf8Console() {}   // terminals are UTF-8 already
    ~Utf8Console() {}  // (user-provided: not an "unused variable")
#endif
    Utf8Console(const Utf8Console&) = delete;
    Utf8Console& operator=(const Utf8Console&) = delete;
};

}  // namespace paths
