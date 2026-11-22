"""Part 0: records the machine, the code version and the engine binaries the evidence comes from."""
from __future__ import annotations

import datetime as dt
import os
import platform
import subprocess

from common import REPO, RESULTS, Section, rel, sha256, write_text
from engines import ENGINES


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def cpu_name() -> str:
    if platform.system() == "Windows":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def os_name() -> str:
    if platform.system() == "Windows":
        import sys

        build = sys.getwindowsversion().build  # Windows 11 still reports release "10"; the build tells
        return f"Windows {11 if build >= 22000 else 10} (build {build})"
    return f"{platform.system()} {platform.release()}"


def run(fresh: bool = False):
    s = Section("Environment of the ATAM evidence")
    s.add(f"generated: {dt.datetime.now().strftime('%Y-%m-%d')}")
    paths = ("common", "ewma", "garch_t", "realized_caviar", "data")
    dirty = "yes" if _git("status", "--porcelain", "--", *paths) else "no"
    s.add(f"engines and data: commit {_git('log', '-1', '--format=%h %ad', '--date=short', '--', *paths)}"
          f" (uncommitted changes: {dirty})")
    s.add(f"machine: {cpu_name()}, {os.cpu_count()} hardware threads, {os_name()}")
    s.add(f"python: {platform.python_version()}")
    s.add("")
    s.add("engine binaries (the ready-to-run programs versioned in each module's bin/)")
    for e in ENGINES.values():
        if not e.enabled:
            s.add(f"  {e.label}: not evaluated yet ({e.note})")
            continue
        for prog in [p for p in (e.program, e.benchmark) if p]:  # no separate benchmark: listed once
            exe = e.exe(prog)
            s.add(f"  {rel(exe)}  sha256 {sha256(exe)[:16]}  {exe.stat().st_size / 1e6:.2f} MB")
    write_text(RESULTS / "environment.txt", s.text())
    print("   " + "\n   ".join(s.text().splitlines()[1:4]))
    return None
