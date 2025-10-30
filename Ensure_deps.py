#!/usr/bin/env python
import sys
import os
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEPS_DIR = os.path.join(BASE_DIR, "deps")
REQUIRED = ["numpy", "trimesh", "scipy"]  # scipy optional but included

def ensure_deps_on_path():
    # Ensure local deps directory is visible to this process
    if os.path.isdir(DEPS_DIR) and DEPS_DIR not in sys.path:
        sys.path.insert(0, DEPS_DIR)

def check_import(pkg):
    try:
        __import__(pkg)
        return True
    except Exception:
        return False

def install(pkg):
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "--target", DEPS_DIR, pkg]
    print("Installing:", " ".join(cmd))
    return subprocess.call(cmd) == 0

def main(auto_yes=False):
    ensure_deps_on_path()
    missing = [p for p in REQUIRED if not check_import(p)]
    if not missing:
        print("All required packages available (local or global). Skipping install.")
        return 0

    print("Missing packages:", ", ".join(missing))
    if not os.path.isdir(DEPS_DIR):
        os.makedirs(DEPS_DIR, exist_ok=True)
        # make sure newly created folder is visible immediately
        ensure_deps_on_path()

    if not auto_yes:
        resp = input("Install missing packages into ./deps ? [y/N] ").strip().lower()
        if resp != "y":
            print("Installation aborted by user. Missing packages remain:", ", ".join(missing))
            return 2

    for p in missing:
        ok = install(p)
        if not ok:
            print(f"Failed to install {p}. Check pip output and install manually into ./deps.")
            return 3

    # After install, ensure deps are on path and verify
    ensure_deps_on_path()
    still_missing = [p for p in REQUIRED if not check_import(p)]
    if still_missing:
        print("Some packages still missing after install:", ", ".join(still_missing))
        return 3

    print("Installed missing packages into ./deps successfully.")
    return 0

if __name__ == "__main__":
    auto = "--yes" in sys.argv or "--auto" in sys.argv
    rc = main(auto)
    sys.exit(rc)