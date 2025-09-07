#!/usr/bin/env python3
"""
Discover the *actual* installable distributions your project imports at runtime.

Usage (from repo root):
  python tools/minimal_requirements_from_runtime.py --module ai_emtac
  # or explicitly import more modules to exercise code paths:
  python tools/minimal_requirements_from_runtime.py --module ai_emtac --also modules.database_manager.db_manager

What it does:
- Imports the given module(s)
- Collects all loaded top-level packages from sys.modules
- Maps top-level package -> PyPI distribution(s) using importlib.metadata
- Writes requirements.discovered.txt with pinned versions
"""
from __future__ import annotations
import argparse, importlib, importlib.metadata as md, sys, sysconfig, os, traceback
from pathlib import Path

def is_stdlib(module_file: str | None) -> bool:
    if not module_file:
        return True
    libpath = sysconfig.get_paths().get("stdlib") or ""
    platlib = sysconfig.get_paths().get("platstdlib") or ""
    # stdlib modules live under stdlib/platstdlib; site-packages live elsewhere
    return module_file.startswith(libpath) or module_file.startswith(platlib)

def top_level_name(modname: str) -> str:
    return modname.split(".", 1)[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", action="append", required=True,
                    help="Module(s) to import, e.g. ai_emtac")
    ap.add_argument("--also", action="append", default=[],
                    help="Extra modules to import to exercise code paths")
    ap.add_argument("--outfile", default="requirements.discovered.txt")
    args = ap.parse_args()

    # Try to avoid running servers on import if app uses __main__ guards/env checks
    os.environ.setdefault("EMTAC_DISCOVERY_MODE", "1")

    imported = set()
    errors = []

    for name in list(args.module) + list(args.also):
        try:
            importlib.import_module(name)
            print(f"[imported] {name}")
        except Exception as e:
            print(f"[ERROR] while importing {name}: {e}", file=sys.stderr)
            traceback.print_exc()
            errors.append((name, e))

    # Collect top-level package names actually loaded from site-packages
    site_tops = set()
    for modname, mod in list(sys.modules.items()):
        if not mod:
            continue
        try:
            f = getattr(mod, "__file__", None)
            if f and not is_stdlib(f):
                site_tops.add(top_level_name(modname))
        except Exception:
            continue

    # Map top-level import names -> distributions (and versions)
    pkg_map = md.packages_distributions()  # { "PIL": ["Pillow"], "yaml": ["PyYAML"], ... }
    dists = {}
    for top in sorted(site_tops):
        for dist in pkg_map.get(top, []):
            try:
                ver = md.version(dist)
                dists[dist] = ver
            except md.PackageNotFoundError:
                # Not an installed dist in this env (could be namespace or edited)
                pass

    # A few common import->dist fallbacks (in case top-level wasn't in packages_distributions map)
    FALLBACKS = {
        "PIL": "Pillow",
        "cv2": "opencv-python-headless",
        "yaml": "PyYAML",
        "sklearn": "scikit-learn",
        "Crypto": "pycryptodome",
        "OpenSSL": "pyOpenSSL",
        "win32com": "pywin32",
        "pptx": "python-pptx",
        "fitz": "PyMuPDF",
        "bs4": "beautifulsoup4",
        "Levenshtein": "python-Levenshtein",
    }
    for top, dist in FALLBACKS.items():
        if top in site_tops and dist not in dists:
            try:
                dists[dist] = md.version(dist)
            except md.PackageNotFoundError:
                pass

    # Write result
    out = Path(args.outfile)
    with out.open("w", encoding="utf-8") as f:
        for dist in sorted(dists):
            f.write(f"{dist}=={dists[dist]}\n")

    print(f"\nWrote {out} with {len(dists)} discovered distributions.")
    if errors:
        print("\nSome imports failed (these may indicate missing deps or runtime-only branches):")
        for name, e in errors:
            print(f"  - {name}: {e}")

if __name__ == "__main__":
    main()
