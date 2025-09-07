# scan_imports.py
from __future__ import annotations
import argparse
import ast
import os
import sys
from pathlib import Path

# --- maps "import name" -> pip package name
NAME_MAP = {
    # common renames
    "fitz": "PyMuPDF",
    "pptx": "python-pptx",
    "PIL": "pillow",
    "cv2": "opencv-python-headless",
    "yaml": "PyYAML",
    "sklearn": "scikit-learn",
    "pdfminer": "pdfminer.six",
    "sqlalchemy": "SQLAlchemy",
    "flask": "Flask",
    "jinja2": "Jinja2",
    "werkzeug": "Werkzeug",
    "wtforms": "WTForms",
    "wtforms_sqlalchemy": "WTForms-SQLAlchemy",
    "pythoncom": "pywin32",
    "win32com": "pywin32",
    "pillow_heif": "pillow_heif",
    "openpyxl": "openpyxl",
    "pandas": "pandas",
    "numpy": "numpy",
    "reportlab": "reportlab",
    "pdfplumber": "pdfplumber",
    "pdf2image": "pdf2image",
    "requests": "requests",
    "pyodbc": "pyodbc",
    "psycopg2": "psycopg2-binary",
    "pgvector": "pgvector",
    "spacy": "spacy",
    "docx2pdf": "docx2pdf",
    "piexif": "piexif",
    "pyexiv2": "py3exiv2",  # sometimes 'pyexiv2' is provided by 'py3exiv2'
    "gpt4all": "gpt4all",
    "sentence_transformers": "sentence-transformers",
    "transformers": "transformers",
    "tqdm": "tqdm",
    "uvicorn": "uvicorn",
    "redis": "redis",
    "flask_sqlalchemy": "Flask-SQLAlchemy",
    "flask_migrate": "Flask-Migrate",
    "flask_wtf": "Flask-WTF",
    "python_docx": "python-docx",
    "PyPDF2": "PyPDF2",
}

# Very rough stdlib filter (Python 3.11 baseline)
# Anything not in here and not in your tree is treated as 3rd-party.
# (We keep it short on purpose; better to over-include third-party than miss one.)
STDLIB_LIKE = {
    "sys","os","re","json","csv","io","typing","pathlib","subprocess","time","datetime",
    "math","hashlib","logging","argparse","functools","itertools","traceback","tempfile",
    "shutil","glob","base64","email","sqlite3","http","html","urllib","statistics",
    "threading","asyncio","unittest","dataclasses","enum","types","importlib","contextlib",
    "tkinter","queue","zipfile","gzip","bz2","lzma","select","socket","struct","platform",
}

# Windows-only modules to skip when platform!=windows
WINDOWS_ONLY = {
    "pythoncom", "win32com", "pywinauto", "pywin32", "win32ctypes", "win32cred", "wmi", "comtypes",
    "pypiwin32", "pywin32_ctypes", "pywintypes",
}

# Things that look like *your* packages (in-repo) to ignore as pip deps
LOCAL_PACKAGE_HINTS = {"modules", "plugins", "utility_tools", "utilities", "utils", "main_app", "models"}

def discover_py_files(root: Path, exclude: set[str]) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        # prune excluded dirs
        parts = set(rel.parts)
        if parts & exclude:
            continue
        for name in filenames:
            if name.endswith(".py"):
                files.append(Path(dirpath) / name)
    return files

def parse_imports(pyfile: Path) -> set[str]:
    out: set[str] = set()
    try:
        src = pyfile.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(src, filename=str(pyfile))
    except Exception:
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                root = (n.name.split(".")[0]).strip()
                if root:
                    out.add(root)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = (node.module.split(".")[0]).strip()
                if root:
                    out.add(root)
    return out

def is_local_module(name: str, root: Path) -> bool:
    # Treat imports that resolve to files/dirs in tree as local
    candidates = [
        root / (name + ".py"),
        root / name / "__init__.py",
    ]
    return any(p.exists() for p in candidates) or (name in LOCAL_PACKAGE_HINTS)

def to_pip_name(mod: str) -> str:
    return NAME_MAP.get(mod, mod)

def load_existing_requirements(req_path: Path) -> dict[str, str]:
    """
    Return dict {lower_pkg: exact_line} to preserve pins/extras.
    """
    pinned: dict[str, str] = {}
    if not req_path.exists():
        return pinned
    for raw in req_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Split on version specifiers crudely; keep full line to preserve pins
        pkg = (line.split("[")[0]  # drop extras for key
                    .split("==")[0]
                    .split(">=")[0]
                    .split("<=")[0]
                    .split("~=")[0]
                    .split("!=")[0]
                    .split(">")[0]
                    .split("<")[0]).strip()
        if pkg:
            pinned[pkg.lower()] = line
    return pinned

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".", help="Project root to scan")
    p.add_argument("--exclude", default=".git,.venv,__pycache__,build,dist,venv,tests",
                   help="Comma-separated directories to exclude")
    p.add_argument("--platform", choices=["auto","windows","linux","mac"], default="auto",
                   help="Target platform (affects skipping Windows-only modules)")
    p.add_argument("--out", default=None, help="Write results to this requirements file")
    p.add_argument("--merge", default="requirements.txt", help="Merge & preserve pins from this file if it exists")
    p.add_argument("--keep-unpinned", action="store_true", help="Do not pin versions (leave as bare names)")
    args = p.parse_args()

    root = Path(args.root).resolve()
    exclude = set(part.strip() for part in args.exclude.split(",") if part.strip())

    plat = args.platform
    if plat == "auto":
        plat = {"win32":"windows","cygwin":"windows","darwin":"mac"}.get(sys.platform, "linux")

    print(f"[scan] root={root}")
    print(f"[scan] exclude={sorted(exclude)}")
    print(f"[scan] platform={plat}")

    pyfiles = discover_py_files(root, exclude)
    mods: set[str] = set()
    for f in pyfiles:
        mods |= parse_imports(f)

    # Filter out stdlib + local modules
    third_party = set()
    for m in sorted(mods):
        if m in STDLIB_LIKE:
            continue
        if is_local_module(m, root):
            continue
        third_party.add(m)

    # Platform-specific skip
    if plat != "windows":
        third_party -= WINDOWS_ONLY

    # Map to pip names
    pip_names = sorted({to_pip_name(m) for m in third_party})

    print(f"[scan] discovered modules: {sorted(mods)}")
    print(f"[scan] resolved packages: {pip_names}")

    # Merge with existing requirements to preserve pins
    merged_lines: list[str] = []
    existing = load_existing_requirements((root / args.merge) if args.merge else Path())
    seen_lower: set[str] = set()

    # Keep existing (preserve order)
    if existing:
        for line in (root / args.merge).read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                merged_lines.append(line)
                continue
            key = s.split("[")[0].split("==")[0].split(">=")[0].split("<=")[0].split("~=")[0].split("!=")[0].split(">")[0].split("<")[0].strip().lower()
            if key:
                seen_lower.add(key)
                merged_lines.append(line)
            else:
                merged_lines.append(line)

    # Append any newly discovered packages not already present
    for pkg in pip_names:
        key = pkg.lower()
        if key in seen_lower:
            continue
        if args.keep_unpinned:
            merged_lines.append(pkg)
        else:
            # leave unpinned unless you want to auto-pin later (best done via pip-compile)
            merged_lines.append(pkg)

    output = "\n".join(merged_lines or pip_names) + "\n"

    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"[done] wrote {args.out} with {len(output.splitlines())} lines")
    else:
        print(output)

if __name__ == "__main__":
    main()
