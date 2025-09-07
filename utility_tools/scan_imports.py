#!/usr/bin/env python3
"""
gen_requirements.py
-------------------
Scan a Python project for imports, map them to PyPI package names, and
merge the result with the existing requirements.txt (preserving pins).

Outputs:
  - requirements.resolved.txt  (safe to use for Docker builds)

Usage:
  python gen_requirements.py
  python gen_requirements.py --root . --out requirements.lock.txt --exclude tests,migrations --platform linux
"""

from __future__ import annotations
import argparse
import ast
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple, List

# ---------- configuration ----------

# Common module -> PyPI package mappings (extend as needed)
MODULE_TO_PACKAGE: Dict[str, str] = {
    # PDF / Office
    "fitz": "PyMuPDF",
    "PyPDF2": "PyPDF2",
    "pdfplumber": "pdfplumber",
    "pdf2image": "pdf2image",
    "pypdfium2": "pypdfium2",
    "python_docx": "python-docx",
    "docx": "python-docx",
    "python_pptx": "python-pptx",
    "pptx": "python-pptx",
    "openpyxl": "openpyxl",
    "xlrd": "xlrd",
    "XlsxWriter": "XlsxWriter",
    "reportlab": "reportlab",
    "docx2pdf": "docx2pdf",

    # Data / DS
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "cv2": "opencv-python-headless",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "plotly": "plotly",

    # NLP / AI
    "transformers": "transformers",
    "sentence_transformers": "sentence-transformers",
    "sentencepiece": "sentencepiece",
    "tiktoken": "tiktoken",
    "torch": "torch",
    "torchaudio": "torchaudio",
    "torchvision": "torchvision",
    "spacy": "spacy",
    "en_core_web_sm": "en-core-web-sm",
    "gpt4all": "gpt4all",
    "openai": "openai",
    "anthropic": "anthropic",

    # Web / App
    "flask": "Flask",
    "flask_sqlalchemy": "Flask-SQLAlchemy",
    "flask_migrate": "Flask-Migrate",
    "flask_wtf": "Flask-WTF",
    "flask_uploads": "Flask-Uploads",
    "flask_testing": "Flask-Testing",
    "werkzeug": "Werkzeug",
    "jinja2": "Jinja2",
    "starlette": "starlette",
    "uvicorn": "uvicorn",
    "redis": "redis",
    "alembic": "alembic",
    "psycopg2": "psycopg2-binary",  # prefer binary for containers
    "pgvector": "pgvector",
    "python-dotenv": "python-dotenv",
    "pydantic": "pydantic",

    # Images / OCR
    "easyocr": "easyocr",
    "PIL": "pillow",
    "Pillow": "pillow",
    "pytesseract": "pytesseract",
    "imageio": "imageio",
    "scikit_image": "scikit-image",

    # Misc
    "regex": "regex",
    "requests": "requests",
    "sqlalchemy": "SQLAlchemy",
    "filetype": "filetype",
    "pydot": "pydot",
    "graphviz": "graphviz",
    "kaleido": "kaleido",

    # Windows-only (we exclude on non-Windows unless explicitly asked)
    "win32com": "pywin32",
    "pythoncom": "pywin32",
    "pywintypes": "pywin32",
    "comtypes": "comtypes",
    "pywin32": "pywin32",
}

# Some modules import under one name but the import statement is different
ALIASES = {
    "skimage": "scikit-image",
}

WINDOWS_ONLY_PACKAGES = {"pywin32", "comtypes", "pypiwin32"}

# ---------- stdlib detection ----------

def stdlib_names() -> Set[str]:
    # Python 3.10+ provides this (not perfect, but good):
    names = set()
    if hasattr(sys, "stdlib_module_names") and sys.stdlib_module_names:
        names.update(sys.stdlib_module_names)  # type: ignore[attr-defined]
    # Add common builtins just in case
    names.update({
        "sys","os","re","json","subprocess","pathlib","typing","itertools","functools",
        "collections","math","random","datetime","time","argparse","logging","tempfile",
        "shutil","unittest","threading","concurrent","asyncio","zipfile","tarfile",
        "html","http","email","urllib","base64","hashlib","inspect","importlib","types",
        "dataclasses","statistics","uuid","enum","pprint","contextlib","traceback",
        "copy","glob","pickle","struct","codecs","csv","configparser","io","tokenize",
    })
    return names

STDLIB = stdlib_names()

# ---------- parsing imports ----------

class ImportCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.modules: Set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            self.modules.add(root)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            root = node.module.split(".")[0]
            # skip relative imports
            if node.level == 0:
                self.modules.add(root)

    def visit_Call(self, node: ast.Call) -> None:
        # Catch importlib.import_module("pkg") patterns
        try:
            if isinstance(node.func, ast.Attribute) and node.func.attr == "import_module":
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "importlib":
                    if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                        root = node.args[0].value.split(".")[0]
                        self.modules.add(root)
        except Exception:
            pass
        self.generic_visit(node)

def iter_py_files(root: Path, exclude: Set[str]) -> Iterable[Path]:
    for p in root.rglob("*.py"):
        parts = set(p.parts)
        if parts & exclude:
            continue
        yield p

def collect_imports(root: Path, exclude: Set[str]) -> Set[str]:
    found: Set[str] = set()
    for py in iter_py_files(root, exclude):
        try:
            src = py.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(src, filename=str(py))
            ic = ImportCollector()
            ic.visit(tree)
            found.update(ic.modules)
        except Exception:
            # Don't break on a single bad file
            continue
    return found

# ---------- requirements merging ----------

REQ_LINE_RE = re.compile(r"^\s*([A-Za-z0-9._\-]+)\s*(==|>=|<=|~=|>|<)?\s*([A-Za-z0-9.*+!_\-]+)?")

def parse_requirements(path: Path) -> Dict[str, str]:
    """Return {package_name_lower: pinned_line} for existing requirements."""
    if not path.exists():
        return {}
    pinned: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = REQ_LINE_RE.match(s)
        if not m:
            continue
        name, op, ver = m.group(1), m.group(2), m.group(3)
        if name:
            key = name.lower()
            pinned[key] = s  # preserve the exact spec (pin or range)
    return pinned

def module_to_package(mod: str) -> str:
    mod_lower = mod.lower()
    if mod in MODULE_TO_PACKAGE:
        return MODULE_TO_PACKAGE[mod]
    if mod_lower in MODULE_TO_PACKAGE:
        return MODULE_TO_PACKAGE[mod_lower]
    if mod in ALIASES:
        return ALIASES[mod]
    if mod_lower in ALIASES:
        return ALIASES[mod_lower]
    # Heuristic: module name often equals the package name
    return mod

def resolve_packages(modules: Set[str], platform: str) -> Set[str]:
    packages: Set[str] = set()
    for mod in modules:
        if not mod or mod in STDLIB:
            continue
        pkg = module_to_package(mod)
        # Skip Windows-only packages on non-Windows unless explicitly requested
        if platform != "windows" and pkg in WINDOWS_ONLY_PACKAGES:
            continue
        packages.add(pkg)
    return packages

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Project root to scan")
    ap.add_argument("--out", default="requirements.resolved.txt", help="Output file")
    ap.add_argument("--exclude", default="venv,.venv,.git,__pycache__,build,dist,migrations",
                    help="Comma-separated folders to exclude anywhere in path")
    ap.add_argument("--platform", choices=["auto","windows","linux","mac"], default="auto",
                    help="Which platform to target for requirements filtering")
    ap.add_argument("--keep-unpinned", action="store_true",
                    help="If set, do NOT add version pins for new packages (just names).")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    exclude = set(x.strip() for x in args.exclude.split(",") if x.strip())

    plat = args.platform
    if plat == "auto":
        if sys.platform.startswith("win"):
            plat = "windows"
        elif sys.platform == "darwin":
            plat = "mac"
        else:
            plat = "linux"

    print(f"[scan] root={root}")
    print(f"[scan] exclude={sorted(exclude)}")
    print(f"[scan] platform={plat}")

    modules = collect_imports(root, exclude)
    print(f"[scan] discovered modules: {sorted(modules)}")

    needed = resolve_packages(modules, plat)
    print(f"[scan] resolved packages: {sorted(needed)}")

    existing = parse_requirements(root / "requirements.txt")
    print(f"[merge] existing pinned entries: {len(existing)}")

    # Merge while preserving existing pins
    final_lines: List[str] = []

    # 1) Add all existing pinned/declared lines first
    seen_pkgs_lower: Set[str] = set()
    for _, line in existing.items():
        final_lines.append(line)
        name = REQ_LINE_RE.match(line)
        if name:
            seen_pkgs_lower.add(name.group(1).lower())

    # 2) Add missing packages (unpinned unless keep-unpinned=False and we want to pin)
    # We won't auto-pin versions here; safer to leave unpinned for new entries.
    for pkg in sorted(needed, key=str.lower):
        if pkg.lower() not in seen_pkgs_lower:
            final_lines.append(pkg)

    # 3) De-duplicate (preserve last occurrence) and sort lightly for readability
    dedup: Dict[str, str] = {}
    for line in final_lines:
        m = REQ_LINE_RE.match(line)
        if not m:
            # Keep unknown lines as-is with a synthetic key
            dedup[line] = line
            continue
        name = m.group(1).lower()
        dedup[name] = line

    final = sorted(dedup.values(), key=str.lower)
    out_path = root / args.out
    out_path.write_text("\n".join(final) + "\n", encoding="utf-8")

    print(f"[done] wrote {out_path} with {len(final)} entries")

if __name__ == "__main__":
    main()
