#!/usr/bin/env python3
"""
Drop this file into any package directory and run it:
    python make_init.py

It will create (or update) __init__.py in THIS directory,
importing all public functions, classes, and constants from
the .py scripts alongside it.

Private modules (_foo.py) and private names (_bar) are skipped.
"""

import ast
from pathlib import Path

MANAGED_START = "# === AUTO-IMPORTS: BEGIN (managed) ==="
MANAGED_END   = "# === AUTO-IMPORTS: END (managed) ==="

def get_public_symbols(py_file: Path):
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
    except Exception:
        return []
    names = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            names.append(node.name)
        elif isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and not t.id.startswith("_"):
                    names.append(t.id)
    return sorted(names)

def build_imports_block(pkg_dir: Path) -> str:
    lines = [MANAGED_START]
    all_syms = []
    for py in sorted(pkg_dir.glob("*.py")):
        if py.name in {"__init__.py", "make_init.py"}:
            continue
        if py.stem.startswith("_"):
            continue
        symbols = get_public_symbols(py)
        if symbols:
            line = f"from .{py.stem} import {', '.join(symbols)}"
            all_syms.extend(symbols)
        else:
            line = f"from . import {py.stem}"
            all_syms.append(py.stem)
        lines.append(line)
    lines.append("")
    lines.append("__all__ = [")
    for s in all_syms:
        lines.append(f'    "{s}",')
    lines.append("]")
    lines.append(MANAGED_END)
    lines.append("")
    return "\n".join(lines)

def main():
    pkg_dir = Path(__file__).resolve().parent
    init_path = pkg_dir / "__init__.py"
    block = build_imports_block(pkg_dir)

    if init_path.exists():
        orig = init_path.read_text(encoding="utf-8")
        if MANAGED_START in orig and MANAGED_END in orig:
            before, _, tail = orig.partition(MANAGED_START)
            _, _, after = tail.partition(MANAGED_END)
            new = before + block + after
        else:
            new = orig.rstrip() + "\n\n" + block
    else:
        banner = '"""Package initializer (auto-generated)"""\n\n'
        new = banner + block

    init_path.write_text(new, encoding="utf-8")
    print(f"✅ Wrote imports to {init_path}")

if __name__ == "__main__":
    main()
