#!/usr/bin/env python3
import re
from datetime import datetime
from pathlib import Path

# Import the path from config.py
from modules.configuration.config import ORC_QUERY_TEMPLATE_PARTS

# Target file using config setting
TARGET = Path(ORC_QUERY_TEMPLATE_PARTS) / "PARTS_ENHANCED_QUERY_TEMPLATES.txt"

# Patterns to replace
REPLACEMENTS = [
    (re.compile(r"\{itemnum_(?:formal|casual|contextual)\}", re.I), "{itemnum}"),
    (re.compile(r"\{description_(?:formal|casual|contextual)\}", re.I), "{description}"),
    (re.compile(r"\{manufacturer_(?:formal|casual|contextual)\}", re.I), "{manufacturer}"),
    (re.compile(r"\{model_(?:formal|casual|contextual)\}", re.I), "{model}"),
    (re.compile(r"\{item_number_formal\}", re.I), "{itemnum}"),
    (re.compile(r"\{long_description\}", re.I), "{description}"),
    (re.compile(r"\{oem_mfg\}", re.I), "{manufacturer}"),
    (re.compile(r"\{mfg\}", re.I), "{manufacturer}"),
    (re.compile(r"\{mpn\}", re.I), "{model}"),
]

# Lines to remove entirely
STRAY_LINE_PATTERNS = [
    re.compile(r"DRAWINGS_ENHANCED_QUERY_TEMPLATES"),
    re.compile(r'^\s*print\s*\(', re.I),
]

PLACEHOLDER_FINDER = re.compile(r"\{([a-zA-Z0-9_]+)\}")

def clean_text(text: str):
    lines = text.splitlines()
    kept = []
    removed = 0
    for line in lines:
        if any(p.search(line) for p in STRAY_LINE_PATTERNS):
            removed += 1
            continue
        for pat, repl in REPLACEMENTS:
            line = pat.sub(repl, line)
        kept.append(line)
    cleaned = "\n".join(kept) + "\n"
    return cleaned, removed

def find_variants(text: str):
    bad = set()
    for ph in PLACEHOLDER_FINDER.findall(text):
        if ph.lower().endswith(("_formal", "_casual", "_contextual")):
            bad.add(ph)
    return sorted(bad)

def main():
    if not TARGET.exists():
        raise SystemExit(f"[ERR] File not found: {TARGET}")

    orig = TARGET.read_text(encoding="utf-8")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = TARGET.with_suffix(TARGET.suffix + f".bak_{ts}")
    backup.write_text(orig, encoding="utf-8")

    cleaned, removed = clean_text(orig)
    leftovers = find_variants(cleaned)
    TARGET.write_text(cleaned, encoding="utf-8")

    print(f"[OK] Cleaned: {TARGET}")
    print(f"[OK] Backup: {backup}")
    print(f"[OK] Removed lines: {removed}")
    if leftovers:
        print("[WARN] Still found variant placeholders:")
        for k in leftovers:
            print(f"  - {k}")
    else:
        print("[OK] No _formal/_casual/_contextual placeholders remain.")

if __name__ == "__main__":
    main()
