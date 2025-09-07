# utility_tools/verify_imports.py
"""
Verify that all packages listed in requirements.full.txt can be imported.

Usage (inside container):
    python utility_tools/verify_imports.py
"""

import re
import importlib

# Path to your "frozen" requirements file
REQ_FILE = "requirements.full.txt"

# Some packages need special import names different from PyPI names
IMPORT_ALIASES = {
    "pillow": "PIL",
    "python-docx": "docx",
    "python-pptx": "pptx",
    "pdfplumber": "pdfplumber",
    "fuzzywuzzy": "fuzzywuzzy",
    "python-levenshtein": "Levenshtein",
    "opencv-python": "cv2",
    "pypdf2": "PyPDF2",
    "scikit-learn": "sklearn",
    "pyyaml": "yaml",
    "torchvision": "torchvision",
    "torchaudio": "torchaudio",
}

def normalize_package(pkg: str) -> str:
    """
    Strip extras and version specifiers from a requirement line
    and map to its likely import name.
    """
    base = re.split(r"[=<>! \[]", pkg, 1)[0].lower().strip()
    return IMPORT_ALIASES.get(base, base)

def main():
    # Read packages from requirements.full.txt
    with open(REQ_FILE, "r", encoding="utf-8") as f:
        pkgs = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

    imports = [normalize_package(p) for p in pkgs]
    failed = []

    for m in sorted(set(imports)):
        try:
            importlib.import_module(m)
        except Exception as e:
            failed.append((m, str(e)))

    if not failed:
        print("✅ All imports succeeded")
    else:
        print(f"❌ {len(failed)} failed imports:")
        for f in failed:
            print("   ", f[0], "->", f[1])

if __name__ == "__main__":
    main()
