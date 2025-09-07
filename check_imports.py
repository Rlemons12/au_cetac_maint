import os
import sys
import ast
import importlib.util
from pathlib import Path
import pathspec

# -----------------------
# Locate project root
# -----------------------
PROJECT_ROOT = Path(__file__).resolve().parent
print(f"[IMPORT CHECK] Project root: {PROJECT_ROOT}")

# -----------------------
# Load .dockerignore if present
# -----------------------
dockerignore_path = PROJECT_ROOT / ".dockerignore"
ignore_spec = None
if dockerignore_path.exists():
    with open(dockerignore_path, "r", encoding="utf-8", errors="ignore") as f:
        patterns = f.read().splitlines()
    ignore_spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns)
    print(f"[IMPORT CHECK] Loaded {len(patterns)} ignore patterns from .dockerignore")
else:
    print("[IMPORT CHECK] No .dockerignore found, scanning everything")

# -----------------------
# Recursively scan .py files
# -----------------------
imports = set()
for root, _, files in os.walk(PROJECT_ROOT):
    for file in files:
        if not file.endswith(".py"):
            continue

        path = Path(root) / file
        rel_path = str(path.relative_to(PROJECT_ROOT))

        # Skip if ignored by .dockerignore
        if ignore_spec and ignore_spec.match_file(rel_path):
            continue

        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                tree = ast.parse(f.read(), filename=file)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.add(alias.name.split(".")[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.add(node.module.split(".")[0])
        except SyntaxError:
            print(f"[IMPORT CHECK] Skipped invalid Python syntax in {path}")
        except Exception as e:
            print(f"[IMPORT CHECK] Error parsing {path}: {e}")

if not imports:
    print("[IMPORT CHECK] Found 0 unique modules to check...")
else:
    print(f"[IMPORT CHECK] Found {len(imports)} unique modules to check...")

# -----------------------
# Load requirements.txt
# -----------------------
requirements = set()
req_file = PROJECT_ROOT / "requirements.txt"
if req_file.exists():
    try:
        with open(req_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                pkg = line.strip()
                if pkg and not pkg.startswith("#"):
                    requirements.add(pkg.split("==")[0].lower())
        print(f"[IMPORT CHECK] Loaded {len(requirements)} packages from requirements.txt")
    except Exception as e:
        print(f"[IMPORT CHECK] Could not read requirements.txt: {e}")
else:
    print("[IMPORT CHECK] No requirements.txt found")

# -----------------------
# Check availability
# -----------------------
missing = []
skip_modules = {
    "__future__", "__main__", "__spec__", "__init__",
    "builtins", "site", "this", "antigravity",
    "typing", "dataclasses", "os", "sys",
    "re", "json", "logging", "time", "datetime", "pathlib"
}

for module in sorted(imports):
    if module in skip_modules:
        continue  # always safe / stdlib / pseudo

    try:
        spec = importlib.util.find_spec(module)
        if spec is None:
            status = "NOT INSTALLED"
            if module.lower() in requirements:
                status += " (but in requirements.txt — install issue?)"
            else:
                status += " (and NOT in requirements.txt — add it!)"
            missing.append((module, status))
    except ValueError:
        # Skip invalid pseudo-modules like __spec__
        print(f"[IMPORT CHECK] Skipping invalid module name: {module}")
        continue
    except Exception as e:
        print(f"[IMPORT CHECK] Error checking {module}: {e}")
        continue

# -----------------------
# Report
# -----------------------
if missing:
    print("\n[IMPORT CHECK] Missing modules:")
    for mod, status in missing:
        print(f"  - {mod}: {status}")
else:
    print("\n[IMPORT CHECK] All imports satisfied.")

print("\n[IMPORT CHECK] Completed.\n")

# Always soft exit 0 so Docker build does not fail
sys.exit(0)
