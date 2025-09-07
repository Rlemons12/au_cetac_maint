import importlib, sys

mods = [
    # web stack
    "flask","werkzeug","jinja2","itsdangerous",
    # db
    "sqlalchemy","psycopg2","pgvector",
    # data/utils
    "numpy","pandas","requests","redis","dotenv",
    # docs/pdf
    "docx","pptx","fitz","pdfplumber","pdf2image","pypdfium2","PIL","openpyxl",
    # ml
    "spacy",
]

failed = []
for m in mods:
    try:
        importlib.import_module(m)
        print(f"OK   {m}")
    except Exception as e:
        print(f"FAIL {m}: {e}")
        failed.append((m, str(e)))

print("\nSummary:")
if failed:
    print("Some imports failed:")
    for m, err in failed:
        print(f" - {m}: {err}")
    sys.exit(1)
else:
    print("All imports succeeded.")
