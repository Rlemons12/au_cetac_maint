# init_db.py
from __future__ import annotations

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from sqlalchemy import text, select
from sqlalchemy.orm import Session

# Engine / logging from your existing config
from modules.configuration.config_env import DatabaseConfig

from modules.configuration.log_config import logger
from modules.configuration.config import ORC_TRAINING_DATA_LOADSHEET
from modules.emtac_ai.models.emtac_ai_db_models import (
    Base,
    DatasetSource,
    Intent,
    QueryTemplate,
    LabelSet,
    Placeholder,
    PlaceholderColumnMap,
    NLVariation,
)
from pathlib import Path

def _to_relative_dataset_path(path: str | Path) -> str:
    """
    Convert absolute dataset path to a repo-relative string.
    """
    path = Path(path).resolve()
    try:
        # Find repo root (adjust if needed)
        repo_root = Path(__file__).resolve().parents[3]  # up from modules/emtac_ai/models/init_db.py
        return str(path.relative_to(repo_root))
    except ValueError:
        # Fallback: return basename only if outside repo
        return path.name
import os
from pathlib import Path

def _repo_root() -> Path:
    """
    Best-effort repo root:
    - If ORC_REPO_ROOT is set, use it.
    - Else, walk up from this file to find 'modules' and take its parent.
    - Else, fall back to 3 parents up from this file.
    """
    env = os.getenv("ORC_REPO_ROOT")
    if env:
        return Path(env).resolve()

    here = Path(__file__).resolve()
    for p in here.parents:
        if p.name == "modules":
            return p.parent.resolve()

    # Fallback (adjust if your layout is different)
    return here.parents[3].resolve()

def _to_repo_relative(path_like: str | Path) -> str:
    """
    Convert a path to a repo-relative POSIX-style string.
    If it's already relative, normalize to POSIX.
    If it's absolute but outside the repo, fall back to basename.
    """
    p = Path(path_like)
    if not p.is_absolute():
        return p.as_posix()

    try:
        rel = p.resolve().relative_to(_repo_root())
        return rel.as_posix()
    except ValueError:
        # Outside repo: store just the filename to avoid leaking absolute host paths
        return p.name

def _abs_from_repo_relative(rel_or_abs: str | Path) -> Path:
    """
    If the given path is absolute, return it.
    If it's relative, resolve it against repo root.
    """
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p.resolve()
    return (_repo_root() / p).resolve()

# ---------------- Extensions / Tables / Indexes ---------------- #

def ensure_extensions(engine, is_postgres: bool):
    if not is_postgres:
        logger.info("SQLite detected — skipping Postgres extensions.")
        return
    with engine.begin() as conn:
        logger.info("Ensuring PostgreSQL extensions…")
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        logger.info("Extensions ready: pg_trgm, vector")

def create_tables_and_indexes(engine):
    logger.info("Creating tables & indexes from SQLAlchemy models…")
    Base.metadata.create_all(engine)
    logger.info("Tables & indexes created (via __table_args__).")

def analyze_vector_tables(engine, is_postgres: bool):
    if not is_postgres:
        return
    with engine.begin() as conn:
        logger.info("ANALYZE user_query, query_template for IVFFlat stats…")
        for tbl in ("user_query", "query_template"):
            try:
                conn.execute(text(f"ANALYZE {tbl}"))
                logger.info(f"ANALYZE {tbl} done.")
            except Exception as e:
                logger.warning(f"ANALYZE {tbl}: {e}")

def verify_setup(engine, is_postgres: bool):
    with engine.begin() as conn:
        if is_postgres:
            ext_rows = conn.execute(text("""
                SELECT extname FROM pg_extension
                WHERE extname IN ('pg_trgm','vector')
                ORDER BY extname
            """)).fetchall()
            logger.info(f"Loaded extensions: {[r[0] for r in ext_rows]}")

            idx_rows = conn.execute(text("""
                SELECT schemaname, tablename, indexname
                FROM pg_indexes
                WHERE tablename IN ('user_query','query_template','prediction_log')
                ORDER BY schemaname, tablename, indexname
            """)).fetchall()
            for sch, tbl, idx in idx_rows:
                logger.info(f"Index present: {sch}.{tbl} -> {idx}")

        # quick existence checks
        for tbl in ("user_query", "query_template"):
            try:
                conn.execute(text(f"SELECT 1 FROM {tbl} LIMIT 1"))
                logger.info(f"{tbl} table OK")
            except Exception as e:
                logger.warning(f"{tbl} not accessible yet: {e}")

# ---------------- LabelSet seeders (parts/drawings/intent) ---------------- #

def seed_labelset_parts_ner(session: Session) -> LabelSet:
    name = "parts_bio"
    existing = session.execute(
        select(LabelSet).where(LabelSet.task == "ner", LabelSet.name == name)
    ).scalar_one_or_none()
    if existing:
        if not existing.is_active:
            existing.is_active = True
            session.commit()
        return existing

    entities = ["PART_NUMBER", "PART_NAME", "MANUFACTURER", "MODEL"]
    labels = ["O"] + [f"{p}-{e}" for e in entities for p in ("B", "I")]
    id2label = {i: lab for i, lab in enumerate(labels)}
    label2id = {lab: i for i, lab in enumerate(labels)}

    ls = LabelSet(
        task="ner", name=name, scheme="BIO",
        entities=entities, labels=labels,
        id2label=id2label, label2id=label2id,
        is_active=True,
    )
    session.add(ls); session.commit()
    return ls

def seed_labelset_drawings_ner(session: Session) -> LabelSet:
    name = "drawings_bio"
    existing = session.execute(
        select(LabelSet).where(LabelSet.task == "ner", LabelSet.name == name)
    ).scalar_one_or_none()
    if existing:
        if not existing.is_active:
            existing.is_active = True
            session.commit()
        return existing

    entities = ["EQUIPMENT_NUMBER"]
    labels = ["O"] + [f"{p}-{e}" for e in entities for p in ("B", "I")]
    id2label = {i: lab for i, lab in enumerate(labels)}
    label2id = {lab: i for i, lab in enumerate(labels)}

    ls = LabelSet(
        task="ner", name=name, scheme="BIO",
        entities=entities, labels=labels,
        id2label=id2label, label2id=label2id,
        is_active=True,
    )
    session.add(ls); session.commit()
    return ls

def seed_labelset_intents_from_dirs(session: Session, intent_names: List[str], name: str = "intents_from_templates") -> LabelSet:
    intent_names = sorted({n for n in intent_names if n})
    existing = session.execute(
        select(LabelSet).where(LabelSet.task == "intent", LabelSet.name == name)
    ).scalar_one_or_none()

    labels = intent_names
    id2label = {i: lab for i, lab in enumerate(labels)}
    label2id = {lab: i for i, lab in enumerate(labels)}

    if existing:
        existing.labels = labels
        existing.id2label = id2label
        existing.label2id = label2id
        existing.scheme = "NONE"
        existing.is_active = True
        session.commit()
        return existing

    ls = LabelSet(
        task="intent", name=name, scheme="NONE",
        entities=None, labels=labels,
        id2label=id2label, label2id=label2id,
        is_active=True,
    )
    session.add(ls); session.commit()
    return ls

# ---------------- Template ingestion (query_templates/<intent>/*.{txt,jsonl}) ---------------- #

def discover_templates_root(cli_override: str | None = None) -> Path:
    if cli_override:
        return Path(cli_override).expanduser().resolve()
    env_p = os.getenv("ORC_QUERY_TEMPLATES_DIR")
    if env_p:
        return Path(env_p).expanduser().resolve()
    # fallback to repo path
    here = Path(__file__).resolve()
    for up in [here] + list(here.parents):
        candidate = up.parent / "modules" / "emtac_ai" / "training_data" / "query_templates"
        if candidate.exists():
            return candidate
    return Path("modules/emtac_ai/training_data/query_templates").resolve()

# Robust reader: handles plain TXT, Python-like lists (VAR=[...]),
# quotes/commas, and JSONL {"template": "..."} or {"text": "..."}
_ASSIGN_RE        = re.compile(r"^\s*[A-Za-z0-9_]+\s*=\s*\[\s*$")
_BRACKET_OPEN_RE  = re.compile(r"\[\s*$")
_BRACKET_CLOSE_RE = re.compile(r"^\s*\]\s*,?\s*$")

def _read_templates_from_file(file_path: Path) -> List[str]:
    out: List[str] = []
    suffix = file_path.suffix.lower()

    # JSONL support
    if suffix == ".jsonl":
        with file_path.open("r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except Exception:
                    continue
                val = obj.get("template") or obj.get("text")
                if isinstance(val, str):
                    v = val.strip()
                    if v:
                        out.append(v)
        return out

    # TXT: robust handling (plain lines or Python-like list)
    in_list = False
    with file_path.open("r", encoding="utf-8") as f:
        for raw in f:
            ln = raw.strip()
            if not ln or ln.startswith("#"):
                continue

            # entering a list after an assignment
            if _ASSIGN_RE.match(ln) or (not in_list and _BRACKET_OPEN_RE.search(ln) and "=" in ln):
                in_list = True
                continue

            # plain open bracket line
            if not in_list and _BRACKET_OPEN_RE.search(ln):
                in_list = True
                continue

            # closing bracket line
            if in_list and _BRACKET_CLOSE_RE.match(ln):
                in_list = False
                continue

            # normalize candidate line
            cand = ln
            # strip trailing comma
            if cand.endswith(","):
                cand = cand[:-1].rstrip()

            # strip surrounding quotes ("..." or '...')
            if (cand.startswith('"') and cand.endswith('"')) or (cand.startswith("'") and cand.endswith("'")):
                cand = cand[1:-1].strip()

            # skip empty/garbage
            if not cand or _ASSIGN_RE.match(cand) or _BRACKET_OPEN_RE.match(cand) or _BRACKET_CLOSE_RE.match(cand):
                continue

            out.append(cand)

    return out

def ingest_query_templates(session: Session, root: Path) -> Tuple[int, Dict[str, int]]:
    if not root.exists():
        logger.warning(f"Templates dir not found: {root}")
        return 0, {}

    total = 0
    per_intent: Dict[str, int] = {}

    for sub in sorted([p for p in root.iterdir() if p.is_dir()]):
        intent_name = sub.name.strip().lower()
        if not intent_name:
            continue

        # Ensure intent row
        intent = session.execute(
            select(Intent).where(Intent.name == intent_name)
        ).scalar_one_or_none()
        if not intent:
            intent = Intent(name=intent_name, is_active=True)
            session.add(intent); session.flush()

        # Gather existing templates to avoid dup inserts
        existing = {
            t.template_text for t in session.query(QueryTemplate.template_text)
            .filter(QueryTemplate.intent_id == intent.id)
            .all()
        }

        # Read all .txt and .jsonl files in folder
        new_count = 0
        for fp in sorted(list(sub.glob("*.txt")) + list(sub.glob("*.jsonl"))):
            for tpl in _read_templates_from_file(fp):
                if tpl in existing:
                    continue
                session.add(QueryTemplate(
                    intent_id=intent.id,
                    template_text=tpl,
                    lang="en",
                    is_active=True,
                ))
                existing.add(tpl)
                new_count += 1

        if new_count:
            session.commit()
            total += new_count
            per_intent[intent_name] = new_count
            logger.info(f"Ingested {new_count} template(s) for intent '{intent_name}' from {sub}")

    if not total:
        logger.info(f"No new templates found under {root}")
    return total, per_intent

# ---- Cleanup previously ingested rows (headers/brackets/quotes/commas) ---- #

def _normalize_template_text(s: str) -> str:
    t = s.strip()
    if t.endswith(","):
        t = t[:-1].rstrip()
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        t = t[1:-1].strip()
    return t

def cleanup_ingested_templates(session: Session) -> Tuple[int, int]:
    """
    Returns (deleted, updated).
    Deleted: headers/brackets like 'VAR = [' or '[' or ']'
    Updated: rows that needed quote/comma normalization.
    """
    deleted = updated = 0
    bad_patterns = [
        r'^\s*\[$', r'^\s*\]\s*,?\s*$', r'^\s*[A-Za-z0-9_]+\s*=\s*\[\s*$'
    ]
    bad_re = re.compile("|".join(bad_patterns))

    rows = session.query(QueryTemplate).all()
    for r in rows:
        if bad_re.match((r.template_text or "")):
            session.delete(r)
            deleted += 1
    if deleted:
        session.commit()

    rows = session.query(QueryTemplate).all()
    for r in rows:
        old = r.template_text or ""
        new = _normalize_template_text(old)
        if new and new != old:
            r.template_text = new
            updated += 1
    if updated:
        session.commit()

    return deleted, updated

# ---------------- Optional: seed parts PCM (column fallbacks) ---------------- #

def optional_seed_parts_pcm(engine):
    """
    Seed DatasetSource + PlaceholderColumnMap for the parts loadsheet,
    using ENV overrides when present, otherwise your repo config defaults.
    Stores repo-relative path in DB, validates absolute path locally.
    """
    from modules.configuration.config import ORC_TRAINING_DATA_PARTS_LOADSHEET_PATH

    configured_path = os.getenv("ORC_PARTS_LOADSHEET_PATH", ORC_TRAINING_DATA_PARTS_LOADSHEET_PATH)
    dataset_rel_path = _to_repo_relative(configured_path)          # store relative
    dataset_abs_path = _abs_from_repo_relative(dataset_rel_path)   # check existence / local IO
    sheet_name       = os.getenv("ORC_PARTS_LOADSHEET_SHEET", "Parts")

    if not dataset_abs_path.exists():
        logger.warning(
            "Parts loadsheet not found at '%s' (abs). "
            "Set ORC_PARTS_LOADSHEET_PATH or verify config.ORC_TRAINING_DATA_LOADSHEET.",
            str(dataset_abs_path)
        )

    from modules.emtac_ai.models.emtac_ai_db_models import DatasetSource

    with Session(engine) as s:
        # If your seed_parts_pcm implementation expects a path it only stores (no IO),
        # passing the repo-relative string is correct; it will be what persists in DB.
        # If it opens the file, consider updating that function to resolve to abs before IO.
        DatasetSource.seed_parts_pcm(
            s,
            dataset_name="parts_loadsheet",
            dataset_path=dataset_rel_path,   # <= repo-relative path goes into DB
            sheet_name=sheet_name,
        )
        logger.info(
            "Seeded DatasetSource + PlaceholderColumnMap for parts: "
            "path(rel)='%s', path(abs)='%s', sheet='%s'",
            dataset_rel_path, str(dataset_abs_path), sheet_name
        )

def optional_seed_drawings_pcm(engine):
    """
    Seed DatasetSource + PlaceholderColumnMap for the *Active Drawing List*.
    ENV overrides:
      - ORC_DRAWINGS_LOADSHEET_PATH
      - ORC_DRAWINGS_LOADSHEET_SHEET
      - ORC_DRAWINGS_ACTIVE_STATUS  (comma-separated; stored in DatasetSource.extra)
    """
    from modules.configuration.config import ORC_TRAINING_DATA_DRAWINGS_LOADSHEET_PATH

    # Read configured/ENV path (may be absolute or relative)
    configured_path = os.getenv("ORC_DRAWINGS_LOADSHEET_PATH", ORC_TRAINING_DATA_DRAWINGS_LOADSHEET_PATH)

    # Store RELATIVE path in DB
    dataset_rel_path = _to_repo_relative(configured_path)
    # Use ABSOLUTE path for existence checks / any local file IO
    dataset_abs_path = _abs_from_repo_relative(dataset_rel_path)

    sheet_name   = os.getenv("ORC_DRAWINGS_LOADSHEET_SHEET", "ActiveDrawings")
    active_vals  = os.getenv("ORC_DRAWINGS_ACTIVE_STATUS", "Active,Current,Approved")
    active_list  = [v.strip() for v in active_vals.split(",") if v.strip()]

    if not dataset_abs_path.exists():
        logger.warning(
            "Active drawing loadsheet not found at '%s' (abs). "
            "Set ORC_DRAWINGS_LOADSHEET_PATH or verify config.ORC_TRAINING_DATA_LOADSHEET.",
            str(dataset_abs_path)
        )

    from sqlalchemy.orm import Session
    from modules.emtac_ai.models.emtac_ai_db_models import DatasetSource

    with Session(engine) as s:
        # Create/ensure DatasetSource (DB stores RELATIVE path)
        ds = DatasetSource.get_or_create(
            s,
            name="drawings_loadsheet",
            path=dataset_rel_path,        # <= repo-relative string
            file_type="xlsx",
            extra={"sheet_name": sheet_name, "active_status_values": active_list},
        )

        # Create/ensure placeholders
        ph_eq   = DatasetSource._get_or_create_placeholder(s, "EQUIPMENT_NUMBER", "EQUIPMENT_NUMBER", "Equipment/asset number")
        ph_dnum = DatasetSource._get_or_create_placeholder(s, "DRAWING_NUMBER",   "DRAWING_NUMBER",   "Drawing number/id")
        ph_rev  = DatasetSource._get_or_create_placeholder(s, "REVISION",         "REVISION",         "Revision id")
        ph_stat = DatasetSource._get_or_create_placeholder(s, "STATUS",           "STATUS",           "Lifecycle status")
        ph_titl = DatasetSource._get_or_create_placeholder(s, "TITLE",            "TITLE",            "Drawing title/name")

        # Prioritized header candidates per placeholder (tune to your sheet headers)
        cand_map = {
            ph_eq.id  : ["EQUIPMENT_NUMBER", "EQUIP_NO", "ASSET", "ASSET_NO", "EQUIPID"],
            ph_dnum.id: ["DRAWING_NUMBER", "DWG_NO", "DWG", "DRAWING", "FILE_NO"],
            ph_rev.id : ["REVISION", "REV", "REV_NO"],
            ph_stat.id: ["STATUS", "STATE", "LIFECYCLE_STATUS"],
            ph_titl.id: ["TITLE", "DRAWING_TITLE", "NAME", "DESCRIPTION"],
        }

        for ph_id, cols in cand_map.items():
            for i, col in enumerate(cols, start=1):
                DatasetSource._ensure_pcm(s, ds.id, ph_id, i, col)

        s.commit()
        logger.info(
            "Seeded DatasetSource + PlaceholderColumnMap for drawings: "
            "path(rel)='%s', path(abs)='%s', sheet='%s', active=%s",
            dataset_rel_path, str(dataset_abs_path), sheet_name, active_list
        )


# ---------------- DEBUG DUMP (tables, counts, samples) ---------------- #

def debug_dump_schema(engine):
    with engine.begin() as conn:
        logger.info("==== TABLES (information_schema.tables) ====")
        rows = conn.execute(text("""
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema NOT IN ('pg_catalog','information_schema')
            ORDER BY table_schema, table_name
        """)).fetchall()
        for sch, tbl in rows:
            logger.info(f"{sch}.{tbl}")

def debug_dump_row_counts(engine):
    with engine.begin() as conn:
        logger.info("==== ROW COUNTS ====")
        tables = [
            "intent", "query_template", "placeholder", "template_placeholder",
            "dataset_source", "placeholder_column_map",
            "nl_variation",
            "label_set",
            "user_query", "prediction_log",
            "intent_annotation", "entity_annotation",
        ]
        for t in tables:
            try:
                n = conn.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
                logger.info(f"{t:26s} : {n}")
            except Exception as e:
                logger.info(f"{t:26s} : (not found) {e}")

def debug_dump_intents_and_templates(session: Session, max_per_intent: int = 5):
    logger.info("==== INTENTS & TEMPLATES (sample) ====")
    intents = session.query(Intent).order_by(Intent.name.asc()).all()
    for it in intents:
        q = (session.query(QueryTemplate.template_text)
                    .filter(QueryTemplate.intent_id == it.id)
                    .order_by(QueryTemplate.id.asc()))
        count = q.count()
        sample = [r[0] for r in q.limit(max_per_intent).all()]
        logger.info(f"intent='{it.name}' active={it.is_active} templates={count}")
        for s in sample:
            logger.info(f"  - {s}")

def debug_dump_labelsets(session: Session):
    logger.info("==== LABEL SETS ====")
    rows = session.query(LabelSet).order_by(LabelSet.task.asc(), LabelSet.name.asc()).all()
    for ls in rows:
        preview = ", ".join(ls.labels[:8]) + (" ..." if len(ls.labels) > 8 else "")
        logger.info(f"{ls.task}:{ls.name} active={ls.is_active} size={len(ls.labels)} scheme={ls.scheme} labels=[{preview}]")

def debug_dump_datasets(session: Session):
    logger.info("==== DATASET SOURCES & COLUMN MAPS ====")
    ds_rows = session.query(DatasetSource).order_by(DatasetSource.name.asc()).all()
    for ds in ds_rows:
        sheet = (ds.extra or {}).get("sheet_name") if isinstance(ds.extra, dict) else None
        logger.info(f"dataset='{ds.name}' path='{ds.path}' type={ds.file_type} sheet={sheet}")
        maps = (session.query(PlaceholderColumnMap, Placeholder)
                      .join(Placeholder, Placeholder.id == PlaceholderColumnMap.placeholder_id)
                      .filter(PlaceholderColumnMap.dataset_id == ds.id)
                      .order_by(PlaceholderColumnMap.placeholder_id.asc(), PlaceholderColumnMap.priority.asc())
                      .all())
        last_tok = None
        line = []
        for pcm, ph in maps:
            tok = (ph.token or "").upper()
            if tok != last_tok and line:
                logger.info("   " + " | ".join(line)); line = []
            prefix = f"{tok}:" if tok != last_tok else "   "
            line.append(f"{prefix} {pcm.column_name} (p{pcm.priority})")
            last_tok = tok
        if line:
            logger.info("   " + " | ".join(line))

def debug_dump_variations(session: Session):
    logger.info("==== NL VARIATIONS (counts) ====")
    rows = (session.query(NLVariation.entity_key, NLVariation.bucket, text("COUNT(*)"))
                  .filter(NLVariation.is_active.is_(True))
                  .group_by(NLVariation.entity_key, NLVariation.bucket)
                  .order_by(NLVariation.entity_key.asc(), NLVariation.bucket.asc())
                  .all())
    if not rows:
        logger.info("(no active variations)")
    for entity_key, bucket, cnt in rows:
        logger.info(f"{entity_key:12s} {bucket:12s} : {cnt}")

def debug_dump_everything(engine):
    debug_dump_schema(engine)
    debug_dump_row_counts(engine)
    with Session(engine) as s:
        debug_dump_intents_and_templates(s, max_per_intent=5)
        debug_dump_labelsets(s)
        debug_dump_datasets(s)
        debug_dump_variations(s)

# ---------------- Orchestration ---------------- #

def main(templates_dir_override: str | None = None):
    db = DatabaseConfig()
    engine = db.get_engine()
    is_pg = db.is_postgresql

    ensure_extensions(engine, is_pg)
    create_tables_and_indexes(engine)
    analyze_vector_tables(engine, is_pg)
    verify_setup(engine, is_pg)

    # Ingest templates and seed label sets
    root = discover_templates_root(templates_dir_override)
    logger.info(f"Using query templates root: {root}")
    with Session(engine) as s:
        total, per_intent = ingest_query_templates(s, root)
        # NEW: cleanup pass for previously-ingested headers/brackets/quotes/commas
        deleted, updated = cleanup_ingested_templates(s)
        logger.info(f"Template ingestion cleanup: deleted={deleted}, normalized={updated}")

        logger.info(f"Template ingestion complete. Total new: {total}, per intent: {per_intent}")

        seed_labelset_parts_ner(s)
        seed_labelset_drawings_ner(s)

        intent_names = list(per_intent.keys()) or [row.name for row in s.query(Intent).all()]
        seed_labelset_intents_from_dirs(s, intent_names)

    # Optional: seed parts mapping (column fallbacks)
    optional_seed_drawings_pcm(engine)
    optional_seed_parts_pcm(engine)
    with Session(engine) as s:
        logger.info("Verifying 'parts_loadsheet' dataset preview…")
        ds = s.query(DatasetSource).filter(DatasetSource.name == "parts_loadsheet").first()
        if ds:
            logger.info(f"parts_loadsheet: path='{ds.path}', type='{ds.file_type}', extra={ds.extra}")

    # ---- DEBUG SNAPSHOT ----
    debug_dump_everything(engine)

if __name__ == "__main__":
    main()
