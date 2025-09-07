"""
config_env.py
Database configuration for EMTAC project.
- Main DB: Always PostgreSQL (from .env / Docker)
- Revision DB: Stays SQLite (separate schema)
- Provides both context managers and legacy .get_*_session() callables
- Includes connection limiting, logging, and Postgres FTS setup
"""

import os
import logging
from contextlib import contextmanager
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, scoped_session
from dotenv import load_dotenv

# --------------------------------------------------------------------
# Logging setup
# --------------------------------------------------------------------
logger = logging.getLogger("config_env")
logger.setLevel(logging.DEBUG)

# --------------------------------------------------------------------
# Load environment
# --------------------------------------------------------------------
load_dotenv()

POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "emtacdb")

DATABASE_URL = (
    f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# Revision DB stays SQLite
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REVISION_DB_PATH = os.path.join(BASE_DIR, "revision_control.db")
REVISION_DATABASE_URL = f"sqlite:///{REVISION_DB_PATH}"

# --------------------------------------------------------------------
# Connection limiting event (prevents accidental overload)
# --------------------------------------------------------------------
def add_connection_limit(engine: Engine, max_conns: int = 10):
    @event.listens_for(engine, "connect")
    def connect_event(dbapi_connection, connection_record):
        active = engine.pool.checkedout()
        if active > max_conns:
            raise RuntimeError(
                f"Too many active connections: {active}/{max_conns}"
            )

# --------------------------------------------------------------------
# DatabaseConfig
# --------------------------------------------------------------------
class DatabaseConfig:
    def __init__(self, enable_connection_limit: bool = True, max_connections: int = 10):
        # --- Main Postgres engine ---
        self.engine = create_engine(
            DATABASE_URL,
            echo=False,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            client_encoding="utf8",
        )

        if enable_connection_limit:
            add_connection_limit(self.engine, max_connections)

        # scoped_session ensures thread safety
        self.MainSessionMaker = scoped_session(sessionmaker(bind=self.engine))

        # --- Revision SQLite engine ---
        self.revision_control_engine = create_engine(
            REVISION_DATABASE_URL,
            echo=False,
            connect_args={"check_same_thread": False},
        )
        self.RevisionSessionMaker = scoped_session(
            sessionmaker(bind=self.revision_control_engine)
        )

        logger.info(
            f"DatabaseConfig initialized with PostgreSQL (main) "
            f"and SQLite (revision), connection limiting={enable_connection_limit}, "
            f"max_connections={max_connections}"
        )

        # Ensure Postgres FTS extensions exist
        self._init_postgres_extensions()

    # --- Context managers ---
    @contextmanager
    def main_session(self):
        session = self.MainSessionMaker()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @contextmanager
    def revision_session(self):
        session = self.RevisionSessionMaker()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # --- Legacy aliases (callables) ---
    def get_main_session(self):
        """Legacy alias for code expecting a callable session factory"""
        return self.MainSessionMaker()

    def get_revision_control_session(self):
        """Legacy alias for code expecting a callable revision session factory"""
        return self.RevisionSessionMaker()

    # --- Extra legacy aliases (old registry calls) ---
    def get_main_session_registry(self):
        """Legacy alias used in older modules"""
        return self.MainSessionMaker

    def get_revision_control_session_registry(self):
        """Legacy alias used in older modules"""
        return self.RevisionSessionMaker

    # --- Postgres FTS setup ---
    def _init_postgres_extensions(self):
        try:
            with self.engine.begin() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS unaccent"))
            logger.info("Postgres FTS extensions ensured (pg_trgm, unaccent).")
        except Exception as e:
            logger.warning(f"Could not init Postgres extensions: {e}")

    # --- Fallback guard ---
    def __getattr__(self, item):
        if item in (
            "get_main_session",
            "get_revision_control_session",
            "get_main_session_registry",
            "get_revision_control_session_registry",
        ):
            logger.warning(f"[DatabaseConfig] {item} accessed via getattr (legacy).")
            return getattr(self, item)
        logger.warning(
            f"[DatabaseConfig] Attempted access of missing attribute '{item}'. Returning None."
        )
        return None



# Instantiate global db_config
db_config = DatabaseConfig()
