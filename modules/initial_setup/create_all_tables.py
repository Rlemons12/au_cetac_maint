"""
create_all_tables.py
-----------------------------------
Utility script to create all database tables in PostgreSQL.
Ensures pgvector extension is installed before creating tables.
Also prints a summary of created tables.
"""

import logging
import sys
import os
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError

# Import EMTAC modules
try:
    from modules.configuration.config_env import DatabaseConfig, DATABASE_URL
    from modules.emtacdb.emtacdb_fts import Base
except ImportError as e:
    print(f"❌ Could not import EMTAC modules: {e}")
    sys.exit(1)

# Logger setup
logger = logging.getLogger("create_all_tables")
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def ensure_extensions(engine):
    """Ensure all required PostgreSQL extensions are installed"""
    extensions = [
        ("vector", "pgvector extension for embeddings"),
        ("pg_trgm", "trigram matching for full-text search"),
        ("unaccent", "unaccent function for text normalization")
    ]

    try:
        with engine.begin() as conn:
            for ext_name, description in extensions:
                try:
                    conn.execute(text(f"CREATE EXTENSION IF NOT EXISTS {ext_name}"))
                    logger.info(f"✅ Ensured extension: {ext_name} ({description})")
                except Exception as e:
                    logger.error(f"❌ Failed to create extension {ext_name}: {e}")
                    if ext_name == "vector":
                        logger.error("This likely means:")
                        logger.error("  - pgvector is not installed in the PostgreSQL image")
                        logger.error("  - You need to use pgvector/pgvector Docker image")
                        logger.error("  - Or install pgvector manually in your PostgreSQL instance")
                        raise
    except Exception as e:
        logger.error(f"❌ Failed to ensure extensions: {e}")
        raise


def create_all_tables():
    """Create all tables in the main PostgreSQL database."""
    try:
        db_config = DatabaseConfig()

        # Get the database URL from multiple possible sources
        db_url = None

        # Try to get from the imported DATABASE_URL module variable
        if 'DATABASE_URL' in globals() and DATABASE_URL:
            db_url = DATABASE_URL
            logger.info("Using DATABASE_URL from config_env module")

        # Try environment variable as fallback
        if not db_url:
            db_url = os.getenv('DATABASE_URL')
            if db_url:
                logger.info("Using DATABASE_URL from environment variable")

        # Try constructing from individual environment variables
        if not db_url:
            postgres_user = os.getenv('POSTGRES_USER', 'postgres')
            postgres_password = os.getenv('POSTGRES_PASSWORD', 'postgres')
            postgres_host = os.getenv('POSTGRES_HOST', 'localhost')
            postgres_port = os.getenv('POSTGRES_PORT', '5432')
            postgres_db = os.getenv('POSTGRES_DB', 'emtacdb')

            db_url = (
                f"postgresql+psycopg2://{postgres_user}:{postgres_password}"
                f"@{postgres_host}:{postgres_port}/{postgres_db}"
            )
            logger.info("Constructed DATABASE_URL from environment variables")

        if not db_url:
            logger.error("❌ No DATABASE_URL found. Check environment configuration.")
            logger.error("Expected environment variables:")
            logger.error("  - DATABASE_URL (complete URL), OR")
            logger.error("  - POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB")
            return

        logger.info(f"=== Creating all tables in PostgreSQL (main DB) ===")
        # Don't log the full URL as it contains the password
        safe_url = db_url.split('@')[1] if '@' in db_url else db_url
        logger.info(f"Connecting to database at: ...@{safe_url}")

        # Connect and create engine
        engine = create_engine(db_url)

        # Test the connection first
        try:
            with engine.connect() as conn:
                logger.info("✅ Database connection successful")
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            return

        # Ensure required extensions are installed
        logger.info("=== Installing required PostgreSQL extensions ===")
        ensure_extensions(engine)

        # Create all tables
        logger.info("=== Creating database tables ===")
        Base.metadata.create_all(engine)
        logger.info("✅ Tables created successfully")

        # Inspect tables
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        logger.info(f"✅ Found {len(tables)} tables in main DB:")
        for t in sorted(tables):
            # Get table info
            columns = inspector.get_columns(t)
            column_count = len(columns)

            # Check for vector columns
            vector_cols = [col['name'] for col in columns if 'vector' in str(col['type']).lower()]
            vector_info = f" (with vector columns: {', '.join(vector_cols)})" if vector_cols else ""

            logger.info(f"   - {t} ({column_count} columns){vector_info}")

        if not tables:
            logger.warning("⚠️  No tables found. This might indicate:")
            logger.warning("   - No SQLAlchemy models are defined in the Base")
            logger.warning("   - Import issues with the models")
            logger.warning("   - The Base.metadata is empty")

            # Show what's in the Base metadata
            if hasattr(Base, 'metadata') and Base.metadata.tables:
                logger.info("Tables defined in Base.metadata:")
                for table_name in Base.metadata.tables.keys():
                    logger.info(f"   - {table_name}")
            else:
                logger.warning("Base.metadata appears to be empty")

        # Verify vector extension is working
        logger.info("=== Verifying pgvector functionality ===")
        try:
            with engine.connect() as conn:
                result = conn.execute(text("SELECT '[1,2,3]'::vector"))
                logger.info("✅ pgvector extension is working correctly")
        except Exception as e:
            logger.error(f"❌ pgvector verification failed: {e}")

    except SQLAlchemyError as e:
        logger.error(f"❌ SQLAlchemy error while creating tables: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", exc_info=True)


def show_environment_info():
    """Show current environment configuration for debugging"""
    logger.info("=== Environment Information ===")
    env_vars = [
        'DATABASE_URL', 'POSTGRES_USER', 'POSTGRES_PASSWORD',
        'POSTGRES_HOST', 'POSTGRES_PORT', 'POSTGRES_DB',
        'DOCKER_ENVIRONMENT'
    ]

    for var in env_vars:
        value = os.getenv(var)
        if value and 'PASSWORD' in var:
            # Don't log passwords
            logger.info(f"{var}: {'*' * len(value)}")
        elif value:
            logger.info(f"{var}: {value}")
        else:
            logger.info(f"{var}: (not set)")


def check_database_setup():
    """Check if the database and extensions are properly set up"""
    logger.info("=== Database Setup Check ===")

    try:
        # Get database URL
        db_url = os.getenv('DATABASE_URL')
        if not db_url:
            postgres_user = os.getenv('POSTGRES_USER', 'postgres')
            postgres_password = os.getenv('POSTGRES_PASSWORD', 'postgres')
            postgres_host = os.getenv('POSTGRES_HOST', 'localhost')
            postgres_port = os.getenv('POSTGRES_PORT', '5432')
            postgres_db = os.getenv('POSTGRES_DB', 'emtacdb')

            db_url = (
                f"postgresql+psycopg2://{postgres_user}:{postgres_password}"
                f"@{postgres_host}:{postgres_port}/{postgres_db}"
            )

        engine = create_engine(db_url)

        with engine.connect() as conn:
            # Check PostgreSQL version
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            logger.info(f"PostgreSQL version: {version}")

            # Check available extensions
            result = conn.execute(text(
                "SELECT name, installed_version FROM pg_available_extensions "
                "WHERE name IN ('vector', 'pg_trgm', 'unaccent') ORDER BY name"
            ))

            logger.info("Available extensions:")
            for row in result:
                name, version = row
                status = "✅ Installed" if version else "❌ Not installed"
                logger.info(f"   - {name}: {status}")

    except Exception as e:
        logger.error(f"❌ Database setup check failed: {e}")


if __name__ == "__main__":
    show_environment_info()
    print()
    check_database_setup()
    print()
    create_all_tables()