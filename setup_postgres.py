#!/usr/bin/env python3
"""
Fixed PostgreSQL Setup Script that works with your existing DatabaseConfig class
Save as: setup_postgres_fixed.py
Run with: python setup_postgres_fixed.py
"""

import os
import sys
from sqlalchemy import create_engine, text, Column, Integer, String, Text, ForeignKey, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

# Add your project path to sys.path so we can import your modules
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Try to import your existing configuration
try:
    from modules.configuration.config_env import DatabaseConfig

    print("✅ Successfully imported your DatabaseConfig")
    USE_EXISTING_CONFIG = True
except ImportError as e:
    print(f"⚠️  Could not import your DatabaseConfig: {e}")
    print("   Using fallback configuration...")
    USE_EXISTING_CONFIG = False

# Read values from environment, with sensible defaults
POSTGRES_USER = os.getenv("POSTGRES_USER", "emtac_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "emtac123")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "emtac_postgres")   # 👈 updated default
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "emtacdb")

FALLBACK_DATABASE_URL = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@"
    f"{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

print(f"Using database URL: {FALLBACK_DATABASE_URL}")
# Create SQLAlchemy base
Base = declarative_base()


# Define your main models (same as before)
class Part(Base):
    __tablename__ = 'part'

    id = Column(Integer, primary_key=True)
    part_number = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class Image(Base):
    __tablename__ = 'image'

    id = Column(Integer, primary_key=True)
    title = Column(String(200), index=True)
    filename = Column(String(255))
    filepath = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)


class Drawing(Base):
    __tablename__ = 'drawing'

    id = Column(Integer, primary_key=True)
    drw_number = Column(String(100), index=True)
    drw_spare_part_number = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)


class PartsPositionImageAssociation(Base):
    __tablename__ = 'parts_position_image_association'

    id = Column(Integer, primary_key=True)
    part_id = Column(Integer, ForeignKey('part.id'), nullable=False)
    image_id = Column(Integer, ForeignKey('image.id'), nullable=False)
    position_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    part = relationship("Part", backref="image_associations")
    image = relationship("Image", backref="part_associations")


class DrawingPartAssociation(Base):
    __tablename__ = 'drawing_part_association'

    id = Column(Integer, primary_key=True)
    drawing_id = Column(Integer, ForeignKey('drawing.id'), nullable=False)
    part_id = Column(Integer, ForeignKey('part.id'), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    drawing = relationship("Drawing", backref="part_associations")
    part = relationship("Part", backref="drawing_associations")


class UserLogin(Base):
    __tablename__ = 'user_login'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    login_time = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)


class UserLevel(Base):
    __tablename__ = 'user_level'

    id = Column(Integer, primary_key=True)
    level_name = Column(String(50), unique=True, nullable=False)
    description = Column(Text)


class ModelsConfig(Base):
    __tablename__ = 'models_config'

    id = Column(Integer, primary_key=True)
    model_name = Column(String(100), nullable=False)
    model_type = Column(String(20), nullable=False)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


def get_database_connection():
    """Get database engine and session using your existing config or fallback."""

    if USE_EXISTING_CONFIG:
        try:
            print("🔧 Using your existing DatabaseConfig...")
            db_config = DatabaseConfig()
            # Use the main_engine from your DatabaseConfig
            engine = db_config.main_engine
            database_url = db_config.main_database_url
            print(f"   Database URL: {database_url}")
            return engine, db_config
        except Exception as e:
            print(f"❌ Error using existing config: {e}")
            print("   Falling back to direct connection...")

    # Fallback to direct connection
    print("🔧 Using fallback database connection...")
    if POSTGRES_PASSWORD == 'your_actual_password':
        print("❌ Please update POSTGRES_PASSWORD in this script!")
        return None, None

    engine = create_engine(FALLBACK_DATABASE_URL)
    print(f"   Database URL: {FALLBACK_DATABASE_URL}")
    return engine, None


def test_connection():
    """Test database connection."""
    try:
        print("🔍 Testing database connection...")
        engine, db_config = get_database_connection()

        if not engine:
            return False

        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            print(f"✅ Connected successfully!")
            print(f"   PostgreSQL version: {version}")

            result = conn.execute(text("SELECT current_user"))
            user = result.fetchone()[0]
            print(f"   Connected as user: {user}")

            return True

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


def setup_extensions():
    """Set up PostgreSQL extensions."""
    try:
        print("🔧 Setting up PostgreSQL extensions...")
        engine, _ = get_database_connection()

        if not engine:
            return False

        with engine.connect() as conn:
            extensions = ['pg_trgm', 'unaccent', 'uuid-ossp']

            for ext in extensions:
                try:
                    conn.execute(text(f"CREATE EXTENSION IF NOT EXISTS {ext}"))
                    print(f"   ✅ Enabled extension: {ext}")
                except Exception as e:
                    print(f"   ⚠️  Could not enable {ext}: {e}")

            conn.commit()
            return True

    except Exception as e:
        print(f"❌ Error setting up extensions: {e}")
        return False


def create_tables():
    """Create database tables."""
    try:
        print("📋 Creating database tables...")
        engine, _ = get_database_connection()

        if not engine:
            return False

        # Create all tables
        Base.metadata.create_all(engine)
        print("   ✅ All tables created successfully!")

        # List created tables
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """))

            tables = [row[0] for row in result.fetchall()]
            print(f"   📊 Created tables: {', '.join(tables)}")

        return True

    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        print(f"   Error details: {str(e)}")
        return False


def create_indexes():
    """Create useful indexes for better performance."""
    try:
        print("📈 Creating database indexes...")
        engine, _ = get_database_connection()

        if not engine:
            return False

        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_part_number ON part(part_number)",
            "CREATE INDEX IF NOT EXISTS idx_part_description_fts ON part USING gin(to_tsvector('english', COALESCE(description, '')))",
            "CREATE INDEX IF NOT EXISTS idx_image_title ON image(title)",
            "CREATE INDEX IF NOT EXISTS idx_image_filename ON image(filename)",
            "CREATE INDEX IF NOT EXISTS idx_drawing_number ON drawing(drw_number)",
            "CREATE INDEX IF NOT EXISTS idx_drawing_spare_parts ON drawing(drw_spare_part_number)",
            "CREATE INDEX IF NOT EXISTS idx_part_image_assoc ON parts_position_image_association(part_id, image_id)",
            "CREATE INDEX IF NOT EXISTS idx_drawing_part_assoc ON drawing_part_association(drawing_id, part_id)"
        ]

        with engine.connect() as conn:
            for index_sql in indexes:
                try:
                    conn.execute(text(index_sql))
                    print(f"   ✅ Created index")
                except Exception as e:
                    print(f"   ⚠️  Index issue: {e}")

            conn.commit()

        return True

    except Exception as e:
        print(f"❌ Error creating indexes: {e}")
        return False


def insert_sample_data():
    """Insert some sample data to test the setup."""
    try:
        print("📝 Inserting sample data...")
        engine, db_config = get_database_connection()

        if not engine:
            return False

        # Use your existing session factory if available
        if db_config:
            session = db_config.get_main_session()
        else:
            Session = sessionmaker(bind=engine)
            session = Session()

        try:
            # Add sample user levels
            levels = [
                UserLevel(level_name='ADMIN', description='Administrator level'),
                UserLevel(level_name='LEVEL_III', description='Level III user'),
                UserLevel(level_name='STANDARD', description='Standard user')
            ]

            for level in levels:
                existing = session.query(UserLevel).filter_by(level_name=level.level_name).first()
                if not existing:
                    session.add(level)

            # Add sample model configs
            models = [
                ModelsConfig(model_name='gpt-3.5-turbo', model_type='ai', is_active=True),
                ModelsConfig(model_name='text-embedding-ada-002', model_type='embedding', is_active=True)
            ]

            for model in models:
                existing = session.query(ModelsConfig).filter_by(
                    model_name=model.model_name,
                    model_type=model.model_type
                ).first()
                if not existing:
                    session.add(model)

            session.commit()
            print("   ✅ Sample data inserted!")
            return True

        finally:
            session.close()

    except Exception as e:
        print(f"❌ Error inserting sample data: {e}")
        return False


def verify_setup():
    """Verify the database setup is working."""
    try:
        print("🔍 Verifying database setup...")
        engine, db_config = get_database_connection()

        if not engine:
            return False

        # Use your existing session factory if available
        if db_config:
            session = db_config.get_main_session()
        else:
            Session = sessionmaker(bind=engine)
            session = Session()

        try:
            # Count records in each table
            tables_to_check = [
                (Part, 'Parts'),
                (Image, 'Images'),
                (Drawing, 'Drawings'),
                (UserLevel, 'User Levels'),
                (ModelsConfig, 'Model Configs')
            ]

            for model, name in tables_to_check:
                count = session.query(model).count()
                print(f"   📊 {name}: {count} records")

            print("   ✅ Database verification complete!")
            return True

        finally:
            session.close()

    except Exception as e:
        print(f"❌ Error verifying setup: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 PostgreSQL Database Setup Starting...")
    print("=" * 50)

    success_count = 0
    total_steps = 6

    # Step 1: Test connection
    if test_connection():
        success_count += 1
        print("   ✅ Step 1/6: Connection test passed")
    else:
        print("   ❌ Step 1/6: Connection test failed")
        return False

    # Step 2: Setup extensions
    if setup_extensions():
        success_count += 1
        print("   ✅ Step 2/6: Extensions setup completed")
    else:
        print("   ⚠️  Step 2/6: Extensions setup had issues")

    # Step 3: Create tables
    if create_tables():
        success_count += 1
        print("   ✅ Step 3/6: Tables created successfully")
    else:
        print("   ❌ Step 3/6: Table creation failed")
        return False

    # Step 4: Create indexes
    if create_indexes():
        success_count += 1
        print("   ✅ Step 4/6: Indexes created successfully")
    else:
        print("   ⚠️  Step 4/6: Index creation had issues")

    # Step 5: Insert sample data
    if insert_sample_data():
        success_count += 1
        print("   ✅ Step 5/6: Sample data inserted")
    else:
        print("   ⚠️  Step 5/6: Sample data insertion had issues")

    # Step 6: Verify setup
    if verify_setup():
        success_count += 1
        print("   ✅ Step 6/6: Verification completed")
    else:
        print("   ⚠️  Step 6/6: Verification had issues")

    print("\n" + "=" * 50)
    print(f"📊 Setup Results: {success_count}/{total_steps} steps successful")

    if success_count >= 4:  # At least connection, tables, and verification
        print("🎉 PostgreSQL setup completed successfully!")
        print("\n📝 Next steps:")
        print("   1. Your database structure is ready")
        print("   2. Update any remaining SQLite-specific code")
        print("   3. Test your application")
        return True
    else:
        print("❌ Setup completed with issues. Check the errors above.")
        return False


if __name__ == "__main__":
    # Only check password if using fallback config
    if not USE_EXISTING_CONFIG and POSTGRES_PASSWORD == 'your_actual_password':
        print("❌ Please update POSTGRES_PASSWORD in this script!")
        print("   Or make sure your config_env.py is properly configured")
        sys.exit(1)

    success = main()

    if success:
        print("\n🎯 Setup complete! Your PostgreSQL database is ready.")
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)