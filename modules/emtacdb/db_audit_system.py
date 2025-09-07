"""
Enhanced Database Audit System for EMTAC
Supports both SQLAlchemy-based and trigger-based auditing
"""

import os
import sys
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, DateTime, Text, Boolean,
    ForeignKey, event, inspect, text, Table, MetaData
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
import json
from typing import Dict, List, Any, Optional

# Import your existing database configuration
try:
    from modules.configuration.config_env import DatabaseConfig
    from modules.emtacdb.emtacdb_fts import Base as MainBase
    from modules.configuration.log_config import info_id, warning_id, error_id
except ImportError:
    print("Warning: Could not import EMTAC modules. Running in standalone mode.")
    MainBase = declarative_base()


class AuditMixin:
    """Mixin class to add audit fields to any table"""

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_by = Column(String(100))
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(String(100))
    version = Column(Integer, default=1, nullable=False)


class AuditLog(MainBase):
    """Central audit log table that tracks all changes"""

    __tablename__ = 'audit_log'

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Table and record information
    table_name = Column(String(100), nullable=False, index=True)
    record_id = Column(String(100), nullable=False, index=True)  # Store as string to handle different ID types

    # Operation details
    operation = Column(String(10), nullable=False)  # INSERT, UPDATE, DELETE
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    user_id = Column(String(100))  # Who made the change
    user_name = Column(String(200))
    session_id = Column(String(100))  # Track user session

    # Change details
    old_values = Column(Text)  # JSON string of old values
    new_values = Column(Text)  # JSON string of new values
    changed_fields = Column(Text)  # JSON array of changed field names

    # Additional context
    ip_address = Column(String(50))
    user_agent = Column(Text)
    application = Column(String(100), default='EMTAC')
    notes = Column(Text)

    # Create indexes for common queries
    __table_args__ = (
        {'schema': None}  # Use default schema
    )


class TableAuditHistory:
    """Dynamic audit table generator for specific tables"""

    @staticmethod
    def create_audit_table_for(original_table: Table, base_class=MainBase) -> Table:
        """Create an audit table that mirrors the structure of the original table"""

        audit_table_name = f"{original_table.name}_audit"

        # Start with audit-specific columns
        columns = [
            Column('audit_id', Integer, primary_key=True, autoincrement=True),
            Column('audit_operation', String(10), nullable=False),  # INSERT, UPDATE, DELETE
            Column('audit_timestamp', DateTime, default=datetime.utcnow, nullable=False),
            Column('audit_user_id', String(100)),
            Column('audit_user_name', String(200)),
            Column('audit_session_id', String(100)),
            Column('audit_ip_address', String(50)),
        ]

        # Add all columns from the original table (except primary key constraints)
        for col in original_table.columns:
            # Create a copy of the column without primary key or unique constraints
            audit_col = col.copy()
            audit_col.primary_key = False
            audit_col.unique = False
            # Make nullable to handle DELETE operations where we might not have all data
            audit_col.nullable = True
            columns.append(audit_col)

        # Create the audit table
        audit_table = Table(
            audit_table_name,
            base_class.metadata,
            *columns,
            schema=original_table.schema
        )

        return audit_table

    @staticmethod
    def create_audit_model_for(original_model_class, base_class=MainBase):
        """Create a dynamic audit model class for the given model"""

        audit_class_name = f"{original_model_class.__name__}Audit"
        audit_table_name = f"{original_model_class.__tablename__}_audit"

        # Create the audit table attributes
        attrs = {
            '__tablename__': audit_table_name,
            '__table_args__': {'extend_existing': True},

            # Audit-specific fields
            'audit_id': Column(Integer, primary_key=True, autoincrement=True),
            'audit_operation': Column(String(10), nullable=False),
            'audit_timestamp': Column(DateTime, default=datetime.utcnow, nullable=False),
            'audit_user_id': Column(String(100)),
            'audit_user_name': Column(String(200)),
            'audit_session_id': Column(String(100)),
            'audit_ip_address': Column(String(50)),
        }

        # Add all columns from the original model
        for column in original_model_class.__table__.columns:
            if column.name not in attrs:  # Don't override audit fields
                # Create a copy without constraints
                col_copy = column.copy()
                col_copy.primary_key = False
                col_copy.unique = False
                col_copy.nullable = True
                attrs[column.name] = col_copy

        # Create the dynamic class
        audit_model = type(audit_class_name, (base_class,), attrs)

        return audit_model


class AuditManager:
    """Main class to manage audit functionality"""

    def __init__(self, db_config: DatabaseConfig, current_user_func=None):
        self.db_config = db_config
        self.current_user_func = current_user_func or self._default_user_func
        self.audit_models = {}
        self.setup_complete = False

    def _default_user_func(self):
        """Default function to get current user info"""
        return {
            'user_id': 'system',
            'user_name': 'System User',
            'session_id': str(uuid.uuid4()),
            'ip_address': '127.0.0.1'
        }

    def setup_auditing(self, models_to_audit: List = None):
        """Set up auditing for specified models or all models"""

        try:
            info_id("🔍 Setting up database auditing system...")

            # Create audit log table
            with self.db_config.main_session() as session:
                # Check if audit_log table exists
                inspector = inspect(self.db_config.main_engine)
                if 'audit_log' not in inspector.get_table_names():
                    info_id("🏗️ Creating central audit_log table...")
                    AuditLog.__table__.create(self.db_config.main_engine)
                    info_id("✅ Central audit_log table created")

            # Get all models to audit
            if models_to_audit is None:
                models_to_audit = self._get_all_models()

            # Set up auditing for each model
            for model_class in models_to_audit:
                self._setup_model_auditing(model_class)

            info_id(f"✅ Auditing setup complete for {len(models_to_audit)} models")
            self.setup_complete = True

        except Exception as e:
            error_id(f"❌ Failed to setup auditing: {e}")
            raise

    def _get_all_models(self):
        """Get all SQLAlchemy models from the main base"""
        models = []
        for cls in MainBase.registry._class_registry.values():
            if hasattr(cls, '__tablename__') and cls != AuditLog:
                models.append(cls)
        return models

    def _setup_model_auditing(self, model_class):
        """Set up auditing for a specific model"""

        try:
            model_name = model_class.__name__
            info_id(f"🔧 Setting up auditing for {model_name}...")

            # Create audit model
            audit_model = TableAuditHistory.create_audit_model_for(model_class, MainBase)
            self.audit_models[model_name] = audit_model

            # Create audit table
            with self.db_config.main_session() as session:
                inspector = inspect(self.db_config.main_engine)
                audit_table_name = f"{model_class.__tablename__}_audit"

                if audit_table_name not in inspector.get_table_names():
                    info_id(f"🏗️ Creating audit table: {audit_table_name}")
                    audit_model.__table__.create(self.db_config.main_engine)

            # Set up event listeners
            self._setup_event_listeners(model_class, audit_model)

            info_id(f"✅ Auditing setup complete for {model_name}")

        except Exception as e:
            error_id(f"❌ Failed to setup auditing for {model_class.__name__}: {e}")

    def _setup_event_listeners(self, model_class, audit_model):
        """Set up SQLAlchemy event listeners for audit logging"""

        @event.listens_for(model_class, 'after_insert')
        def after_insert(mapper, connection, target):
            self._log_change('INSERT', target, None, target, connection)

        @event.listens_for(model_class, 'after_update')
        def after_update(mapper, connection, target):
            # Get the old values from the session
            old_values = {}
            new_values = {}

            # Get changed attributes
            state = inspect(target)
            for attr in state.attrs:
                if state.attrs[attr].history.has_changes():
                    old_value = state.attrs[attr].history.deleted
                    new_value = state.attrs[attr].history.added
                    if old_value:
                        old_values[attr.key] = old_value[0]
                    if new_value:
                        new_values[attr.key] = new_value[0]

            self._log_change('UPDATE', target, old_values, new_values, connection)

        @event.listens_for(model_class, 'after_delete')
        def after_delete(mapper, connection, target):
            self._log_change('DELETE', target, target, None, connection)

    def _log_change(self, operation: str, target, old_values, new_values, connection):
        """Log a change to both central audit log and specific audit table"""

        try:
            user_info = self.current_user_func()
            table_name = target.__tablename__
            record_id = str(getattr(target, 'id', 'unknown'))

            # Prepare values for JSON serialization
            old_json = self._serialize_values(old_values) if old_values else None
            new_json = self._serialize_values(new_values) if new_values else None

            # Log to central audit table
            audit_entry = AuditLog(
                table_name=table_name,
                record_id=record_id,
                operation=operation,
                user_id=user_info.get('user_id'),
                user_name=user_info.get('user_name'),
                session_id=user_info.get('session_id'),
                ip_address=user_info.get('ip_address'),
                old_values=old_json,
                new_values=new_json,
                changed_fields=json.dumps(list(new_values.keys())) if new_values else None
            )

            # Insert using the connection (we're in an event handler)
            connection.execute(
                AuditLog.__table__.insert().values(**{
                    'table_name': table_name,
                    'record_id': record_id,
                    'operation': operation,
                    'timestamp': datetime.utcnow(),
                    'user_id': user_info.get('user_id'),
                    'user_name': user_info.get('user_name'),
                    'session_id': user_info.get('session_id'),
                    'ip_address': user_info.get('ip_address'),
                    'old_values': old_json,
                    'new_values': new_json,
                    'changed_fields': json.dumps(list(new_values.keys())) if new_values else None
                })
            )

            # Also log to specific audit table if it exists
            model_name = target.__class__.__name__
            if model_name in self.audit_models:
                audit_model = self.audit_models[model_name]

                # Prepare audit record data
                audit_data = {
                    'audit_operation': operation,
                    'audit_timestamp': datetime.utcnow(),
                    'audit_user_id': user_info.get('user_id'),
                    'audit_user_name': user_info.get('user_name'),
                    'audit_session_id': user_info.get('session_id'),
                    'audit_ip_address': user_info.get('ip_address'),
                }

                # Add the actual record data
                values_to_use = new_values if operation != 'DELETE' else old_values
                if values_to_use:
                    for key, value in values_to_use.items():
                        if hasattr(audit_model, key):
                            audit_data[key] = value

                # Insert into specific audit table
                connection.execute(
                    audit_model.__table__.insert().values(**audit_data)
                )

        except Exception as e:
            # Don't let audit failures break the main operation
            print(f"Audit logging failed: {e}")

    def _serialize_values(self, values):
        """Serialize values for JSON storage"""
        if not values:
            return None

        serializable = {}
        for key, value in values.items():
            try:
                # Handle different data types
                if isinstance(value, datetime):
                    serializable[key] = value.isoformat()
                elif hasattr(value, '__dict__'):
                    # For complex objects, just store the string representation
                    serializable[key] = str(value)
                else:
                    serializable[key] = value
            except:
                serializable[key] = str(value)

        return json.dumps(serializable, default=str)

    def get_audit_history(self, table_name: str, record_id: str, limit: int = 100):
        """Get audit history for a specific record"""

        with self.db_config.main_session() as session:
            return session.query(AuditLog).filter(
                AuditLog.table_name == table_name,
                AuditLog.record_id == str(record_id)
            ).order_by(AuditLog.timestamp.desc()).limit(limit).all()

    def get_user_activity(self, user_id: str, days: int = 30):
        """Get recent activity for a specific user"""

        from datetime import timedelta
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        with self.db_config.main_session() as session:
            return session.query(AuditLog).filter(
                AuditLog.user_id == user_id,
                AuditLog.timestamp >= cutoff_date
            ).order_by(AuditLog.timestamp.desc()).all()


# PostgreSQL Trigger-based Auditing (Alternative approach)
class PostgreSQLAuditTriggers:
    """Create PostgreSQL triggers for automatic auditing"""

    def __init__(self, db_config: DatabaseConfig):
        self.db_config = db_config

    def create_audit_triggers(self, table_names: List[str] = None):
        """Create PostgreSQL triggers for audit logging"""

        if not self.db_config.is_postgresql:
            warning_id("⚠️ Trigger-based auditing is only available for PostgreSQL")
            return

        try:
            info_id("🔧 Creating PostgreSQL audit triggers...")

            with self.db_config.main_session() as session:
                # Create the generic audit function
                self._create_audit_function(session)

                # Get table names if not provided
                if table_names is None:
                    inspector = inspect(self.db_config.main_engine)
                    table_names = [t for t in inspector.get_table_names()
                                   if not t.endswith('_audit') and t != 'audit_log']

                # Create triggers for each table
                for table_name in table_names:
                    self._create_table_trigger(session, table_name)

                session.commit()
                info_id(f"✅ Created audit triggers for {len(table_names)} tables")

        except Exception as e:
            error_id(f"❌ Failed to create audit triggers: {e}")
            raise

    def _create_audit_function(self, session):
        """Create the PostgreSQL function that handles audit logging"""

        audit_function_sql = """
        CREATE OR REPLACE FUNCTION audit_trigger_function()
        RETURNS TRIGGER AS $$
        BEGIN
            IF TG_OP = 'DELETE' THEN
                INSERT INTO audit_log (
                    table_name, record_id, operation, timestamp,
                    old_values, user_id, session_id
                ) VALUES (
                    TG_TABLE_NAME,
                    OLD.id::text,
                    TG_OP,
                    NOW(),
                    row_to_json(OLD)::text,
                    current_setting('audit.user_id', true),
                    current_setting('audit.session_id', true)
                );
                RETURN OLD;
            ELSIF TG_OP = 'UPDATE' THEN
                INSERT INTO audit_log (
                    table_name, record_id, operation, timestamp,
                    old_values, new_values, user_id, session_id
                ) VALUES (
                    TG_TABLE_NAME,
                    NEW.id::text,
                    TG_OP,
                    NOW(),
                    row_to_json(OLD)::text,
                    row_to_json(NEW)::text,
                    current_setting('audit.user_id', true),
                    current_setting('audit.session_id', true)
                );
                RETURN NEW;
            ELSIF TG_OP = 'INSERT' THEN
                INSERT INTO audit_log (
                    table_name, record_id, operation, timestamp,
                    new_values, user_id, session_id
                ) VALUES (
                    TG_TABLE_NAME,
                    NEW.id::text,
                    TG_OP,
                    NOW(),
                    row_to_json(NEW)::text,
                    current_setting('audit.user_id', true),
                    current_setting('audit.session_id', true)
                );
                RETURN NEW;
            END IF;
            RETURN NULL;
        END;
        $$ LANGUAGE plpgsql;
        """

        session.execute(text(audit_function_sql))
        info_id("✅ Created PostgreSQL audit function")

    def _create_table_trigger(self, session, table_name: str):
        """Create audit trigger for a specific table"""

        trigger_name = f"{table_name}_audit_trigger"

        # Drop existing trigger
        drop_trigger_sql = f"""
        DROP TRIGGER IF EXISTS {trigger_name} ON {table_name};
        """

        # Create new trigger
        create_trigger_sql = f"""
        CREATE TRIGGER {trigger_name}
            AFTER INSERT OR UPDATE OR DELETE ON {table_name}
            FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();
        """

        session.execute(text(drop_trigger_sql))
        session.execute(text(create_trigger_sql))

        info_id(f"✅ Created audit trigger for table: {table_name}")


# Utility functions for easy setup
def setup_complete_audit_system(db_config: DatabaseConfig,
                                models_to_audit: List = None,
                                use_triggers: bool = False,
                                current_user_func=None):
    """Complete audit system setup"""

    try:
        info_id("🚀 Setting up complete audit system...")

        # Create audit manager
        audit_manager = AuditManager(db_config, current_user_func)

        # Setup SQLAlchemy-based auditing
        audit_manager.setup_auditing(models_to_audit)

        # Setup trigger-based auditing if requested and PostgreSQL
        if use_triggers and db_config.is_postgresql:
            trigger_manager = PostgreSQLAuditTriggers(db_config)
            trigger_manager.create_audit_triggers()

        info_id("✅ Complete audit system setup finished!")
        return audit_manager

    except Exception as e:
        error_id(f"❌ Failed to setup complete audit system: {e}")
        raise


# Example usage and testing
def example_usage():
    """Example of how to use the audit system"""

    # Initialize database config (your existing setup)
    db_config = DatabaseConfig()

    # Define custom user function
    def get_current_user():
        return {
            'user_id': 'admin123',
            'user_name': 'Admin User',
            'session_id': str(uuid.uuid4()),
            'ip_address': '192.168.1.100'
        }

    # Setup complete audit system
    audit_manager = setup_complete_audit_system(
        db_config=db_config,
        models_to_audit=None,  # Audit all models
        use_triggers=True,  # Also use PostgreSQL triggers
        current_user_func=get_current_user
    )

    # Now all changes to your models will be automatically audited!

    # Query audit history
    history = audit_manager.get_audit_history('parts', '123', limit=50)
    for entry in history:
        print(f"{entry.timestamp}: {entry.operation} by {entry.user_name}")

    # Get user activity
    activity = audit_manager.get_user_activity('admin123', days=7)
    print(f"User has {len(activity)} activities in the last 7 days")


if __name__ == "__main__":
    # Run example
    example_usage()