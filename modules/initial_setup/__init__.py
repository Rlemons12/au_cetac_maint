"""Package initializer (auto-generated but cleaned)."""

import logging

logger = logging.getLogger(__name__)

# === AUTO-IMPORTS: BEGIN (managed) ===

# --- Optional audit imports (wrapped safely) ---
try:
    from .audit_system_setup import AuditSystemSetup, main as audit_main, project_root, script_dir
    from .setup_audit_system import setup_basic_audit_system
    from .setup_audit_triggers import AuditTriggersManager
except ImportError as e:
    AuditSystemSetup = None
    audit_main = None
    project_root = None
    script_dir = None
    setup_basic_audit_system = None
    AuditTriggersManager = None
    logger.warning(f"⚠️ Audit system skipped: {e}")

from utility_tools.auto_init import (
    MANAGED_END,
    MANAGED_START,
    build_imports_block,
    get_public_symbols,
    main as auto_main,
    write_init,
)
from .bootstrap import create_essential_dirs
from .emtac_postgres_db_initlal_setup import (
    DatabaseConfig,
    MainBase,
    RevisionControlBase,
    check_and_create_database_schema,
    check_existing_data,
    check_prerequisites,
    create_base_directories,
    create_directories,
    directories_to_check,
    display_setup_summary,
    early_database_creation,
    import_modules_after_directory_setup,
    main as postgres_setup_main,
    run_post_setup_ai_configuration,
    run_post_setup_associations,
    run_setup_scripts,
    setup_state,
    setup_virtual_environment_and_install_requirements,
    setup_windows_console,
    suggest_database_backup,
    test_database_connection,
)
from modules.initial_setup.emtacdb_initial_setup import (
    DatabaseConfig as InitDBConfig,
    MainBase as InitMainBase,
    RevisionControlBase as InitRevisionBase,
    check_and_create_database,
    create_base_directories as init_create_base_dirs,
    create_directories as init_create_dirs,
    directories_to_check as init_dirs_check,
    import_modules_after_directory_setup as init_import_modules,
    main as emtacdb_init_main,
    run_post_setup_ai_configuration as init_ai_config,
    run_post_setup_associations as init_associations,
    run_setup_scripts as init_run_scripts,
    setup_virtual_environment_and_install_requirements as init_setup_venv,
)
from .init__ import logger as init_logger
from .initial_admin import ADMIN_CREATION_PASSWORD, PostgreSQLAdminCreator, main as admin_main, prompt_for_admin_password
from .initializer_logger import (
    BASE_DIR,
    LOG_DIRECTORY,
    close_initializer_logger,
    compress_logs_except_most_recent,
    console_handler,
    initializer_file_handler,
    initializer_log_path,
    initializer_logger,
    log_formatter,
    log_initialization_step,
)
from .load_active_drawing_list import PostgreSQLDrawingListLoader, main as drawing_list_main
from .load_bom_loadsheet import PostgreSQLBOMLoadsheetProcessor, main as bom_main
from .load_equipment_relationships_table_data import PostgreSQLEquipmentRelationshipsLoader, main as equip_rel_main
from .load_image_folder import OptimizedImageFolderProcessor, main as image_loader_main
from .load_parts_sheet import OptimizedPostgreSQLPartsSheetLoader, main as parts_loader_main
from .load_parts_sheet.py_2 import SuperOptimizedPostgreSQLPartsSheetLoader, main as super_parts_loader_main
from .position_creation_from_rel import OptimizedPositionCreator, main as position_creator_main
from .test_setup import (
    DummyBase,
    DummyEngine,
    DummyInspector,
    DummyLogger,
    dummy_logger,
    fake_subprocess,
    mock_root,
    module_path,
    rcm,
    setup,
    snap,
    spec,
    test_check_and_create_database_creates,
    test_check_and_create_database_skips,
    test_check_and_install_requirements_installs,
    test_check_and_install_requirements_skips,
    test_create_directories,
    test_run_setup_scripts,
)
from modules.initial_setup.intent_NER_model_db_setup import (
    ensure_extensions,
    create_tables_and_indexes,
    analyze_vector_tables,
    verify_setup,
    seed_labelset_parts_ner,
    seed_labelset_drawings_ner,
    seed_labelset_intents_from_dirs,
    discover_templates_root,
    ingest_query_templates,
    cleanup_ingested_templates,
    optional_seed_parts_pcm,
    optional_seed_drawings_pcm,
    debug_dump_everything,
)

# === AUTO-IMPORTS: END (managed) ===


# === AUTO-IMPORTS: END (managed) ===

__all__ = [
    # keep your existing exports...
    "DatabaseConfig",
    "MainBase",
    "RevisionControlBase",
    "check_and_create_database_schema",
    "check_existing_data",
    "check_prerequisites",
    "create_base_directories",
    "create_directories",
    "directories_to_check",
    "display_setup_summary",
    "early_database_creation",
    "import_modules_after_directory_setup",
    "postgres_setup_main",
    "run_post_setup_ai_configuration",
    "run_post_setup_associations",
    "run_setup_scripts",
    "setup_state",
    "setup_virtual_environment_and_install_requirements",
    "setup_windows_console",
    "suggest_database_backup",
    "test_database_connection",
    "InitDBConfig",
    "InitMainBase",
    "InitRevisionBase",
    "check_and_create_database",
    "init_create_base_dirs",
    "init_create_dirs",
    "init_dirs_check",
    "init_import_modules",
    "emtacdb_init_main",
    "init_ai_config",
    "init_associations",
    "init_run_scripts",
    "init_setup_venv",
    "logger",
    "ADMIN_CREATION_PASSWORD",
    "PostgreSQLAdminCreator",
    "admin_main",
    "prompt_for_admin_password",
    "BASE_DIR",
    "LOG_DIRECTORY",
    "close_initializer_logger",
    "compress_logs_except_most_recent",
    "console_handler",
    "initializer_file_handler",
    "initializer_log_path",
    "initializer_logger",
    "log_formatter",
    "log_initialization_step",
    "PostgreSQLDrawingListLoader",
    "drawing_list_main",
    "PostgreSQLBOMLoadsheetProcessor",
    "bom_main",
    "PostgreSQLEquipmentRelationshipsLoader",
    "equip_rel_main",
    "OptimizedImageFolderProcessor",
    "image_loader_main",
    "OptimizedPostgreSQLPartsSheetLoader",
    "parts_loader_main",
    "SuperOptimizedPostgreSQLPartsSheetLoader",
    "super_parts_loader_main",
    "OptimizedPositionCreator",
    "position_creator_main",
    "DummyBase",
    "DummyEngine",
    "DummyInspector",
    "DummyLogger",
    "dummy_logger",
    "fake_subprocess",
    "mock_root",
    "module_path",
    "rcm",
    "setup",
    "snap",
    "spec",
    "test_check_and_create_database_creates",
    "test_check_and_create_database_skips",
    "test_check_and_install_requirements_installs",
    "test_check_and_install_requirements_skips",
    "test_create_directories",
    "test_run_setup_scripts",
    "ensure_extensions",
    "create_tables_and_indexes",
    "analyze_vector_tables",
    "verify_setup",
    "seed_labelset_parts_ner",
    "seed_labelset_drawings_ner",
    "seed_labelset_intents_from_dirs",
    "discover_templates_root",
    "ingest_query_templates",
    "cleanup_ingested_templates",
    "optional_seed_parts_pcm",
    "optional_seed_drawings_pcm",
    "debug_dump_everything",
]
