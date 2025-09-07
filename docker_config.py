"""Docker-specific configuration overrides for EMTAC project"""
import os
import sys

# Import the original configuration
sys.path.append('/app')
from modules.configuration.config import *

# Override configuration for Docker environment
if os.getenv('DOCKER_ENVIRONMENT'):
    print("Loading Docker environment configuration...")

    # Database URL for Docker containers (this will be used by DatabaseConfig)
    DATABASE_URL = os.getenv('DATABASE_URL', "postgresql://postgres:emtac123@postgres:5432/emtac")

    # Override base directory for Docker
    BASE_DIR = '/app'

    # Update all paths to work within Docker container
    TEMPLATE_FOLDER_PATH = os.path.join(BASE_DIR, 'templates')
    LOAD_FOLDER = os.path.join(BASE_DIR, 'load_process')
    LOAD_FOLDER_REFERENCE = os.path.join(BASE_DIR, 'load_process', 'load_reference')
    LOAD_FOLDER_INTAKE = os.path.join(BASE_DIR, 'load_process', 'load_intake_sheets')
    LOAD_FOLDER_OUTPUT = os.path.join(BASE_DIR, 'load_process', 'load_output')
    KEYWORDS_FILE_PATH = os.path.join(BASE_DIR, "static", 'keywords_file.xlsx')
    DATABASE_DIR = os.path.join(BASE_DIR, 'Database')
    DATABASE_PATH = os.path.join(DATABASE_DIR, 'emtac_db.db')
    REVISION_CONTROL_DB_PATH = os.path.join(DATABASE_DIR, 'emtac_revision_control_db.db')
    CSV_DIR = DATABASE_DIR
    COMMENT_IMAGES_FOLDER = os.path.join(BASE_DIR, 'static', 'comment_images')
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
    IMAGES_FOLDER = os.path.join(BASE_DIR, "static", "images")
    DATABASE_PATH_IMAGES_FOLDER = os.path.join(DATABASE_DIR, 'DB_IMAGES')
    PDF_FOR_EXTRACTION_FOLDER = os.path.join(BASE_DIR, "static", "image_extraction")
    IMAGES_EXTRACTED = os.path.join(BASE_DIR, "static", "extracted_pdf_images")
    TEMPORARY_FILES = os.path.join(DATABASE_DIR, 'temp_files')
    PPT2PDF_PPT_FILES_PROCESS = os.path.join(DATABASE_DIR, 'PPT_FILES')
    PPT2PDF_PDF_FILES_PROCESS = os.path.join(DATABASE_DIR, 'PDF_FILES')
    DATABASE_DOC = os.path.join(DATABASE_DIR, 'DB_DOC')
    TEMPORARY_UPLOAD_FILES = os.path.join(DATABASE_DIR, 'temp_upload_files')
    DB_LOADSHEET = os.path.join(DATABASE_DIR, "DB_LOADSHEETS")
    DB_LOADSHEETS_BACKUP = os.path.join(DATABASE_DIR, "DB_LOADSHEETS_BACKUP")
    DB_LOADSHEET_BOMS = os.path.join(DATABASE_DIR, "DB_LOADSHEET_BOMS")
    DRAWING_IMPORT_DATA_DIR = os.path.join(DB_LOADSHEET, "drawing_import_data")
    BACKUP_DIR = os.path.join(DATABASE_DIR, "db_backup")
    Utility_tools = os.path.join(BASE_DIR, "utility_tools")
    UTILITIES = os.path.join(BASE_DIR, 'utilities')

    # Update AI models paths for Docker
    GPT4ALL_MODELS_PATH = os.path.join(BASE_DIR, 'plugins', 'ai_modules', 'gpt4all')
    SENTENCE_TRANSFORMERS_MODELS_PATH = os.path.join(BASE_DIR, 'plugins', 'huggingface')

    # Update orchestrator paths
    ORC_BASE_DIR = os.getenv(
        "ORCHESTRATOR_BASE_DIR",
        os.path.join(BASE_DIR, "modules", "emtac_ai")
    )

    # Recalculate all dependent paths with new BASE_DIR and ORC_BASE_DIR
    ORC_MODELS_DIR = os.path.join(ORC_BASE_DIR, "models")
    ORC_TRAINING_DATA_DIR = os.path.join(ORC_BASE_DIR, "training_data", "datasets")
    ORC_TRAINING_DATA_LOADSHEET = os.path.join(ORC_TRAINING_DATA_DIR, "loadsheet")
    ORC_TRAINING_DATA_ROOT = os.path.join(ORC_BASE_DIR, "training_data")

    # Update all model directories
    ORC_INTENT_MODEL_DIR = os.path.join(ORC_MODELS_DIR, "intent_classifier")
    ORC_PARTS_MODEL_DIR = os.path.join(ORC_MODELS_DIR, "parts")
    ORC_IMAGES_MODEL_DIR = os.path.join(ORC_MODELS_DIR, "images")
    ORC_DOCUMENTS_MODEL_DIR = os.path.join(ORC_MODELS_DIR, "documents")
    ORC_DRAWINGS_MODEL_DIR = os.path.join(ORC_MODELS_DIR, "drawings")
    ORC_TOOLS_MODEL_DIR = os.path.join(ORC_MODELS_DIR, "tools")
    ORC_TROUBLESHOOTING_MODEL_DIR = os.path.join(ORC_MODELS_DIR, "troubleshooting")

    # Update training data directories
    ORC_INTENT_TRAIN_DATA_DIR = os.path.join(ORC_TRAINING_DATA_DIR, "intent_classifier")
    ORC_PARTS_TRAIN_DATA_DIR = os.path.join(ORC_TRAINING_DATA_DIR, "parts")
    ORC_IMAGES_TRAIN_DATA_DIR = os.path.join(ORC_TRAINING_DATA_DIR, "images")
    ORC_DOCUMENTS_TRAIN_DATA_DIR = os.path.join(ORC_TRAINING_DATA_DIR, "documents")
    ORC_DRAWINGS_TRAIN_DATA_DIR = os.path.join(ORC_TRAINING_DATA_DIR, "drawings")
    ORC_TOOLS_TRAIN_DATA_DIR = os.path.join(ORC_TRAINING_DATA_DIR, "tools")
    ORC_TROUBLESHOOTING_TRAIN_DATA_DIR = os.path.join(ORC_TRAINING_DATA_DIR, "troubleshooting")

    # Update query template directories
    ORC_QUERY_TEMPLATES_TRAIN_DATA_DIR = os.path.join(ORC_BASE_DIR, "training_data", "query_templates")
    ORC_QUERY_TEMPLATE_PARTS = os.path.join(ORC_BASE_DIR, "training_data", "query_templates", "parts")
    ORC_QUERY_TEMPLATE_DRAWINGS = os.path.join(ORC_BASE_DIR, "training_data", "query_templates", "drawings")

    # Update other orchestrator paths
    ORC_ORCHESTRATOR_DIR = os.path.join(ORC_BASE_DIR, "orchestrator")
    ORC_ORCHESTRATOR_TEST_SCRIPTS_DIR = os.path.join(ORC_ORCHESTRATOR_DIR, "test_scripts_orchestrator")
    ORC_TEST_SCRIPTS_DIR = os.path.join(ORC_BASE_DIR, "test_scripts")
    ORC_TRAINING_MODULE_DIR = os.path.join(ORC_BASE_DIR, "training_module")
    ORC_TRAINING_DATA_DRAWINGS_LOADSHEET_PATH = os.path.join(ORC_TRAINING_DATA_LOADSHEET, "drawing_loadsheet.xlsx")
    ORC_TRAINING_DATA_PARTS_LOADSHEET_PATH = os.path.join(ORC_TRAINING_DATA_LOADSHEET, "parts_loadsheet.xlsx")
    ORC_TRAINING_SCRIPTS_DIR = os.path.join(ORC_BASE_DIR, "training_scripts")
    ORC_TRAINING_SCRIPTS_DATASET_GEN_DIR = os.path.join(ORC_TRAINING_SCRIPTS_DIR, "dataset_gen")
    ORC_TRAINING_SCRIPTS_INTENT_TRAIN_DIR = os.path.join(ORC_TRAINING_SCRIPTS_DIR, "dataset_intent_train")
    ORC_TRAINING_SCRIPTS_PERFORMANCE_DIR = os.path.join(ORC_TRAINING_SCRIPTS_DIR, "performance_tst_model")
    ORC_TRAINING_SCRIPTS_TST_DIR = os.path.join(ORC_TRAINING_SCRIPTS_DIR, "tst")
    ORC_UTIL_SCRIPTS_DIR = os.path.join(ORC_BASE_DIR, "util_scripts")

    # Redis URL for Docker
    REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379/0')

    # Update directory lists
    directories_to_check = [
        TEMPLATE_FOLDER_PATH,
        DATABASE_DIR,
        UPLOAD_FOLDER,
        IMAGES_FOLDER,
        DATABASE_PATH_IMAGES_FOLDER,
        PDF_FOR_EXTRACTION_FOLDER,
        IMAGES_EXTRACTED,
        TEMPORARY_FILES,
        PPT2PDF_PPT_FILES_PROCESS,
        PPT2PDF_PDF_FILES_PROCESS,
        DATABASE_DOC,
        TEMPORARY_UPLOAD_FILES,
        DB_LOADSHEET,
        DB_LOADSHEETS_BACKUP,
        BACKUP_DIR,
        Utility_tools,
        UTILITIES
    ]

    # Update MODEL_DIRS dictionary
    MODEL_DIRS = {
        "intent_classifier": ORC_INTENT_MODEL_DIR,
        "parts": ORC_PARTS_MODEL_DIR,
        "images": ORC_IMAGES_MODEL_DIR,
        "documents": ORC_DOCUMENTS_MODEL_DIR,
        "drawings": ORC_DRAWINGS_MODEL_DIR,
        "tools": ORC_TOOLS_MODEL_DIR,
        "troubleshooting": ORC_TROUBLESHOOTING_MODEL_DIR,
    }

    # Update TRAIN_DATA_DIRS dictionary
    TRAIN_DATA_DIRS = {
        "intent_classifier": ORC_INTENT_TRAIN_DATA_DIR,
        "parts": ORC_PARTS_TRAIN_DATA_DIR,
        "images": ORC_IMAGES_TRAIN_DATA_DIR,
        "documents": ORC_DOCUMENTS_TRAIN_DATA_DIR,
        "drawings": ORC_DRAWINGS_TRAIN_DATA_DIR,
        "tools": ORC_TOOLS_TRAIN_DATA_DIR,
        "troubleshooting": ORC_TROUBLESHOOTING_TRAIN_DATA_DIR,
    }

    # Update PROJECT_DIRS dictionary
    PROJECT_DIRS = {
        "base": ORC_BASE_DIR,
        "models": ORC_MODELS_DIR,
        "training_data_root": ORC_TRAINING_DATA_ROOT,
        "training_data_datasets": ORC_TRAINING_DATA_DIR,
        "training_data_loadsheet": ORC_TRAINING_DATA_LOADSHEET,
        "query_templates": ORC_QUERY_TEMPLATES_TRAIN_DATA_DIR,
        "query_template_parts": ORC_QUERY_TEMPLATE_PARTS,
        "query_template_drawings": ORC_QUERY_TEMPLATE_DRAWINGS,
        "orchestrator": ORC_ORCHESTRATOR_DIR,
        "orchestrator_test_scripts": ORC_ORCHESTRATOR_TEST_SCRIPTS_DIR,
        "test_scripts": ORC_TEST_SCRIPTS_DIR,
        "training_module": ORC_TRAINING_MODULE_DIR,
        "training_scripts": ORC_TRAINING_SCRIPTS_DIR,
        "training_scripts_dataset_gen": ORC_TRAINING_SCRIPTS_DATASET_GEN_DIR,
        "training_scripts_intent_train": ORC_TRAINING_SCRIPTS_INTENT_TRAIN_DIR,
        "training_scripts_performance": ORC_TRAINING_SCRIPTS_PERFORMANCE_DIR,
        "training_scripts_tst": ORC_TRAINING_SCRIPTS_TST_DIR,
        "util_scripts": ORC_UTIL_SCRIPTS_DIR,
    }

    # Complete list of all directories for setup scripts
    ALL_DIRS = [
        ORC_BASE_DIR,
        ORC_MODELS_DIR,
        ORC_TRAINING_DATA_ROOT,
        ORC_TRAINING_DATA_DIR,
        ORC_TRAINING_DATA_LOADSHEET,
        ORC_QUERY_TEMPLATES_TRAIN_DATA_DIR,
        ORC_QUERY_TEMPLATE_PARTS,
        ORC_QUERY_TEMPLATE_DRAWINGS,
        ORC_ORCHESTRATOR_DIR,
        ORC_ORCHESTRATOR_TEST_SCRIPTS_DIR,
        ORC_TEST_SCRIPTS_DIR,
        ORC_TRAINING_MODULE_DIR,
        ORC_TRAINING_SCRIPTS_DIR,
        ORC_TRAINING_SCRIPTS_DATASET_GEN_DIR,
        ORC_TRAINING_SCRIPTS_INTENT_TRAIN_DIR,
        ORC_TRAINING_SCRIPTS_PERFORMANCE_DIR,
        ORC_TRAINING_SCRIPTS_TST_DIR,
        ORC_UTIL_SCRIPTS_DIR,
    ] + list(MODEL_DIRS.values()) + list(TRAIN_DATA_DIRS.values())

    def ensure_directories():
        """Create necessary directories in Docker container"""
        import os
        all_dirs = directories_to_check + list(MODEL_DIRS.values()) + list(TRAIN_DATA_DIRS.values()) + ALL_DIRS
        for directory in set(all_dirs):  # Remove duplicates
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception as e:
                print(f"Warning: Could not create directory {directory}: {e}")

    # Call this when the app starts
    ensure_directories()
    print("Docker environment configuration loaded successfully")
    print(f"Database URL configured: {DATABASE_URL}")
    print(f"Base directory: {BASE_DIR}")
else:
    print("Using local environment configuration")