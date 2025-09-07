#!/usr/bin/env python3
"""
Docker Setup Script for EMTAC Project
This script creates all necessary Docker files in your project root directory.
"""

import os
import sys
from pathlib import Path


def create_dockerfile():
    """Create Dockerfile"""
    content = '''FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (including those needed for your AI models)
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    libpq-dev \\
    curl \\
    wget \\
    git \\
    build-essential \\
    libffi-dev \\
    libssl-dev \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install debugpy for PyCharm debugging (development only)
RUN pip install debugpy

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
  CMD curl -f http://localhost:5000/health || exit 1

# Set environment variable for Docker
ENV DOCKER_ENVIRONMENT=true

# Run the application
CMD ["python", "ai_emtac.py"]'''

    with open('Dockerfile', 'w') as f:
        f.write(content)
    print("Created: Dockerfile")


def create_docker_compose():
    """Create docker-compose.yml"""
    content = '''version: '3.8'

services:
  # PostgreSQL Database with pgvector extension for embeddings
  postgres:
    image: pgvector/pgvector:pg15
    container_name: emtac_postgres
    environment:
      POSTGRES_DB: emtac
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: emtac123
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres -d emtac"]
      interval: 5s
      timeout: 5s
      retries: 5

  # Flask Application
  flask_app:
    build: .
    container_name: emtac_flask_app
    environment:
      DATABASE_URL: postgresql://postgres:emtac123@postgres:5432/emtac
      FLASK_ENV: production
      DOCKER_ENVIRONMENT: "true"
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      HUGGINGFACE_API_KEY: ${HUGGINGFACE_API_KEY}
      CURRENT_AI_MODEL: ${CURRENT_AI_MODEL:-OpenAIModel}
      CURRENT_EMBEDDING_MODEL: ${CURRENT_EMBEDDING_MODEL:-OpenAIEmbeddingModel}
      OPENAI_MODEL_NAME: ${OPENAI_MODEL_NAME:-text-embedding-3-small}
    ports:
      - "5000:5000"
    depends_on:
      postgres:
        condition: service_healthy
    volumes:
      # Mount all the directories your app needs
      - ./static:/app/static
      - ./templates:/app/templates
      - ./Database:/app/Database
      - ./modules:/app/modules
      - ./plugins:/app/plugins
      - ./utilities:/app/utilities
      - ./utility_tools:/app/utility_tools
      - ./load_process:/app/load_process
      - ./logs:/app/logs
    restart: unless-stopped

  # Optional: Redis for caching embeddings/responses
  redis:
    image: redis:7-alpine
    container_name: emtac_redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:'''

    with open('docker-compose.yml', 'w') as f:
        f.write(content)
    print("Created: docker-compose.yml")


def create_requirements():
    """Create requirements.txt"""
    content = '''# Flask Framework
Flask==3.0.0
Flask-CORS==4.0.0
Flask-SQLAlchemy==3.1.1
Flask-Migrate==4.0.5

# Database
psycopg2-binary==2.9.9
pgvector==0.2.4
SQLAlchemy==2.0.23

# AI/ML Libraries (based on your config.py)
openai==1.3.7
sentence-transformers==2.2.2
transformers==4.35.2
torch==2.1.0
numpy==1.24.3
scikit-learn==1.3.2

# Document Processing
PyPDF2==3.0.1
python-docx==1.1.0
openpyxl==3.1.2
python-multipart==0.0.6
Pillow==10.1.0

# Text Processing
tiktoken==0.5.2
nltk==3.8.1

# Environment and Configuration
python-dotenv==1.0.0

# Utilities
requests==2.31.0
gunicorn==21.2.0

# Optional: Redis for caching
redis==5.0.1

# Additional dependencies that might be needed for your AI modules
anthropic==0.8.1
huggingface_hub==0.19.4

# Development dependencies
debugpy==1.8.0'''

    with open('requirements.txt', 'w') as f:
        f.write(content)
    print("Created: requirements.txt")


def create_dockerignore():
    """Create .dockerignore"""
    content = '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/

# Flask
instance/
.flaskenv

# Development
.env
.env.local
.env.development
.env.test
.env.production

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Git
.git/
.gitignore

# Docker
Dockerfile*
docker-compose*
.dockerignore

# Documentation
README.md
*.md

# Logs
*.log

# Testing
.pytest_cache/
.coverage
htmlcov/

# Node modules (if any frontend)
node_modules/

# Temporary files
*.tmp
*.temp

# Exclude large model files and training data during build
# (mount them as volumes instead)
plugins/ai_modules/gpt4all/*.bin
plugins/huggingface/models--*
Database/temp_files/
Database/db_backup/
Database/PPT_FILES/
Database/PDF_FILES/
logs/
log_backup/'''

    with open('.dockerignore', 'w') as f:
        f.write(content)
    print("Created: .dockerignore")


def create_env_example():
    """Create .env.example"""
    content = '''# Database Configuration (matches your config.py)
DATABASE_URL=postgresql://postgres:emtac123@localhost:5432/emtac

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-here

# AI API Keys (from your config.py)
OPENAI_API_KEY=sk-your-openai-api-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
HUGGINGFACE_API_KEY=hf_your-huggingface-token-here

# AI Model Configuration (from your config.py)
CURRENT_AI_MODEL=OpenAIModel
CURRENT_EMBEDDING_MODEL=OpenAIEmbeddingModel
OPENAI_MODEL_NAME=text-embedding-3-small

# Optional: Redis Configuration
REDIS_URL=redis://localhost:6379/0

# Application Settings
MAX_FILE_SIZE=10485760  # 10MB in bytes
ALLOWED_EXTENSIONS=pdf,txt,docx,md,xlsx,jpg,jpeg,png,gif
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Admin Settings (from your config.py)
ADMIN_CREATION_PASSWORD=12345
NUM_VERSIONS_TO_KEEP=3
ENABLE_REVISION_CONTROL=False

# NLP Configuration
nlp_model_name=your-nlp-model-name
auth_token=your-auth-token

# Visual Code API (if needed)
Visual_Code_api=your-visual-code-api-key

# Database connection limiting (optional)
MAX_DB_CONNECTIONS=10
DB_CONNECTION_TIMEOUT=60
CONNECTION_LIMITING_ENABLED=false'''

    with open('.env.example', 'w') as f:
        f.write(content)
    print("Created: .env.example")


def check_env_file():
    """Check if .env file exists and validate it has required keys"""
    if not os.path.exists('.env'):
        print("ERROR: .env file not found!")
        print("Please make sure your .env file is in the project root directory")
        return False

    # Load and check for important keys
    try:
        from dotenv import load_dotenv
        load_dotenv()

        required_keys = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY']
        missing_keys = []

        for key in required_keys:
            if not os.getenv(key):
                missing_keys.append(key)

        if missing_keys:
            print(f"WARNING: Found .env but missing keys: {', '.join(missing_keys)}")
            print("You may need to add these keys to your .env file")
        else:
            print("Found existing .env file with API keys")

        return True
    except ImportError:
        print("NOTE: python-dotenv not installed, skipping .env validation")
        print("Found existing .env file (cannot validate contents)")
        return True


def create_init_sql():
    """Create init.sql for PostgreSQL initialization"""
    content = '''-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create tables for RAG system and document processing
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    document_type VARCHAR(50),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create table for document chunks with embeddings
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    embedding vector(1536), -- Adjust dimension based on your embedding model
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create table for parts (based on your loadsheet structure)
CREATE TABLE IF NOT EXISTS parts (
    id SERIAL PRIMARY KEY,
    part_number VARCHAR(255) UNIQUE,
    part_name VARCHAR(255),
    description TEXT,
    category VARCHAR(100),
    metadata JSONB,
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create table for drawings
CREATE TABLE IF NOT EXISTS drawings (
    id SERIAL PRIMARY KEY,
    drawing_number VARCHAR(255) UNIQUE,
    drawing_name VARCHAR(255),
    description TEXT,
    file_path VARCHAR(500),
    metadata JSONB,
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create table for training data and models
CREATE TABLE IF NOT EXISTS training_datasets (
    id SERIAL PRIMARY KEY,
    dataset_name VARCHAR(255) NOT NULL,
    dataset_type VARCHAR(100), -- intent_classifier, parts, images, etc.
    file_path VARCHAR(500),
    description TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create table for conversation history
CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    user_message TEXT NOT NULL,
    assistant_response TEXT NOT NULL,
    intent_classification VARCHAR(100),
    context_used TEXT[],
    model_used VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding ON document_chunks USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_parts_embedding ON parts USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_drawings_embedding ON drawings USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_parts_part_number ON parts(part_number);
CREATE INDEX IF NOT EXISTS idx_drawings_drawing_number ON drawings(drawing_number);

-- Create table for images (from your comment_images and database structure)
CREATE TABLE IF NOT EXISTS images (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(500),
    image_type VARCHAR(50),
    associated_part_id INTEGER REFERENCES parts(id),
    associated_drawing_id INTEGER REFERENCES drawings(id),
    description TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);'''

    with open('init.sql', 'w') as f:
        f.write(content)
    print("Created: init.sql")


def create_docker_config():
    """Create docker_config.py for Docker environment overrides"""
    content = '''"""Docker-specific configuration overrides for EMTAC project"""
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
    print("Using local environment configuration")'''

    with open('docker_config.py', 'w') as f:
        f.write(content)
    print("Created: docker_config.py")


def main():
    """Main setup function"""
    print("Docker Setup Script for EMTAC Project")
    print("=" * 40)

    # Check if we're in the right directory - look for your specific config structure
    config_paths = [
        'config.py',  # If config.py is in root
        'modules/configuration/config.py',  # Your actual structure
        os.path.join('modules', 'configuration', 'config.py')  # Alternative check
    ]

    config_found = False
    for config_path in config_paths:
        if os.path.exists(config_path):
            config_found = True
            print(f"Found config file at: {config_path}")
            break

    if not config_found:
        print("ERROR: config.py not found!")
        print("Looking for config.py in these locations:")
        for path in config_paths:
            print(f"   - {path}")
        print("\nMake sure you're running this script from your project root directory")
        print("(the directory that contains the 'modules' folder)")
        sys.exit(1)

    # Check for existing .env file
    if not check_env_file():
        sys.exit(1)

    print()
    print("Creating Docker files in current directory...")
    print()

    try:
        # Create all files (except .env since it already exists)
        create_dockerfile()
        create_docker_compose()
        create_requirements()
        create_dockerignore()
        create_env_example()  # Still create as reference
        create_init_sql()
        create_docker_config()

        print()
        print("Docker setup complete!")
        print("=" * 40)
        print("Next steps:")
        print("1. Your existing .env file will be used")
        print("2. Update your existing Flask app to use docker_config.py for Docker compatibility")
        print("3. Run: docker-compose up --build")
        print("4. Open http://localhost:5000 in your browser")
        print()
        print("To make your existing Flask app Docker-compatible, add this at the top:")
        print("   if os.getenv('DOCKER_ENVIRONMENT'):")
        print("       from docker_config import *")
        print("   else:")
        print("       from modules.configuration.config import *")
        print()
        print("For PyCharm integration:")
        print("- File > Settings > Build, Execution, Deployment > Docker")
        print("- Add Docker connection and test")
        print("- File > Settings > Project > Python Interpreter")
        print("- Add > Docker Compose > Select docker-compose.yml > Service: flask_app")
        print()
        print("Files created:")
        files_created = [
            "Dockerfile", "docker-compose.yml", "requirements.txt",
            ".dockerignore", ".env.example", "init.sql",
            "docker_config.py"
        ]
        for f in files_created:
            print(f"  - {f}")

    except Exception as e:
        print(f"ERROR during setup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()