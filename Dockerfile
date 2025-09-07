# =========
# Builder image
# =========
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install build dependencies (needed only for compiling wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libexiv2-dev \
    libboost-python-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and build wheels in /wheels
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


# =========
# Runtime image
# =========
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    HF_HOME=/home/appuser/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface \
    HF_DATASETS_CACHE=/home/appuser/.cache/huggingface \
    HF_HUB_CACHE=/home/appuser/.cache/huggingface

# Install only minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libmagic1 \
    tini \
    libexiv2-dev \
    libreoffice \
    socat \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy prebuilt wheels from builder and install
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir /wheels/*

# Add non-root user
RUN useradd -m -u 1000 appuser

# Create Hugging Face cache dir with permissions
RUN mkdir -p /home/appuser/.cache/huggingface && chown -R appuser:appuser /home/appuser/.cache

# Copy application code
COPY --chown=appuser:appuser ai_emtac.py /app/
COPY --chown=appuser:appuser docker_config.py docker_setup.py /app/
COPY --chown=appuser:appuser modules/  /app/modules/
COPY --chown=appuser:appuser plugins/  /app/plugins/
COPY --chown=appuser:appuser templates/ /app/templates/
COPY --chown=appuser:appuser static/    /app/static/
COPY --chown=appuser:appuser blueprints/  /app/blueprints/
COPY --chown=appuser:appuser utilities/ /app/utilities/
COPY check_imports.py /app/
COPY --chown=appuser:appuser clip_healthcheck.py /app/

# Run check (non-fatal)
RUN python /app/check_imports.py || true

# Create writable dirs
RUN install -d -o appuser -g appuser \
    /app/logs \
    /app/log_backup \
    /app/Database \
    /app/Database/DB_IMAGES \
    /app/Database/PDF_FILES \
    /app/Database/PPT_FILES \
    /app/Database/DB_LOADSHEETS_BACKUP \
    /app/Database/temp_upload_files \
    /app/Database/temp_files \
    /app/Database/DB_DOC \
    /app/Database/DB_LOADSHEETS \
    /app/Database/db_backup \
    /app/utility_tools && \
    touch /app/app.log && chown appuser:appuser /app/app.log

USER appuser

EXPOSE 5000

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "ai_emtac.py"]
