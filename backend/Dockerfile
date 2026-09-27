# Build context is the REPOSITORY ROOT (see docker-compose.yml: `context: .`), not
# backend/, because this image needs both backend/app and the shared ml/ package.
FROM python:3.11-slim AS base

# RDKit's wheels need a few system libraries for parsing and headless 2D rendering.
# Keep this list minimal and limited to runtime dependencies required by RDKit.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libexpat1 \
	libxrender1 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt /app/backend-requirements.txt
RUN pip install --no-cache-dir -r /app/backend-requirements.txt

# Shared ML package (feature engineering + inference) used by the backend.
COPY ml/ /app/ml/
# FastAPI application code.
COPY backend/app/ /app/app/
COPY backend/alembic/ /app/alembic/
COPY backend/alembic.ini /app/alembic.ini

# Trained model artifacts (produced by `python scripts/train_model.py`). If this
# directory doesn't exist at build time, the backend will still start but /health
# and /predict will report the model as unavailable until it's provided (e.g. via a
# volume mount — see docker-compose.yml).
COPY models/ /app/models/

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    MODEL_ARTIFACT_DIR=/app/models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
