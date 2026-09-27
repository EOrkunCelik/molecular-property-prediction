.PHONY: help install download-data train evaluate test test-ml test-backend lint format \
        docker-up docker-down docker-build migrate backend-dev frontend-dev

help:
	@echo "Available targets:"
	@echo "  install         Install Python dependencies (root + backend) into the current environment"
	@echo "  download-data   Download the raw ESOL dataset (idempotent)"
	@echo "  train           Run the full ML training pipeline and save the model to models/"
	@echo "  evaluate        Re-evaluate the currently saved model on the persisted test split"
	@echo "  test            Run the full test suite (ML pipeline + backend API)"
	@echo "  test-ml         Run only the ML pipeline tests"
	@echo "  test-backend    Run only the backend API tests"
	@echo "  lint            Run ruff lint checks"
	@echo "  format          Auto-format with ruff"
	@echo "  docker-up       Build and start the full stack (db + backend + frontend)"
	@echo "  docker-down     Stop the full stack"
	@echo "  migrate         Run Alembic migrations against the configured database"
	@echo "  backend-dev     Run the FastAPI backend locally with auto-reload"
	@echo "  frontend-dev    Run the React frontend locally with hot reload"

install:
	pip install -r requirements.txt
	pip install -r backend/requirements.txt

download-data:
	python scripts/download_data.py

train:
	python scripts/train_model.py

evaluate:
	python scripts/evaluate_model.py

test:
	pytest

test-ml:
	pytest tests/

test-backend:
	pytest backend/tests/

lint:
	ruff check .

format:
	ruff format .
	ruff check --fix .

docker-up:
	docker compose up --build

docker-down:
	docker compose down

migrate:
	cd backend && alembic upgrade head

backend-dev:
	cd backend && uvicorn app.main:app --reload --port 8000

frontend-dev:
	cd frontend && npm run dev
