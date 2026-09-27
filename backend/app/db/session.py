"""Database engine and session management.

`get_db` is a FastAPI dependency that yields a `Session` and guarantees it is closed
after the request, even if the endpoint raises. This is the standard SQLAlchemy 2.0 +
FastAPI pattern and keeps connection handling out of the endpoint/service code.
"""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings

settings = get_settings()

engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
