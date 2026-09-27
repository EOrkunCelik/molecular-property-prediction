"""SQLAlchemy declarative base.

Kept in its own tiny module (rather than inside `models.py`) so that both `models.py`
and `alembic/env.py` can import `Base` without importing each other and risking a
circular import.
"""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
