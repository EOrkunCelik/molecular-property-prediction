"""SQLAlchemy ORM models.

Only one table for this project: `prediction_history`, which stores every prediction
made through the API. Using `JSON` (rather than Postgres-specific `JSONB`) for the
descriptor column keeps the model portable enough to run against SQLite in tests
without a second code path; in production (Postgres) SQLAlchemy still stores it as a
proper `json`/`jsonb`-compatible column.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class PredictionRecord(Base):
    __tablename__ = "prediction_history"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    original_smiles: Mapped[str] = mapped_column(String(2000), nullable=False)
    canonical_smiles: Mapped[str] = mapped_column(String(2000), nullable=False, index=True)
    molecular_formula: Mapped[str] = mapped_column(String(200), nullable=False)

    predicted_log_solubility: Mapped[float] = mapped_column(Float, nullable=False)

    descriptors: Mapped[dict] = mapped_column(JSON, nullable=False)

    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_metadata: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return (
            f"<PredictionRecord id={self.id} smiles={self.canonical_smiles!r} "
            f"prediction={self.predicted_log_solubility:.3f}>"
        )
