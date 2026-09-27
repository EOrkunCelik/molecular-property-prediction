"""create prediction_history table

Revision ID: 0001_initial
Revises:
Create Date: 2026-09-18

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "prediction_history",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("original_smiles", sa.String(length=2000), nullable=False),
        sa.Column("canonical_smiles", sa.String(length=2000), nullable=False),
        sa.Column("molecular_formula", sa.String(length=200), nullable=False),
        sa.Column("predicted_log_solubility", sa.Float(), nullable=False),
        sa.Column("descriptors", sa.JSON(), nullable=False),
        sa.Column("model_name", sa.String(length=100), nullable=False),
        sa.Column("model_metadata", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_prediction_history_canonical_smiles",
        "prediction_history",
        ["canonical_smiles"],
    )
    op.create_index(
        "ix_prediction_history_created_at",
        "prediction_history",
        ["created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_prediction_history_created_at", table_name="prediction_history")
    op.drop_index("ix_prediction_history_canonical_smiles", table_name="prediction_history")
    op.drop_table("prediction_history")
