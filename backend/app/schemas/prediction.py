"""Pydantic schemas: the API's request/response contract.

Kept separate from the SQLAlchemy models (`app.db.models`) on purpose — the API shape
(e.g. `predicted_solubility_mol_per_l`, a derived/display-only field) is not
necessarily the same as the storage shape, and coupling them would make either the API
or the schema harder to evolve independently.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictionRequest(BaseModel):
    smiles: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="A SMILES (Simplified Molecular-Input Line-Entry System) string describing the molecule.",
        examples=["CCO"],
    )

    @field_validator("smiles")
    @classmethod
    def smiles_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("smiles must not be blank")
        return v.strip()


class MolecularDescriptors(BaseModel):
    """Human-readable molecular descriptors, matching `ml.config.DESCRIPTOR_NAMES`."""

    model_config = ConfigDict(extra="allow")  # future-proof: new descriptors don't break the schema

    MolWt: float = Field(..., description="Molecular weight (g/mol)")
    MolLogP: float = Field(..., description="Crippen LogP (octanol/water partition coefficient)")
    TPSA: float = Field(..., description="Topological polar surface area (Å²)")
    NumHDonors: int = Field(..., description="Number of hydrogen bond donors")
    NumHAcceptors: int = Field(..., description="Number of hydrogen bond acceptors")
    NumRotatableBonds: int = Field(..., description="Number of rotatable bonds")
    RingCount: int = Field(..., description="Number of rings")
    HeavyAtomCount: int = Field(..., description="Number of heavy (non-hydrogen) atoms")
    FractionCSP3: float = Field(..., description="Fraction of sp3-hybridized carbons")
    NumAromaticRings: int
    NumAliphaticRings: int
    NumHeteroatoms: int
    BalabanJ: float = Field(..., description="Balaban's J topological index")


class ModelInfo(BaseModel):
    model_name: str
    trained_at_utc: str | None = None
    split_strategy: str | None = None
    test_metrics: dict | None = None


class PredictionResponse(BaseModel):
    id: str
    original_smiles: str
    canonical_smiles: str
    molecular_formula: str
    descriptors: MolecularDescriptors
    predicted_log_solubility: float = Field(
        ..., description="Predicted log10(solubility) in mol/L — the model's native output unit."
    )
    predicted_solubility_mol_per_l: float = Field(
        ..., description="predicted_log_solubility converted back to mol/L (10**predicted_log_solubility), for convenience."
    )
    model_info: ModelInfo
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PredictionListItem(BaseModel):
    """Lighter-weight shape used in the paginated history list endpoint."""

    id: str
    original_smiles: str
    canonical_smiles: str
    predicted_log_solubility: float
    model_name: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PredictionListResponse(BaseModel):
    items: list[PredictionListItem]
    total: int
    limit: int
    offset: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None
    database_connected: bool
