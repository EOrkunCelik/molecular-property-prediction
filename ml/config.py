"""Central configuration for the ML pipeline.

Keeping these values in one place (instead of scattered magic numbers/strings across
scripts and notebooks) is what makes the pipeline reproducible: the same constants are
used at training time and at inference time.
"""

from __future__ import annotations

from pathlib import Path

# --- Reproducibility ---
RANDOM_SEED: int = 42

# --- Paths ---
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_PATH: Path = DATA_DIR / "raw" / "delaney.csv"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"

# --- Raw dataset schema (ESOL / Delaney solubility dataset) ---
RAW_ID_COL: str = "Compound ID"
RAW_SMILES_COL: str = "SMILES"
RAW_TARGET_COL: str = "measured log(solubility:mol/L)"

# --- Canonical column names used internally after loading/cleaning ---
ID_COL: str = "compound_id"
SMILES_COL: str = "smiles"
CANONICAL_SMILES_COL: str = "canonical_smiles"
TARGET_COL: str = "log_solubility"

# --- Splitting ---
TEST_SIZE: float = 0.15
VAL_SIZE: float = 0.15  # fraction of the full dataset
SCAFFOLD_SPLIT: bool = True  # use Bemis-Murcko scaffold split instead of random split

# --- RDKit 2D descriptors used as the primary feature set ---
# Chosen to be scientifically defensible for a solubility-prediction task: these are
# exactly the physicochemical properties Delaney (2004) and subsequent ESOL-style
# models (e.g. Pat Walters' reproduction) identify as the main drivers of aqueous
# solubility: size/mass, lipophilicity, polarity, hydrogen bonding and flexibility.
DESCRIPTOR_NAMES: list[str] = [
    "MolWt",  # molecular weight (size)
    "MolLogP",  # Crippen LogP (lipophilicity)
    "TPSA",  # topological polar surface area (polarity)
    "NumHDonors",  # H-bond donors
    "NumHAcceptors",  # H-bond acceptors
    "NumRotatableBonds",  # molecular flexibility
    "RingCount",  # number of rings
    "HeavyAtomCount",  # heavy atom count (alternate size proxy)
    "FractionCSP3",  # fraction of sp3 carbons (saturation)
    "NumAromaticRings",  # aromaticity (affects crystallinity/solubility)
    "NumAliphaticRings",
    "NumHeteroatoms",
    "BalabanJ",  # topological complexity index
]

# --- Morgan (ECFP-like) fingerprint settings, evaluated as an alternative representation ---
FINGERPRINT_RADIUS: int = 2
FINGERPRINT_N_BITS: int = 1024

# --- Model registry filenames (relative to MODELS_DIR) ---
MODEL_FILENAME: str = "solubility_model.joblib"
METADATA_FILENAME: str = "model_metadata.json"
METRICS_FILENAME: str = "metrics.json"
