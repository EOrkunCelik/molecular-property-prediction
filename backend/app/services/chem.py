"""Chemistry service layer.

Thin wrapper around `ml.data.validation` and `ml.features.descriptors` so that
endpoint/service code in the backend never imports `rdkit` directly. This keeps RDKit
usage confined to two places in the whole codebase (the `ml` package and this module),
which is where you'd look first if RDKit ever needed to be upgraded or swapped.
"""

from __future__ import annotations

from ml.data.validation import ValidationResult, validate_smiles
from ml.features.descriptors import describe_molecule

# Re-export the single canonical exception type (defined once, in ml.inference) rather
# than declaring a second `InvalidSMILESError` class here. An earlier version of this
# module defined its own subclass of ValueError with the same name; because Python
# matches `except` clauses by actual class identity, that meant `except InvalidSMILESError`
# in the API layer (which imported *this* module's class) silently failed to catch the
# exception actually raised by `ml.inference.SolubilityPredictor.predict()` (which raises
# *its own* module's class) — invalid SMILES submitted to POST /predict would have
# fallen through to the generic 500 handler instead of returning the intended 422.
# Importing (not redefining) the same class here closes that gap for good: there is now
# exactly one `InvalidSMILESError` in the whole codebase, so every `except` clause that
# imports it — from `ml.inference`, `app.services.chem`, or `app.services.prediction_service`
# — refers to the identical class.
from ml.inference import InvalidSMILESError  # noqa: F401


def validate_and_describe(smiles: str) -> dict:
    """Validate a SMILES string and, if valid, return its canonical form, formula and descriptors.

    Raises InvalidSMILESError with a human-readable message if the SMILES is invalid.
    """
    result: ValidationResult = validate_smiles(smiles)
    if not result.is_valid:
        raise InvalidSMILESError(result.error or f"Invalid SMILES string: {smiles!r}")

    return describe_molecule(result.canonical_smiles)
