"""Cheminformatics + machine learning pipeline for aqueous solubility (LogS) prediction.

This package is intentionally decoupled from the FastAPI backend: the backend imports
`ml.inference` for reproducible prediction, but everything in `ml/` can also be used
standalone from the command line via the scripts in `scripts/`.
"""

__all__ = ["config"]
