#!/usr/bin/env python3
"""Download the raw ESOL (Delaney) aqueous solubility dataset.

The dataset (1,144 organic molecules with measured aqueous solubility) was originally
published as a supplementary file to:

    Delaney, J. S. "ESOL: Estimating Aqueous Solubility Directly from Molecular
    Structure." J. Chem. Inf. Comput. Sci. 44.3 (2004): 1000-1005.

and is commonly redistributed (MIT-licensed) as a plain CSV. This repository already
ships a copy at `data/raw/delaney.csv` so the project is runnable offline out of the
box; this script exists so the download step is documented and reproducible, and so
the file can be refreshed if needed.

Usage:
    python scripts/download_data.py            # skip if the file already exists
    python scripts/download_data.py --force     # re-download even if it exists
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

SOURCE_URL = "https://raw.githubusercontent.com/dataprofessor/data/master/delaney.csv"
DEST_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "delaney.csv"


def download(force: bool = False) -> None:
    if DEST_PATH.exists() and not force:
        print(f"[skip] {DEST_PATH} already exists (use --force to re-download).")
        return

    DEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from {SOURCE_URL} ...")
    try:
        urllib.request.urlretrieve(SOURCE_URL, DEST_PATH)
    except Exception as exc:
        print(
            f"[error] Could not download the dataset automatically ({exc}).\n"
            f"You can download it manually from {SOURCE_URL} and save it to {DEST_PATH}.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    with open(DEST_PATH) as f:
        n_lines = sum(1 for _ in f)
    print(f"Saved {DEST_PATH} ({n_lines - 1} molecules).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download even if the file exists")
    args = parser.parse_args()
    download(force=args.force)
