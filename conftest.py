"""Root pytest conftest.

Ensures both the top-level `ml` package and the backend's `app` package are
importable when running the full suite from the repository root, e.g.:

    pytest                      # runs tests/ (ml) and backend/tests/ (api) together
    pytest tests/               # ml pipeline tests only
    pytest backend/tests/       # backend API tests only
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "backend"

for path in (ROOT, BACKEND_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
