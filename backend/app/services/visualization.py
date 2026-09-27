"""2D molecular structure visualization.

Renders SVG rather than a raster format (PNG) because SVG is text-based (easy to
return directly as an HTTP response with the right content-type, no binary encoding
headaches), scales cleanly at any size in the browser, and produces tiny payloads for
small molecules — all good fits for a REST API + React frontend.
"""

from __future__ import annotations

from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")


def render_structure_svg(smiles: str, width: int = 400, height: int = 300) -> str:
    """Render a molecule to a 2D SVG structure drawing.

    Raises ValueError if the SMILES cannot be parsed (callers are expected to validate
    first via `app.services.chem.validate_and_describe`, but this guard keeps the
    function safe to call in isolation too, e.g. from a standalone visualization
    endpoint that only receives a raw SMILES string).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES for visualization: {smiles!r}")

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().addStereoAnnotation = True
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    # RDKit emits an XML namespace declaration that some strict SVG parsers dislike
    # when embedded inline; leaving it as-is is fine for both <img src="data:..."> and
    # a dedicated image/svg+xml HTTP response, so no post-processing is needed.
    return svg
