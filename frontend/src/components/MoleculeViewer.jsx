import { api } from "../api/client.js";

export default function MoleculeViewer({ smiles, predictionId }) {
  if (!smiles) return null;

  const src = predictionId
    ? api.structureImageUrl(predictionId)
    : api.visualizeSmilesUrl(smiles);

  return (
    <div className="molecule-viewer">
      <img src={src} alt={`2D structure of ${smiles}`} width={360} height={270} />
    </div>
  );
}
