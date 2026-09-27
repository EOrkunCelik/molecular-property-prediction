import MoleculeViewer from "./MoleculeViewer.jsx";
import DescriptorTable from "./DescriptorTable.jsx";

function interpretSolubility(logS) {
  // Rough, textbook qualitative solubility classes based on log(mol/L),
  // adapted from the "solubility yardstick" used in early-stage drug discovery
  // (e.g. Lipinski's rule-of-thumb solubility categories). This is a simplified,
  // illustrative interpretation, not a regulatory or pharmacopeial classification.
  if (logS >= 0) return "Very soluble";
  if (logS >= -2) return "Soluble";
  if (logS >= -4) return "Moderately soluble";
  if (logS >= -6) return "Poorly soluble";
  return "Practically insoluble";
}

export default function PredictionResult({ result }) {
  if (!result) return null;

  const {
    canonical_smiles,
    molecular_formula,
    predicted_log_solubility,
    predicted_solubility_mol_per_l,
    descriptors,
    model_info,
    id,
  } = result;

  return (
    <div className="prediction-result">
      <div className="result-header">
        <div>
          <h2>{molecular_formula}</h2>
          <code className="canonical-smiles">{canonical_smiles}</code>
        </div>
        <MoleculeViewer smiles={canonical_smiles} predictionId={id} />
      </div>

      <div className="prediction-headline">
        <div className="predicted-value">
          <span className="predicted-number">{predicted_log_solubility.toFixed(3)}</span>
          <span className="predicted-unit">log₁₀(mol/L)</span>
        </div>
        <div className="predicted-secondary">
          ≈ {predicted_solubility_mol_per_l.toExponential(2)} mol/L —{" "}
          <strong>{interpretSolubility(predicted_log_solubility)}</strong>
        </div>
      </div>

      <p className="explainer">
        This is the predicted <strong>aqueous solubility</strong>, expressed as
        log₁₀ of the molar solubility (mol/L) — the same target quantity as the
        ESOL / Delaney dataset this model was trained on. Higher values mean more
        soluble. This is an experimental, educational prediction and is{" "}
        <strong>not a substitute for a laboratory solubility measurement</strong>.
      </p>

      <h3>Molecular descriptors</h3>
      <DescriptorTable descriptors={descriptors} />

      {model_info && (
        <p className="model-footnote">
          Predicted by <code>{model_info.model_name}</code>
          {model_info.split_strategy ? ` · ${model_info.split_strategy} split` : ""}
          {model_info.test_metrics
            ? ` · test RMSE ${Number(model_info.test_metrics.rmse).toFixed(3)}, R² ${Number(
                model_info.test_metrics.r2
              ).toFixed(3)}`
            : ""}
        </p>
      )}
    </div>
  );
}
