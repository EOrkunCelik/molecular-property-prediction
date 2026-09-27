import { useState, useEffect } from "react";
import ExampleMolecules from "./ExampleMolecules.jsx";

export default function SmilesForm({ onSubmit, loading, externalSmiles }) {
  const [smiles, setSmiles] = useState("CCO");

  useEffect(() => {
    if (externalSmiles) setSmiles(externalSmiles);
  }, [externalSmiles]);

  function handleSubmit(e) {
    e.preventDefault();
    if (!smiles.trim()) return;
    onSubmit(smiles.trim());
  }

  return (
    <form className="smiles-form" onSubmit={handleSubmit}>
      <label htmlFor="smiles-input" className="field-label">
        SMILES string
      </label>
      <div className="smiles-input-row">
        <input
          id="smiles-input"
          type="text"
          value={smiles}
          onChange={(e) => setSmiles(e.target.value)}
          placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O"
          spellCheck={false}
          autoComplete="off"
        />
        <button type="submit" disabled={loading || !smiles.trim()}>
          {loading ? "Predicting…" : "Predict solubility"}
        </button>
      </div>
      <ExampleMolecules onSelect={setSmiles} disabled={loading} />
    </form>
  );
}
