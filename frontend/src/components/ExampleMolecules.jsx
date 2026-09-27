const EXAMPLES = [
  { name: "Aspirin", smiles: "CC(=O)Oc1ccccc1C(=O)O" },
  { name: "Caffeine", smiles: "Cn1cnc2n(C)c(=O)n(C)c(=O)c12" },
  { name: "Ethanol", smiles: "CCO" },
  { name: "Glucose", smiles: "OCC1OC(O)C(O)C(O)C1O" },
  { name: "Benzene", smiles: "c1ccccc1" },
  { name: "Ibuprofen", smiles: "CC(C)Cc1ccc(cc1)C(C)C(=O)O" },
  { name: "Paracetamol", smiles: "CC(=O)Nc1ccc(O)cc1" },
  { name: "Naphthalene", smiles: "c1ccc2ccccc2c1" },
];

export default function ExampleMolecules({ onSelect, disabled }) {
  return (
    <div className="examples">
      <span className="examples-label">Try an example:</span>
      <div className="examples-list">
        {EXAMPLES.map((ex) => (
          <button
            key={ex.name}
            type="button"
            className="example-chip"
            onClick={() => onSelect(ex.smiles)}
            disabled={disabled}
            title={ex.smiles}
          >
            {ex.name}
          </button>
        ))}
      </div>
    </div>
  );
}
