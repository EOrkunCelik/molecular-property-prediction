const DESCRIPTOR_META = {
  MolWt: { label: "Molecular weight", unit: "g/mol", digits: 2 },
  MolLogP: { label: "LogP (lipophilicity)", unit: "", digits: 2 },
  TPSA: { label: "Polar surface area (TPSA)", unit: "Å²", digits: 1 },
  NumHDonors: { label: "H-bond donors", unit: "", digits: 0 },
  NumHAcceptors: { label: "H-bond acceptors", unit: "", digits: 0 },
  NumRotatableBonds: { label: "Rotatable bonds", unit: "", digits: 0 },
  RingCount: { label: "Ring count", unit: "", digits: 0 },
  HeavyAtomCount: { label: "Heavy atom count", unit: "", digits: 0 },
  FractionCSP3: { label: "Fraction sp³ carbons", unit: "", digits: 2 },
  NumAromaticRings: { label: "Aromatic rings", unit: "", digits: 0 },
  NumAliphaticRings: { label: "Aliphatic rings", unit: "", digits: 0 },
  NumHeteroatoms: { label: "Heteroatoms", unit: "", digits: 0 },
  BalabanJ: { label: "Balaban J index", unit: "", digits: 2 },
};

export default function DescriptorTable({ descriptors }) {
  if (!descriptors) return null;

  return (
    <table className="descriptor-table">
      <tbody>
        {Object.entries(descriptors).map(([key, value]) => {
          const meta = DESCRIPTOR_META[key] || { label: key, unit: "", digits: 2 };
          const formatted =
            typeof value === "number" ? value.toFixed(meta.digits) : String(value);
          return (
            <tr key={key}>
              <td className="descriptor-label">{meta.label}</td>
              <td className="descriptor-value">
                {formatted}
                {meta.unit ? ` ${meta.unit}` : ""}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
