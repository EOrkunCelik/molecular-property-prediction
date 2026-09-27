# Dataset

**ESOL (Delaney) aqueous solubility dataset** — 1,144 organic molecules with measured
water solubility.

- **Source paper:** Delaney, J. S. "ESOL: Estimating Aqueous Solubility Directly from
  Molecular Structure." *J. Chem. Inf. Comput. Sci.* 44.3 (2004): 1000–1005.
- **File:** `data/raw/delaney.csv`
- **License:** the redistributed CSV (via the commonly-used
  [dataprofessor/data](https://github.com/dataprofessor/data) mirror) is MIT-licensed.
- **Columns used:**
  | Column | Meaning |
  |---|---|
  | `Compound ID` | Common/IUPAC-ish name of the compound |
  | `measured log(solubility:mol/L)` | **Target variable** — log₁₀ of the measured aqueous solubility in mol/L |
  | `SMILES` | SMILES string describing the molecular structure |

  (The CSV also ships Delaney's own 1980s-era linear-regression ESOL prediction as a
  reference column, `ESOL predicted log(solubility:mol/L)`. It is **not** used
  anywhere in this project's pipeline — it exists purely as a historical baseline you
  could optionally compare against.)

## Why this dataset

- It is small enough (~1,100 molecules) to train and iterate on quickly, while still
  being a real, widely-used cheminformatics benchmark (it's part of
  [MoleculeNet](https://moleculenet.org/)), so results are comparable to published
  work.
- Aqueous solubility is scientifically well-motivated: it's one of the first
  properties assessed in early-stage drug discovery (a compound that can't dissolve
  can't be absorbed), which makes the project's stated portfolio goal (cheminformatics
  + drug discovery relevance) genuine rather than decorative.
- The target is continuous, giving a real regression problem with standard,
  interpretable metrics (MAE, RMSE, R²) rather than a toy classification task.

## Known limitations of the dataset itself

- ~1,100 molecules is small by modern ML standards; expect meaningfully wider
  confidence intervals on held-out metrics than a dataset 100x this size.
- It skews toward small, drug-like and industrially-relevant organic molecules; the
  model should not be expected to extrapolate well to, e.g., large biomolecules,
  organometallics, or highly exotic structures.
- Measured solubility itself has non-trivial experimental uncertainty (different
  sources report different values for the same compound); the dataset's target values
  inherit that noise, which puts a hard ceiling on achievable model accuracy no matter
  how good the model is.

See the main [README](../README.md) for how this dataset flows through the pipeline
(cleaning → validation → scaffold split → featurization → modeling).
