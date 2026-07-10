# Model Information

## Model Development Methodology

### Data Collection
- **Ames Data**: Hansen et al. (2009) benchmark, 6,512 compounds (6,506 usable — 6 failed SMILES parsing), from `http://doc.ml.tu-berlin.de/toxbenchmark/`. Full provenance in `data/raw/README.md`.
- **Carcinogenicity Data**: CPDB (Carcinogenic Potency Database), 1,447 compounds, labeled from the database's own legend (positive if a numeric TD50 appears for any tested species; negative if all tested species were negative; excluded if never tested). Full provenance and label-derivation methodology in `data/raw/README.md`.
- **Quality Control**: Automated — SMILES validity via RDKit parsing, descriptor NaN/Inf cleaning, variance and correlation-based feature filtering. No manual curation/expert review is performed; this is an accurate limitation, not a gap to imply otherwise.

### Feature Engineering
- **Descriptors**: 1,156 Mordred molecular descriptors computed per compound (AtomCount, BondCount, RingCount, Constitutional, Weight, SLogP, plus extended topological/autocorrelation/BCUT/EState groups).
- **Selection**: Variance filtering (threshold 1e-6) and correlation filtering (threshold 0.95), both fit on the training split only — final feature count: 552 (Ames), 550 (carcinogenicity).
- **Scaling**: StandardScaler, fit on the training split only.

### Model Training
- **Algorithm selection**: Random Forest (Ames), Gradient Boosting (carcinogenicity) — carried over from the original design, not re-selected via a new comparison.
- **Hyperparameters**: fixed values (not grid-searched) — see `app/models/train_baseline_v2.py` for exact settings. Grid search was not performed; stating otherwise here previously was inaccurate.
- **Validation**: 5-fold stratified cross-validation on the training split, plus a genuinely held-out test set never touched during feature selection or training. Ames uses the original Hansen et al. authors' own predefined test fold, so results are directly comparable to literature-reported numbers rather than only an internal split.

### Performance Metrics (held-out test set)

| Metric | Ames (RF) | Carcinogenicity (GBM) |
|---|---|---|
| Accuracy | 81.3% | 65.1% |
| Precision | 82.7% | 68.2% |
| Recall | 79.7% | 63.5% |
| F1 | 81.2% | 65.8% |
| AUC-ROC | 0.884 | 0.704 |
| Balanced accuracy | 81.3% | 65.2% |
| Matthews correlation coefficient | 0.626 | 0.304 |

5-fold CV AUC on the training split (Ames: 0.861 ± 0.014, carcinogenicity: 0.705 ± 0.033) is close to the held-out test AUC in both cases — no indication of overfitting to the training split.

**Honest read on the carcinogenicity model:** 65.1% accuracy and 0.70 AUC is a real, usable-but-modest result, not a strong one. This is consistent with carcinogenicity being a harder, noisier prediction target than Ames mutagenicity generally (CPDB's cross-species aggregation, discussed in `data/raw/README.md`, also introduces label noise that a cleaner single-species target wouldn't have). It is not a red flag to fix silently — it's the actual current state, and the GNN work in `docs/gnn_extension_design.md` treats this as one of the numbers a graph-based model needs to try to beat.

## Model Limitations

### Applicability Domain
- Organic compounds with MW 50-1000 Da
- Standard drug-like chemical space
- May not be suitable for:
  - Metal complexes
  - Large biologics
  - Novel chemical classes

### Data Limitations
- CPDB's carcinogenicity call aggregates across species/sex using an "positive in any tested species" rule (documented in `data/raw/README.md`) — this is a defensible convention but not the only possible one, and a compound's call can be sensitive to which rule is chosen.
- No manual expert curation of either dataset beyond the automated pipeline described above.
- Roughly 718 of the 1,447 carcinogenicity compounds also appear in the Ames dataset (by CAS number) — not a leakage concern for these two independently-trained, independently-split single-task models, but relevant if a future multi-task model trains on both endpoints jointly.
