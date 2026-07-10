# Model Information

## Model Development Methodology

### Data Collection
- **Ames Data**: Hansen et al. (2009) benchmark, 6,512 compounds (6,506 usable — 6 failed SMILES parsing), from `http://doc.ml.tu-berlin.de/toxbenchmark/`. Full provenance in `data/raw/README.md`.
- **Carcinogenicity Data**: CPDB (Carcinogenic Potency Database), 1,447 compounds, labeled from the database's own legend (positive if a numeric TD50 appears for any tested species; negative if all tested species were negative; excluded if never tested). Full provenance and label-derivation methodology in `data/raw/README.md`.
- **Quality Control**: Automated — SMILES validity via RDKit parsing, descriptor NaN/Inf cleaning, variance and correlation-based feature filtering. No manual curation/expert review is performed; this is an accurate limitation, not a gap to imply otherwise.

### Feature Engineering
- **Descriptors**: 1,156 Mordred molecular descriptors computed per compound (AtomCount, BondCount, RingCount, Constitutional, Weight, SLogP, plus extended topological/autocorrelation/BCUT/EState groups).
- **Selection**: Variance filtering (threshold 1e-6) and correlation filtering (threshold 0.95), both fit on the training split only — final feature count: 549 (Ames), 552 (carcinogenicity).
- **Scaling**: StandardScaler, fit on the training split only.
- **Held out entirely from train/val**: 14 (Ames) / 22 (carcinogenicity) organochlorine pesticide compounds, deliberately excluded to serve as a dedicated generalization probe set (see below) rather than being left to chance in a random split.

### Model Training
- **Algorithm selection**: Random Forest (Ames), Gradient Boosting (carcinogenicity) — carried over from the original design, not re-selected via a new comparison.
- **Hyperparameters**: fixed values (not grid-searched) — see `app/models/train_baseline_v2.py` for exact settings. Grid search was not performed; stating otherwise here previously was inaccurate.
- **Validation**: 5-fold stratified cross-validation on the training split, plus a genuinely held-out test set never touched during feature selection or training. Ames uses the original Hansen et al. authors' own predefined test fold, so results are directly comparable to literature-reported numbers rather than only an internal split.

### Performance Metrics (held-out test set)

| Metric | Ames (RF) | Carcinogenicity (GBM) |
|---|---|---|
| Accuracy | 81.9% | 67.8% |
| Precision | 82.8% | 69.3% |
| Recall | 81.1% | 69.9% |
| F1 | 82.0% | 69.6% |
| AUC-ROC | 0.883 | 0.730 |
| Balanced accuracy | 81.9% | 67.6% |
| Matthews correlation coefficient | 0.638 | 0.353 |

5-fold CV AUC on the training split (Ames: 0.861 ± 0.011, carcinogenicity: 0.666 ± 0.042) is close to the held-out test AUC for Ames; for carcinogenicity, CV AUC is somewhat below held-out test AUC (0.666 vs 0.730), which given the ±0.042 CV standard deviation on this small dataset is within noise rather than a sign of anything systematic, but is noted rather than smoothed over.

**Honest read on the carcinogenicity model:** ~68% accuracy and 0.73 AUC is a real, usable-but-modest result, not a strong one. This is consistent with carcinogenicity being a harder, noisier prediction target than Ames mutagenicity generally (CPDB's cross-species aggregation, discussed in `data/raw/README.md`, also introduces label noise that a cleaner single-species target wouldn't have). It is not a red flag to fix silently — it's the actual current state.

### GNN comparison and organochlorine probe-set results

A GNN (`app/gnn/`, PyTorch Geometric, trained across 5 seeds — mean ± std, not a single run) was trained on identical splits for direct comparison:

| Metric | Ames RF | Ames GNN | Carc. GBM | Carc. GNN |
|---|---|---|---|---|
| Standard test accuracy | 81.9% | 80.1% ± 0.5% | 67.8% | 66.6% ± 4.2% |
| Standard test AUC-ROC | 0.883 | 0.874 ± 0.005 | 0.730 | 0.739 ± 0.027 |

**Organochlorine probe set** (`data/processed/organochlorine_probe_set.json`, 22 curated organochlorine pesticides cross-referenced against IARC classifications, deliberately excluded from train/val — not a random holdout): this was built after an earlier check found that 6 of 7 originally spot-checked pesticide compounds had landed in training data by chance under a random split, invalidating that check.

| Metric | Ames RF probe (n=14) | Ames GNN probe (n=14) | Carc. GBM probe (n=22) | Carc. GNN probe (n=22) |
|---|---|---|---|---|
| Accuracy | 85.7% | 85.7% | 54.5% | 68.2% |
| AUC-ROC | 0.250 | 0.625 | 0.651 | 0.547 |

Both Ames models predict non-mutagenic for **all 14** held-out organochlorines, missing both true positives — the 85.7% accuracy figure is a base-rate artifact (12 of 14 are true negatives), not genuine discrimination. On carcinogenicity, the GNN's higher raw accuracy is similarly an artifact of the probe set being majority-positive (16/22) combined with a specificity of only 16.7%; by AUC-ROC and Matthews correlation, GBM is the more discriminative model on this specific compound class despite its lower headline accuracy on the standard test set.

**Conclusion: neither the descriptor-based baselines nor the GNN reliably generalize to held-out organochlorine pesticides.** Moving to a more expressive architecture did not close the under-prediction gap this project set out to test. Given the small probe-set sizes — particularly n=14 with only 2 positive examples for Ames — the specific AUC comparisons should be read as suggestive rather than conclusive; the robust, sample-size-independent part of the finding is that both architectures made the identical degenerate binary call on Ames.

### From-scratch GNN validation

A second GNN implementation (`app/gnn/model_scratch.py`) was built without PyTorch Geometric's `MessagePassing` base class or any conv layers — message construction, scatter-sum aggregation (`index_add_`), and graph pooling all implemented directly with basic PyTorch tensor ops, as a check that the library version's results reflect real understanding of the mechanism rather than "it's a library that worked." Trained identically (same splits, same probe set, 5 seeds):

| Metric | Library GNN | Scratch GNN |
|---|---|---|
| Ames standard test accuracy | 80.1% | 79.2% |
| Ames standard test AUC-ROC | 0.874 | 0.869 |
| CPDB standard test accuracy | 66.6% | 65.0% |
| CPDB standard test AUC-ROC | 0.739 | 0.734 |

Close agreement on the standard test sets confirms the from-scratch implementation is not broken or degenerate. On the Ames organochlorine probe set, the two implementations produce the **identical** binary decision pattern (non-mutagenic for all 14 compounds, precision/recall/MCC all 0) — a third independently-built model reproducing the same failure mode as the RF baseline is stronger evidence that this is a real property of the data/problem than any single model's result would be.

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
