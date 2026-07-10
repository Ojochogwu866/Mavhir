# Raw data provenance

## ames_hansen2009/

**Source:** Hansen, K. et al. (2009). "Benchmark Data Set for in Silico Prediction of Ames Mutagenicity." *J. Chem. Inf. Model.* 49, 2077-2081. https://pubs.acs.org/doi/10.1021/ci900161g

**Retrieved:** 2026-07-10, from `http://doc.ml.tu-berlin.de/toxbenchmark/` (the original authors' hosted benchmark site, still live).

**Files:**
- `smiles_cas_N6512.smi` — 6,511 lines. Tab-separated: canonical SMILES, CAS/identifier, activity label (0/1).
- `Mutagenicity_N6512.csv` — 6,512 rows + header. Columns: `CAS_NO, Source, Activity, Steroid, WDI, Canonical_Smiles, REFERENCE`.
- `splits_train_N6512.csv` / `splits_test_N6512.csv` — the original authors' predefined 1×5-fold CV split, as row-per-fold comma-separated compound indices (5 rows each; ~984 compounds per test fold). Using these (or at least reporting against them as a secondary check) allows direct comparison to literature-reported numbers rather than only an arbitrary internal split.

**No further curation done yet.** Not deduplicated against the CPDB set below.

## cpdb/

**Source:** Gold, L.S. et al. — The Carcinogenic Potency Database (CPDB). https://files.toxplanet.com/cpdb/

**Retrieved:** 2026-07-10, from `https://files.toxplanet.com/cpdb/xls/CPDBChemical.xls`.

**File:** `CPDBChemical.xls` — legacy Excel (OLE2/BIFF) format, confirmed authored by Lois Swirsky Gold (metadata: Author, Last Saved By: Thomas Slone). Sheets: `Introduction`, `Rats and Mice` (1,536 chemicals), `Hamsters` (88), `Monkeys` (29), `Other Species` (13).

**Known gaps — do not treat as model-ready:**
1. No SMILES/structure column. Chemical name + CAS number only. The site's dedicated structures page (`structure.html`) links to `toxnet.nlm.nih.gov`, which is dead (NLM decommissioned TOXNET; direct connection fails, confirmed not a transient error). **Next step:** resolve CAS → SMILES via PubChem, following the same method already used in the Niger Delta and pesticide papers.
2. Carcinogenicity call is not a binary column — it's a symbol matrix (`+` / `.` / at least one symbol not cleanly decoded on parse) per species × sex, alongside per-site TD50 potency values. **Next step:** define and document a positive/negative aggregation rule by referencing the database's own legend (`https://files.toxplanet.com/cpdb/pdfs/ChemicalTableLegend.pdf`) before deriving training labels — do not guess at symbol meaning from the raw sheet.

Attempted alternate download paths for a dedicated structures file (`xls/chemical-structures.xls`, `xls/structures.xls`, etc.) all returned HTTP 403 on `files.toxplanet.com` — the working file is `CPDBChemical.xls` only; CAS-based structure resolution via PubChem is the confirmed path forward, not a missing-download-link issue to keep chasing.

### Resolved as of 2026-07-10

Both gaps above are closed:

1. **Carcinogenicity labels derived per the legend** (`pdfs/ChemicalTableLegend.pdf`, downloaded and read — not guessed). Rule used: a compound is **positive** if the CPDB's Rat or Mouse TD50 column contains a numeric value (regardless of superscript codes like `m`, `v`, `i` — these qualify the *evidence*, not whether it's positive at all); **negative** if all tested rodent experiments were negative (`–`); **excluded** if neither rat nor mouse was ever tested (`.`) or the only result was an inadequate NCI/NTP test (`I`). This is the standard "positive in any rodent species" convention. Result: 791 positive / 732 negative / 11 excluded (no usable call), from the "Rats and Mice" sheet (1,536 chemicals) — Hamster/Monkey/Other Species sheets not yet folded in (see below).
   - Superseded by the consolidated, reproducible script below once the Hamster/Monkey/Other-Species merge and CAS-fix (documented next) were folded in.
2. **CAS → SMILES resolved via PubChem PUG REST**, rate-limited at 0.2s/request matching the existing `PUBCHEM_RATE_LIMIT_DELAY` convention already used elsewhere in this codebase. **1,418 of 1,475 resolved (96.1%)**. The 57 unresolved are mixtures, ill-defined polymers, and formulations (e.g. Aroclor 1242, dextran sulfate variants, chlorinated paraffins, Gardenia blue color) that don't have a single well-defined molecular structure — correctly excluded from a SMILES-based dataset, not a resolution failure worth chasing further.

**Final file:** `carcinogenicity_with_smiles.csv` — 1,418 usable compounds, columns `name, cas, carcinogenicity_label, smiles`. Class balance: 751 positive / 667 negative.

### Both follow-up items closed out (2026-07-10)

**1. Hamster/Monkey/Other Species sheets folded in.** While doing this, found and fixed a real bug: Excel/xlrd had silently misparsed CAS numbers matching a date-like pattern (e.g. the real CAS `7758-01-2` for potassium bromate) as literal dates, corrupting them to `7758-01-02 00:00:00`. This affected **26 rows in the original Rats and Mice sheet alone** — meaning the earlier 1,475/1,418 figures had silently and incorrectly excluded legitimate compounds, not just genuine non-chemical entries. Fixed by reconstructing the true CAS format (keeping the 2-digit middle segment zero-padded, stripping only the artificial zero Excel's date formatter added to the 1-digit checksum segment) and re-deriving labels from all four species sheets with the correction applied.
  - Final all-species label set: **1,496 compounds** (784 positive / 712 negative), up from 1,475 (rat/mouse only, pre-fix).
  - Of the newly-recovered/added compounds, 43 of 92 resolved to SMILES via PubChem (47%) — the unresolved remainder are genuine mixtures, technical-grade formulations, and natural-product extracts (Aroclor 1242, dextran/gum/oil mixtures, technical-grade pesticide blends) that don't have a single-molecule structure, not a resolution failure.
  - **Final merged dataset: `carcinogenicity_final.csv` — 1,447 compounds with SMILES (766 positive / 681 negative)**, deduplicated by CAS (26 compounds appeared in more than one species sheet with the same CAS, resolved once, kept once).

**2. Ames/CPDB overlap checked.** 718 of CPDB's 1,447 compounds (~50%) share a CAS number with a compound in the Ames (Hansen 2009) set — a large, expected overlap, since both draw from the same broad universe of well-studied industrial/regulatory compounds. Matching by exact SMILES string only finds 82 of these, which is **not** evidence of fewer real matches — it reflects that Hansen 2009's and PubChem's "canonical" SMILES don't use identical canonicalization rules for the same molecule. **CAS number, not SMILES string equality, is the reliable join key for this check.**
  - **Implication for this project's actual scope (two separate binary models — Ames, carcinogenicity):** this overlap is not a leakage risk as long as each task keeps its own independent split, which is the plan (Ames uses the authors' own 5-fold splits; CPDB gets its own fresh split in Phase 1). It only becomes a leakage concern if a future multi-task model trains on both endpoints jointly — noted here so it isn't forgotten if that's ever considered.
  - **Bonus use:** the 718 shared compounds are a free consistency-check resource — once both models are retrained on real data, their predictions on this shared set can be spot-checked against each other and against known IARC/literature classifications as an extra sanity layer beyond each model's own held-out test set.

### Reproducibility

`cpdb/build_carcinogenicity_dataset.py` is the single, consolidated, rerunnable script that produces `carcinogenicity_final.csv` from `CPDBChemical.xls` — it supersedes the several intermediate exploratory scripts/files used to arrive at this pipeline (deleted after consolidation to avoid confusion about which artifact is authoritative). Requires `xlrd` (for the legacy `.xls` format) and network access (PubChem PUG REST, rate-limited at 0.2s/request — a full run takes roughly 10-15 minutes for ~1,500 compounds).
