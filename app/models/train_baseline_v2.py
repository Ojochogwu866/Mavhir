"""Retrain the Ames (RandomForest) and carcinogenicity (GradientBoosting) baselines
on the real Hansen 2009 / CPDB datasets, replacing the toy ~100-compound generator.

Key differences from the original train_models.py, addressing the issues identified
in docs/gnn_extension_design.md:
  - Real data (6,512 Ames compounds; 1,447 CPDB compounds), not a hardcoded example list.
  - Ames uses the original authors' own train/test split (fold 0 of their published
    5-fold CV) so results are directly comparable to literature-reported numbers,
    not just an arbitrary internal split.
  - CPDB gets a fresh stratified 70/15/15 train/val/test split.
  - Variance/correlation feature filtering and scaling are fit on the TRAINING fold
    only, then applied to val/test — the original train_models.py's _clean_descriptors
    ran filtering on the full dataset before splitting, a mild leakage risk that
    only becomes material with a real-sized dataset.
"""
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    balanced_accuracy_score, matthews_corrcoef, confusion_matrix,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

MODELS_DIR = Path("app/models")
PROCESSED_DIR = Path("data/processed")

with open(PROCESSED_DIR / "organochlorine_probe_set.json") as f:
    PROBE_CAS = set(json.load(f)["cas_numbers"])


def probe_metrics(model, scaler, X_probe_c, y_probe):
    """Evaluate on the deliberately-held-out organochlorine probe set.
    X_probe_c is already feature-selected (same columns as train), just needs scaling."""
    if X_probe_c is None or len(X_probe_c) == 0:
        return None
    X_probe_s = scaler.transform(X_probe_c)
    y_pred = model.predict(X_probe_s)
    y_proba = model.predict_proba(X_probe_s)[:, 1] if hasattr(model, "predict_proba") else y_pred
    return {
        "n": int(len(y_probe)),
        "accuracy": float(accuracy_score(y_probe, y_pred)),
        "predictions": [{"true": int(t), "pred": int(p), "prob": float(pr)} for t, p, pr in zip(y_probe, y_pred, y_proba)],
    }


def clean_and_select_features(X_train, X_val, X_test, feature_names, X_probe=None, variance_threshold=1e-6, corr_threshold=0.95):
    """Fit variance/correlation filtering on TRAIN ONLY, apply the same selected
    columns to val/test/probe. This is the leakage fix relative to the original pipeline."""
    train_df = pd.DataFrame(X_train, columns=feature_names)

    variances = train_df.var()
    keep = variances[variances > variance_threshold].index.tolist()
    train_df = train_df[keep]

    corr = train_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
    keep = [c for c in keep if c not in to_drop]
    train_df = train_df[keep]  # narrow train to the final feature set too, not just val/test

    val_df = pd.DataFrame(X_val, columns=feature_names)[keep]
    test_df = pd.DataFrame(X_test, columns=feature_names)[keep]
    probe_df = pd.DataFrame(X_probe, columns=feature_names)[keep] if X_probe is not None else None

    return train_df.values, val_df.values, test_df.values, (probe_df.values if probe_df is not None else None), keep


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "matthews_corrcoef": float(matthews_corrcoef(y_test, y_pred)),
    }


def train_ames():
    print("=== Ames mutagenicity (RandomForest) ===")
    df = pd.read_pickle(PROCESSED_DIR / "ames_descriptors.pkl")
    feature_names = [c for c in df.columns if c != "ames_label"]
    X, y = df[feature_names].values, df["ames_label"].values.astype(int)

    ames_raw = pd.read_csv("data/raw/ames_hansen2009/Mutagenicity_N6512.csv").dropna(
        subset=["Canonical_Smiles", "Activity"]
    ).reset_index(drop=True)

    if len(ames_raw) != len(df):
        # compute_descriptors.py silently drops rows RDKit can't parse (6 known cases,
        # rare diazo/stereo-bond SMILES notation). Re-derive which original rows those
        # were (fast: just a parse check, not a full Mordred recomputation) so the
        # authors' split indices -- which reference positions in the full 6,512-row
        # file -- still line up with this filtered descriptor set.
        from rdkit import Chem
        valid_mask = ames_raw["Canonical_Smiles"].apply(lambda s: Chem.MolFromSmiles(s) is not None)
        n_dropped = (~valid_mask).sum()
        print(f"Aligning: {n_dropped} rows failed to parse and were dropped during descriptor computation")
        assert valid_mask.sum() == len(df), (
            f"alignment failed: {valid_mask.sum()} parseable rows vs {len(df)} descriptor rows"
        )
        ames_raw = ames_raw[valid_mask].reset_index(drop=True)
        original_positions = np.where(valid_mask.values)[0]
    else:
        original_positions = np.arange(len(df))

    with open("data/raw/ames_hansen2009/splits_test_N6512.csv") as f:
        test_idx_fold0 = set(int(i) for i in f.readline().strip().split(","))
    # map original-file test indices onto positions in the filtered (aligned) array
    test_mask = np.array([pos in test_idx_fold0 for pos in original_positions])

    cas_aligned = ames_raw["CAS_NO"].astype(str).str.strip().values
    probe_mask = np.isin(cas_aligned, list(PROBE_CAS))
    print(f"Organochlorine probe compounds found in Ames dataset: {probe_mask.sum()} (excluded from train/val entirely)")

    trainval_mask = (~test_mask) & (~probe_mask)
    X_trainval, y_trainval = X[trainval_mask], y[trainval_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    X_probe, y_probe = X[probe_mask], y[probe_mask]
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.15, random_state=42, stratify=y_trainval
    )
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test (authors' fold 0): {len(X_test)}, Probe: {len(X_probe)}")

    X_train_c, X_val_c, X_test_c, X_probe_c, kept_features = clean_and_select_features(
        X_train, X_val, X_test, feature_names, X_probe=X_probe
    )
    print(f"Features after train-only variance/correlation filtering: {len(kept_features)} (from {len(feature_names)})")

    scaler = StandardScaler().fit(X_train_c)
    X_train_s, X_test_s = scaler.transform(X_train_c), scaler.transform(X_test_c)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2,
        random_state=42, class_weight="balanced", n_jobs=-1, oob_score=True,
    )
    model.fit(X_train_s, y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="roc_auc")
    print(f"5-fold CV AUC (train only): {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    metrics = evaluate(model, X_test_s, y_test)
    print("Held-out test (authors' fold 0):", json.dumps(metrics, indent=2))

    probe_result = probe_metrics(model, scaler, X_probe_c, y_probe)
    print("Organochlorine probe set:", json.dumps(probe_result, indent=2) if probe_result else "no probe compounds in this dataset")

    with open(MODELS_DIR / "ames_mutagenicity.pkl", "wb") as f:
        pickle.dump(model, f, protocol=4)
    with open(MODELS_DIR / "ames_mutagenicity_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f, protocol=4)

    return {
        "model_type": "RandomForestClassifier",
        "n_features": len(kept_features),
        "feature_names": kept_features,
        "probe_set_result": probe_result,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "version": "2.0",
        "data_source": "Hansen et al. 2009 benchmark (6,512 compounds), http://doc.ml.tu-berlin.de/toxbenchmark/",
        "split": "authors' predefined 5-fold CV, fold 0 held out as test; remaining 85/15 train/val",
        "train_size": len(X_train), "val_size": len(X_val), "test_size": len(X_test),
        "cv_auc_mean": float(cv_scores.mean()), "cv_auc_std": float(cv_scores.std()),
        "metrics": metrics,
    }


def train_carcinogenicity():
    print("\n=== Carcinogenicity (GradientBoosting) ===")
    df = pd.read_pickle(PROCESSED_DIR / "cpdb_descriptors.pkl")
    feature_names = [c for c in df.columns if c != "carc_label"]
    X, y = df[feature_names].values, df["carc_label"].values.astype(int)

    cpdb_raw = pd.read_csv("data/raw/cpdb/carcinogenicity_final.csv").dropna(
        subset=["smiles", "carcinogenicity_label"]
    ).reset_index(drop=True)
    assert len(cpdb_raw) == len(df), "descriptor row count must match source row count for CAS alignment"
    cas_aligned = cpdb_raw["cas"].astype(str).str.strip().values
    probe_mask = np.isin(cas_aligned, list(PROBE_CAS))
    print(f"Organochlorine probe compounds found in CPDB dataset: {probe_mask.sum()} (excluded from train/val entirely)")

    X_probe, y_probe = X[probe_mask], y[probe_mask]
    X_remaining, y_remaining = X[~probe_mask], y[~probe_mask]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_remaining, y_remaining, test_size=0.15, random_state=42, stratify=y_remaining
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.15 / 0.85, random_state=42, stratify=y_trainval
    )
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}, Probe: {len(X_probe)}")

    X_train_c, X_val_c, X_test_c, X_probe_c, kept_features = clean_and_select_features(
        X_train, X_val, X_test, feature_names, X_probe=X_probe
    )
    print(f"Features after train-only variance/correlation filtering: {len(kept_features)} (from {len(feature_names)})")

    scaler = StandardScaler().fit(X_train_c)
    X_train_s, X_test_s = scaler.transform(X_train_c), scaler.transform(X_test_c)

    model = GradientBoostingClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.1, subsample=0.8, random_state=42,
    )
    model.fit(X_train_s, y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="roc_auc")
    print(f"5-fold CV AUC (train only): {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    metrics = evaluate(model, X_test_s, y_test)
    print("Held-out test:", json.dumps(metrics, indent=2))

    probe_result = probe_metrics(model, scaler, X_probe_c, y_probe)
    print("Organochlorine probe set:", json.dumps(probe_result, indent=2) if probe_result else "no probe compounds")

    with open(MODELS_DIR / "carcinogenicity.pkl", "wb") as f:
        pickle.dump(model, f, protocol=4)
    with open(MODELS_DIR / "carcinogenicity_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f, protocol=4)

    return {
        "model_type": "GradientBoostingClassifier",
        "n_features": len(kept_features),
        "feature_names": kept_features,
        "probe_set_result": probe_result,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "version": "2.0",
        "data_source": "CPDB (Carcinogenic Potency Database), all species sheets, https://files.toxplanet.com/cpdb/ "
                        "-- see data/raw/README.md for label-derivation methodology",
        "split": "fresh stratified 70/15/15 train/val/test",
        "train_size": len(X_train), "val_size": len(X_val), "test_size": len(X_test),
        "cv_auc_mean": float(cv_scores.mean()), "cv_auc_std": float(cv_scores.std()),
        "metrics": metrics,
    }


def main():
    ames_meta = train_ames()
    carc_meta = train_carcinogenicity()

    metadata = {"ames_mutagenicity": ames_meta, "carcinogenicity": carc_meta}
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("\nSaved app/models/model_metadata.json")


if __name__ == "__main__":
    main()
