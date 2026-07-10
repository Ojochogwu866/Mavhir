"""Compute Mordred descriptors for the real Ames and CPDB datasets.

Uses the same descriptor groups as the existing training/serving pipeline
(app/services/descriptor_calculator.py) so retrained models stay compatible
with the deployed inference service's descriptor computation.

Writes data/processed/ames_descriptors.pkl and data/processed/cpdb_descriptors.pkl,
each row = one compound's full descriptor vector + label, so downstream training
doesn't need to recompute descriptors (Mordred is the slow part of this pipeline).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from mordred import Calculator, descriptors
from rdkit import Chem

CORE_GROUPS = [
    descriptors.AtomCount,
    descriptors.BondCount,
    descriptors.RingCount,
    descriptors.Constitutional,
    descriptors.Weight,
    descriptors.SLogP,
]
EXTENDED_GROUP_NAMES = [
    "TopologicalIndex", "Polarizability", "FragmentComplexity", "Framework",
    "Autocorrelation", "BCUT", "DistanceMatrix", "EState", "Aromatic",
    "TopoPSA", "BalabanJ", "BertzCT",
]


def build_calculator() -> Calculator:
    calc = Calculator()
    for group in CORE_GROUPS:
        try:
            calc.register(group)
        except Exception as e:
            print(f"  warning: failed to register {group.__name__}: {e}")
    for name in EXTENDED_GROUP_NAMES:
        if hasattr(descriptors, name):
            try:
                calc.register(getattr(descriptors, name))
            except Exception as e:
                print(f"  warning: failed to register {name}: {e}")
    return calc


def compute_for_dataset(smiles_list, labels, calc, descriptor_names, label_name="label"):
    rows = []
    valid_labels = []
    failed = 0
    start = time.time()
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failed += 1
            continue
        try:
            values = calc(mol)
        except Exception:
            failed += 1
            continue
        clean = []
        for v in values:
            try:
                f = float(v)
                clean.append(f if np.isfinite(f) else 0.0)
            except (TypeError, ValueError):
                clean.append(0.0)
        rows.append(clean)
        valid_labels.append(labels[i])
        if (i + 1) % 200 == 0:
            elapsed = time.time() - start
            print(f"  {i+1}/{len(smiles_list)} processed ({failed} failed), {elapsed:.0f}s elapsed", flush=True)

    df = pd.DataFrame(rows, columns=descriptor_names)
    df[label_name] = valid_labels
    print(f"Done: {len(df)} valid / {len(smiles_list)} total ({failed} failed to parse/compute)")
    return df


def main():
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    calc = build_calculator()
    descriptor_names = [str(d) for d in calc.descriptors]
    print(f"Calculator ready: {len(descriptor_names)} descriptors registered.")

    which = sys.argv[1] if len(sys.argv) > 1 else "both"

    if which in ("ames", "both"):
        print("\n=== Ames (Hansen 2009) ===")
        ames = pd.read_csv("data/raw/ames_hansen2009/Mutagenicity_N6512.csv")
        ames = ames.dropna(subset=["Canonical_Smiles", "Activity"])
        df = compute_for_dataset(
            ames["Canonical_Smiles"].tolist(), ames["Activity"].tolist(), calc, descriptor_names, "ames_label"
        )
        df.to_pickle("data/processed/ames_descriptors.pkl")
        print("Saved data/processed/ames_descriptors.pkl")

    if which in ("cpdb", "both"):
        print("\n=== CPDB carcinogenicity ===")
        cpdb = pd.read_csv("data/raw/cpdb/carcinogenicity_final.csv")
        cpdb = cpdb.dropna(subset=["smiles", "carcinogenicity_label"])
        df = compute_for_dataset(
            cpdb["smiles"].tolist(), cpdb["carcinogenicity_label"].tolist(), calc, descriptor_names, "carc_label"
        )
        df.to_pickle("data/processed/cpdb_descriptors.pkl")
        print("Saved data/processed/cpdb_descriptors.pkl")


if __name__ == "__main__":
    main()
