"""Train and evaluate the GNN baseline on the same splits as the RF/GBM models
(train_baseline_v2.py), for a direct, apples-to-apples comparison.

Reports mean +/- std across multiple seeds, not a single run -- the biggest rigor
gap identified relative to BioNexus's single-run evaluations.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    balanced_accuracy_score, matthews_corrcoef, confusion_matrix,
)
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.gnn.featurize import smiles_to_graph, ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from app.gnn.model import MoleculeGNN


def build_graphs(smiles_list, labels):
    graphs = []
    dropped = 0
    for smi, lab in zip(smiles_list, labels):
        g = smiles_to_graph(smi, label=float(lab))
        if g is None:
            dropped += 1
            continue
        graphs.append(g)
    return graphs, dropped


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels).astype(int)
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
    }


def train_one_seed(train_graphs, val_graphs, test_graphs, seed, epochs, device, pos_weight):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MoleculeGNN(ATOM_FEATURE_DIM, BOND_FEATURE_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=64, shuffle=False)

    best_val_auc, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        val_metrics = evaluate(model, val_loader, device)
        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0:
            print(f"    seed {seed} epoch {epoch+1}/{epochs}: train_loss={total_loss/len(train_graphs):.4f}, val_auc={val_metrics['roc_auc']:.4f}", flush=True)

    model.load_state_dict(best_state)  # early-stopping on best val AUC, not final epoch
    test_metrics = evaluate(model, test_loader, device)
    return test_metrics, best_val_auc, best_state


def run_task(name, smiles_list, labels, test_mask, epochs, seeds, probe_mask=None):
    device = torch.device("cpu")
    print(f"\n=== {name}: building graphs ===")
    graphs, dropped = build_graphs(smiles_list, labels)
    print(f"{len(graphs)} graphs built, {dropped} dropped (unparseable SMILES)")

    # test_mask/probe_mask are aligned to smiles_list/labels order; rebuild against the
    # (possibly shorter, if any dropped) graphs list by re-deriving indices
    valid_indices = []
    idx = 0
    for smi in smiles_list:
        g = smiles_to_graph(smi)
        if g is not None:
            valid_indices.append(idx)
        idx += 1
    test_mask_aligned = test_mask[valid_indices] if len(valid_indices) == len(graphs) else test_mask[:len(graphs)]
    if probe_mask is not None:
        probe_mask_aligned = probe_mask[valid_indices] if len(valid_indices) == len(graphs) else probe_mask[:len(graphs)]
    else:
        probe_mask_aligned = np.zeros(len(graphs), dtype=bool)

    print(f"Organochlorine probe compounds found: {probe_mask_aligned.sum()} (excluded from train/val entirely)")

    probe_graphs = [g for g, m in zip(graphs, probe_mask_aligned) if m]
    test_graphs = [g for g, tm, pm in zip(graphs, test_mask_aligned, probe_mask_aligned) if tm and not pm]
    trainval_graphs = [g for g, tm, pm in zip(graphs, test_mask_aligned, probe_mask_aligned) if not tm and not pm]
    trainval_labels = [g.y.item() for g in trainval_graphs]

    train_graphs, val_graphs = train_test_split(
        trainval_graphs, test_size=0.15, random_state=42, stratify=trainval_labels
    )
    print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}, Probe: {len(probe_graphs)}")

    train_labels = np.array([g.y.item() for g in train_graphs])
    pos_weight = (train_labels == 0).sum() / max((train_labels == 1).sum(), 1)
    print(f"pos_weight for class imbalance: {pos_weight:.3f}")

    all_metrics = []
    best_overall_auc, best_overall_state = -1.0, None
    for seed in seeds:
        print(f"  --- seed {seed} ---")
        test_metrics, val_auc, state = train_one_seed(train_graphs, val_graphs, test_graphs, seed, epochs, device, pos_weight)
        print(f"  seed {seed} test metrics: {test_metrics}")
        all_metrics.append(test_metrics)
        if val_auc > best_overall_auc:
            best_overall_auc, best_overall_state = val_auc, state

    keys = all_metrics[0].keys()
    summary = {k: {"mean": float(np.mean([m[k] for m in all_metrics])), "std": float(np.std([m[k] for m in all_metrics]))} for k in keys}

    probe_result = None
    if probe_graphs:
        model = MoleculeGNN(ATOM_FEATURE_DIM, BOND_FEATURE_DIM).to(device)
        model.load_state_dict(best_overall_state)
        probe_loader = DataLoader(probe_graphs, batch_size=64, shuffle=False)
        probe_result = evaluate(model, probe_loader, device)
        print(f"Organochlorine probe set result: {probe_result}")

    # Save the best-val-AUC seed's weights for downstream inference (e.g. the
    # organochlorine concordance check) -- a representative model, not an ensemble.
    torch.save(best_overall_state, f"app/models/gnn_{name.lower()}.pt")
    print(f"Saved app/models/gnn_{name.lower()}.pt (best val AUC across seeds: {best_overall_auc:.4f})")

    return summary, all_metrics, probe_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["ames", "cpdb"], required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    with open("data/processed/organochlorine_probe_set.json") as f:
        probe_cas = set(json.load(f)["cas_numbers"])

    if args.task == "ames":
        ames = pd.read_csv("data/raw/ames_hansen2009/Mutagenicity_N6512.csv").dropna(subset=["Canonical_Smiles", "Activity"]).reset_index(drop=True)
        with open("data/raw/ames_hansen2009/splits_test_N6512.csv") as f:
            test_idx = set(int(i) for i in f.readline().strip().split(","))
        test_mask = np.array([i in test_idx for i in range(len(ames))])
        probe_mask = np.isin(ames["CAS_NO"].astype(str).str.strip().values, list(probe_cas))
        summary, all_metrics, probe_result = run_task(
            "Ames", ames["Canonical_Smiles"].tolist(), ames["Activity"].tolist(), test_mask, args.epochs, args.seeds, probe_mask
        )
    else:
        cpdb = pd.read_csv("data/raw/cpdb/carcinogenicity_final.csv").dropna(subset=["smiles", "carcinogenicity_label"]).reset_index(drop=True)
        from sklearn.model_selection import train_test_split as tts
        idx = np.arange(len(cpdb))
        trainval_idx, test_idx = tts(idx, test_size=0.15, random_state=42, stratify=cpdb["carcinogenicity_label"])
        test_mask = np.zeros(len(cpdb), dtype=bool)
        test_mask[test_idx] = True
        probe_mask = np.isin(cpdb["cas"].astype(str).str.strip().values, list(probe_cas))
        summary, all_metrics, probe_result = run_task(
            "CPDB", cpdb["smiles"].tolist(), cpdb["carcinogenicity_label"].tolist(), test_mask, args.epochs, args.seeds, probe_mask
        )

    print(f"\n=== {args.task} summary across {len(args.seeds)} seeds ===")
    print(json.dumps(summary, indent=2))
    print(f"Organochlorine probe set result: {json.dumps(probe_result, indent=2) if probe_result else 'no probe compounds in this dataset'}")

    out_path = Path(f"data/processed/gnn_{args.task}_results.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_seed": all_metrics, "probe_result": probe_result, "seeds": args.seeds, "epochs": args.epochs}, f, indent=2)
    print(f"Saved {out_path}")
