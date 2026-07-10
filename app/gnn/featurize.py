"""Convert SMILES strings to molecular graphs (PyTorch Geometric Data objects).

Atom (node) features, one-hot where noted:
  - atom type (one-hot over a fixed vocabulary of common elements + "other")
  - degree (0-5, one-hot)
  - formal charge (raw int)
  - hybridization (one-hot: SP, SP2, SP3, other)
  - aromaticity (binary)
  - implicit H count (0-4, one-hot)
  - in-ring (binary)

Bond (edge) features, one-hot where noted:
  - bond type (single/double/triple/aromatic, one-hot)
  - conjugated (binary)
  - in-ring (binary)
  - stereo configuration (one-hot: none/Z/E/other)

This is the standard atom/bond featurization used in most MPNN molecular property
work (e.g. Chemprop, DeepChem's MolGraphConvFeaturizer) -- reused here rather than
invented from scratch, per the design doc's recommendation to spend novelty on the
training/evaluation methodology, not on reinventing a published featurization.
"""
from __future__ import annotations

import torch
from rdkit import Chem
from torch_geometric.data import Data

ATOM_VOCAB = ["C", "N", "O", "S", "Cl", "F", "Br", "P", "I", "B", "Si"]  # + "other" bucket
HYBRIDIZATION_VOCAB = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
]  # + "other" bucket
BOND_TYPE_VOCAB = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
STEREO_VOCAB = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
]  # + "other" bucket


def _one_hot(value, vocab) -> list[float]:
    """One-hot encode `value` against `vocab`, with a trailing 'other' bucket for
    anything not in vocab (rather than silently erroring on unseen atom types)."""
    vec = [0.0] * (len(vocab) + 1)
    try:
        idx = vocab.index(value)
        vec[idx] = 1.0
    except ValueError:
        vec[-1] = 1.0
    return vec


def atom_features(atom: Chem.Atom) -> list[float]:
    feats = []
    feats += _one_hot(atom.GetSymbol(), ATOM_VOCAB)
    feats += _one_hot(min(atom.GetDegree(), 5), list(range(6)))
    feats += [float(atom.GetFormalCharge())]
    feats += _one_hot(atom.GetHybridization(), HYBRIDIZATION_VOCAB)
    feats += [1.0 if atom.GetIsAromatic() else 0.0]
    feats += _one_hot(min(atom.GetTotalNumHs(), 4), list(range(5)))
    feats += [1.0 if atom.IsInRing() else 0.0]
    return feats


def bond_features(bond: Chem.Bond) -> list[float]:
    feats = []
    feats += _one_hot(bond.GetBondType(), BOND_TYPE_VOCAB)
    feats += [1.0 if bond.GetIsConjugated() else 0.0]
    feats += [1.0 if bond.IsInRing() else 0.0]
    feats += _one_hot(bond.GetStereo(), STEREO_VOCAB)
    return feats


ATOM_FEATURE_DIM = len(_one_hot("C", ATOM_VOCAB)) + len(_one_hot(0, list(range(6)))) + 1 + \
    len(_one_hot(Chem.rdchem.HybridizationType.SP2, HYBRIDIZATION_VOCAB)) + 1 + \
    len(_one_hot(0, list(range(5)))) + 1
BOND_FEATURE_DIM = len(_one_hot(Chem.rdchem.BondType.SINGLE, BOND_TYPE_VOCAB)) + 1 + 1 + \
    len(_one_hot(Chem.rdchem.BondStereo.STEREONONE, STEREO_VOCAB))


def smiles_to_graph(smiles: str, label: float | None = None) -> Data | None:
    """Returns None if RDKit can't parse the SMILES (same ~0.1% failure rate seen
    throughout this project's descriptor computation -- a handful of unusual
    stereo-bond notations RDKit's parser rejects)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    node_feats = [atom_features(atom) for atom in mol.GetAtoms()]
    if not node_feats:
        return None
    x = torch.tensor(node_feats, dtype=torch.float)

    edge_indices, edge_feats = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_features(bond)
        edge_indices += [[i, j], [j, i]]  # undirected: both directions
        edge_feats += [feat, feat]

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_feats, dtype=torch.float)
    else:
        # single-atom molecules (e.g. some ions) have no bonds
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_FEATURE_DIM), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.float)
    return data
