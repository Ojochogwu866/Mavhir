"""D-MPNN-style graph neural network for molecular toxicity classification.

Library implementation (PyTorch Geometric's GINEConv, which supports edge features)
-- per the design doc, this is step one of two: get a working, evaluated comparison
against the RF/GBM baseline first, then implement a from-scratch simplified
message-passing layer afterward and confirm it produces comparable results, which
is the part that actually demonstrates architecture-level understanding rather than
"I called a library."
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool


class MoleculeGNN(nn.Module):
    def __init__(self, atom_feature_dim: int, bond_feature_dim: int, hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.atom_embed = nn.Linear(atom_feature_dim, hidden_dim)
        self.bond_embed = nn.Linear(bond_feature_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = dropout
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2: concat mean + sum pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.atom_embed(x)
        e = self.bond_embed(edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index, e)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new  # residual connection, helps with deeper stacks

        pooled = torch.cat([global_mean_pool(h, batch), global_add_pool(h, batch)], dim=1)
        return self.readout(pooled).squeeze(-1)  # raw logits, apply sigmoid outside
