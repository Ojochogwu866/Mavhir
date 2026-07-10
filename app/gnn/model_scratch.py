"""From-scratch message-passing GNN, built without PyTorch Geometric's
nn.MessagePassing base class or any of its conv layers (GINEConv, etc.) --
only basic PyTorch tensor ops (index_add_ for scatter-sum aggregation).

This exists specifically to demonstrate understanding of what a message-
passing layer actually does mechanically, as a cross-check against the
library implementation in model.py, per docs/gnn_extension_design.md
Section 3.2: get a working library-based comparison first, then implement
the mechanism by hand and confirm it produces comparable results.

Message passing, one layer, in words:
  1. For every directed edge (src -> dst), build a message from the source
     node's current representation and that edge's features.
  2. Sum all incoming messages at each destination node (scatter-sum).
  3. Combine each node's own previous representation with its aggregated
     incoming messages to produce its next representation.
Repeat for num_layers rounds, so information propagates that many hops
through the molecular graph, then pool node representations into one
graph-level vector for the classification head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScratchMessagePassingLayer(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        # Message function: source node state + edge features -> message vector.
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Update function: own previous state + aggregated incoming messages -> new state.
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]

        # Step 1: build one message per directed edge from its source node + edge features.
        messages = self.message_mlp(torch.cat([x[src], edge_attr], dim=-1))

        # Step 2: scatter-sum messages into their destination nodes. This is the
        # manual equivalent of what MessagePassing.propagate() does internally --
        # index_add_ adds messages[i] into aggregated[dst[i]] for every edge i,
        # so a node with three incoming edges sums all three messages.
        num_nodes = x.size(0)
        aggregated = torch.zeros(num_nodes, messages.size(-1), dtype=messages.dtype, device=x.device)
        aggregated.index_add_(0, dst, messages)

        # Step 3: combine each node's previous state with what it just received.
        return self.update_mlp(torch.cat([x, aggregated], dim=-1))


def scatter_mean_sum(x: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Manual graph-level pooling (mean and sum), replacing torch_geometric's
    global_mean_pool/global_add_pool with the same index_add_ mechanism used above."""
    hidden_dim = x.size(-1)
    sum_pool = torch.zeros(num_graphs, hidden_dim, dtype=x.dtype, device=x.device)
    sum_pool.index_add_(0, batch, x)

    counts = torch.zeros(num_graphs, dtype=x.dtype, device=x.device)
    counts.index_add_(0, batch, torch.ones(x.size(0), dtype=x.dtype, device=x.device))
    mean_pool = sum_pool / counts.clamp(min=1).unsqueeze(-1)

    return mean_pool, sum_pool


class ScratchMoleculeGNN(nn.Module):
    def __init__(self, atom_feature_dim: int, bond_feature_dim: int, hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.atom_embed = nn.Linear(atom_feature_dim, hidden_dim)

        self.layers = nn.ModuleList([
            ScratchMessagePassingLayer(hidden_dim, bond_feature_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.dropout = dropout

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2: concat mean + sum pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.atom_embed(x)

        for layer, norm in zip(self.layers, self.norms):
            h_new = layer(h, edge_index, edge_attr)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new  # residual connection, same as the library model

        num_graphs = int(batch.max().item()) + 1
        mean_pool, sum_pool = scatter_mean_sum(h, batch, num_graphs)
        pooled = torch.cat([mean_pool, sum_pool], dim=1)
        return self.readout(pooled).squeeze(-1)
