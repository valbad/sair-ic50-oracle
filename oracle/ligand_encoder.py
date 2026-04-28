"""
Ligand encoder: GNN over the 2D molecular graph using GINEConv.

GINEConv natively supports edge features and is a strong default for
molecular property prediction. Architecture:
    input projection -> N x (GINEConv + BatchNorm + ReLU + Dropout) -> global_mean_pool
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv, global_mean_pool

from oracle.featurise import ATOM_FEATURE_DIM, BOND_FEATURE_DIM


def _make_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    """Two-layer MLP used inside each GINEConv."""
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.ReLU(),
        nn.Linear(out_dim, out_dim),
    )


class LigandEncoder(nn.Module):
    """
    GNN over the 2D molecular graph.

    Args:
        node_dim:   Input atom feature dimension (ATOM_FEATURE_DIM = 14).
        edge_dim:   Input bond feature dimension (BOND_FEATURE_DIM = 3).
        hidden_dim: Hidden and output dimension.
        n_layers:   Number of GINEConv message-passing layers.
        dropout:    Dropout probability applied before the output.

    Forward:
        graph: PyG Batch of molecules (from DataLoader with custom collate_fn)
        Returns: Tensor [batch_size, hidden_dim]
    """

    def __init__(
        self,
        node_dim: int = ATOM_FEATURE_DIM,
        edge_dim: int = BOND_FEATURE_DIM,
        hidden_dim: int = 256,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Linear input projection: map raw atom features to hidden_dim
        self.input_proj = nn.Linear(node_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(
                GINEConv(
                    nn=_make_mlp(hidden_dim, hidden_dim),
                    edge_dim=edge_dim,
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(self, graph: Batch) -> torch.Tensor:
        """
        Args:
            graph: PyG Batch with x [N_total_atoms, node_dim],
                   edge_index [2, E], edge_attr [E, edge_dim], batch [N]
        Returns:
            Tensor [batch_size, hidden_dim]
        """
        x = self.input_proj(graph.x)

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, graph.edge_index, graph.edge_attr)
            x = bn(x)
            x = torch.relu(x)

        x = self.dropout(x)
        return global_mean_pool(x, graph.batch)
