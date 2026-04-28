"""
Protein encoder: processes a precomputed ESM2 mean-pooled embedding.

The ESM2 embedding is already [1280] and lives on disk. This module is an
optional lightweight MLP projection before fusion with the ligand embedding.
Default (project=False) is an identity pass-through.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ProteinEncoder(nn.Module):
    """
    Processes a precomputed ESM2 embedding.

    Args:
        esm_dim:    Dimensionality of the ESM2 embedding (1280 for 650M model).
        hidden_dim: Output dim when project=True.
        project:    If False (default), acts as identity; output_dim == esm_dim.
                    If True, applies Linear(esm_dim, hidden_dim) -> LayerNorm -> ReLU.
        dropout:    Dropout probability (only applied when project=True).
    """

    def __init__(
        self,
        esm_dim: int = 1280,
        hidden_dim: int = 256,
        project: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.esm_dim = esm_dim
        self.project = project
        self.output_dim = hidden_dim if project else esm_dim

        if project:
            self.mlp = nn.Sequential(
                nn.Linear(esm_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.mlp = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, esm_dim] precomputed ESM2 embeddings
        Returns:
            [batch, output_dim]
        """
        return self.mlp(x)
