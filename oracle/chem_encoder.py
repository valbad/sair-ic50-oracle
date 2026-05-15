"""
ChemBERTa chemical encoder projection head.

The ChemBERTa backbone is frozen and run offline by precompute_ligands.py,
which stores CLS tokens as a dict[smiles -> Tensor[768]].  At training time
this module is just the learned Linear(768 -> proj_dim) projection.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ChemEncoder(nn.Module):
    """Projects a precomputed ChemBERTa CLS embedding to the fusion dimension."""

    def __init__(self, chem_dim: int = 768, proj_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(chem_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output_dim = proj_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, chem_dim] -> [B, proj_dim]
        return self.proj(x)
