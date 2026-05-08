"""
Ligand-guided cross-attention pooling over per-residue ESM2 representations.

Replaces the static mean-pooled ProteinEncoder with a dynamic attention step:
the ligand graph embedding acts as a query to identify which protein residues
are most relevant for binding, producing a ligand-conditioned protein context.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionPooling(nn.Module):
    """
    Pools per-residue ESM2 representations [B, L, esm_dim] into a single
    context vector [B, attn_dim] conditioned on the ligand embedding.

    Single-head scaled dot-product attention:
        Q = linear(ligand_emb)               [B, attn_dim]
        K = linear(protein_residues)         [B, L, attn_dim]
        V = linear(protein_residues)         [B, L, attn_dim]
        scores = Q·Kᵀ / sqrt(attn_dim)      [B, L]  (padding masked to -inf)
        context = softmax(scores) @ V        [B, attn_dim]

    Args:
        esm_dim:    Dimensionality of per-residue ESM2 representations (1280).
        ligand_dim: Dimensionality of the ligand graph embedding.
        attn_dim:   Shared projection dimension for keys, queries, and values.
        dropout:    Dropout applied to attention weights.
    """

    def __init__(
        self,
        esm_dim: int = 1280,
        ligand_dim: int = 256,
        attn_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn_dim = attn_dim
        self.scale = attn_dim ** -0.5

        self.query_proj = nn.Linear(ligand_dim, attn_dim, bias=False)
        self.key_proj   = nn.Linear(esm_dim, attn_dim, bias=False)
        self.value_proj = nn.Linear(esm_dim, attn_dim, bias=False)
        self.out_norm   = nn.LayerNorm(attn_dim)
        self.dropout    = nn.Dropout(dropout)

    @property
    def output_dim(self) -> int:
        return self.attn_dim

    def forward(
        self,
        protein_residues: torch.Tensor,  # [B, L, esm_dim] float16 from collate
        protein_mask: torch.Tensor,       # [B, L] bool — True = valid residue
        ligand_emb: torch.Tensor,         # [B, ligand_dim]
    ) -> torch.Tensor:                    # [B, attn_dim]
        # Cast on GPU — avoids the large CPU-side float16→float32 conversion
        # that was the collate bottleneck in the old implementation.
        protein_residues = protein_residues.to(ligand_emb.dtype)

        query  = self.query_proj(ligand_emb)        # [B, D]
        keys   = self.key_proj(protein_residues)    # [B, L, D]
        values = self.value_proj(protein_residues)  # [B, L, D]

        scores  = torch.einsum("bd,bld->bl", query, keys) * self.scale  # [B, L]
        scores  = scores.masked_fill(~protein_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)   # [B, L]
        weights = self.dropout(weights)

        context = torch.einsum("bl,bld->bd", weights, values)  # [B, D]
        return self.out_norm(context)
