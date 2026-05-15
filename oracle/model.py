"""
IC50 oracle: fuses protein (ESM2 cross-attention), chemical (ChemBERTa
projection), and protein-family embeddings to predict normalised pIC50.

Architecture:
    protein_residues [B, L, 1280]  ──┐
                                      ├─ cross-attention ─> [B, 256]   ─┐
    chem_emb         [B, 768]      ──┘                                    ├─ MLP ─> pIC50 [B]
    family_idx       [B]     ──── Embedding(6, 16) ──> [B, 16]       ──┘

The model predicts in normalised pIC50 space.  Callers must denormalise
using per-protein (mean, std) stats stored in the checkpoint before computing
Spearman / MAE metrics or using the oracle as a GFlowNet reward.

Checkpoint format (saved by train.py):
    {
        'model_state_dict': model.state_dict(),
        'config':           config_dict,
        'epoch':            epoch,
        'pic50_norms':      {protein: (mean, std), '__global__': (mean, std)},
        ...
    }

Loading in GFlowNet:
    ckpt  = torch.load(path)
    model = IC50Oracle.from_config(ckpt['config'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from oracle.chem_encoder import ChemEncoder
from oracle.cross_attention import CrossAttentionPooling


class IC50Oracle(nn.Module):

    def __init__(
        self,
        cross_attention: CrossAttentionPooling,
        chem_encoder: ChemEncoder,
        family_embedding: nn.Embedding,
        fusion_hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cross_attention  = cross_attention
        self.chem_encoder     = chem_encoder
        self.family_embedding = family_embedding

        in_dim = (
            cross_attention.output_dim
            + chem_encoder.output_dim
            + family_embedding.embedding_dim
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(in_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim // 2, 1),
        )

    def forward(
        self,
        protein_residues: torch.Tensor,  # [B, L, esm_dim]  float16
        protein_mask: torch.Tensor,       # [B, L]            bool
        chem_emb: torch.Tensor,           # [B, 768]          float32
        family_idx: torch.Tensor,         # [B]               long
    ) -> torch.Tensor:                    # [B]               float32, normalised pIC50
        l = self.chem_encoder(chem_emb)                         # [B, proj_dim]
        p = self.cross_attention(protein_residues, protein_mask, l)  # [B, attn_dim]
        f = self.family_embedding(family_idx).to(l.dtype)       # [B, embed_dim]
        fused = torch.cat([p, l, f], dim=-1)
        return self.fusion_mlp(fused).squeeze(-1)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "IC50Oracle":
        """
        Reconstruct model from a config dict (e.g., loaded from a checkpoint).

        Expected config keys (matching config/default.yaml):
            esm.esm_dim, esm.attn_dim
            chem_encoder.chem_dim, chem_encoder.proj_dim, chem_encoder.dropout
            protein_family.n_families, protein_family.embed_dim
            model.fusion_hidden_dim, model.dropout
        """
        esm_cfg  = config.get("esm", {})
        chem_cfg = config.get("chem_encoder", {})
        fam_cfg  = config.get("protein_family", {})
        mdl_cfg  = config.get("model", {})

        chem_encoder = ChemEncoder(
            chem_dim = chem_cfg.get("chem_dim", 768),
            proj_dim = chem_cfg.get("proj_dim", 256),
            dropout  = chem_cfg.get("dropout",  0.1),
        )
        cross_attention = CrossAttentionPooling(
            esm_dim    = esm_cfg.get("esm_dim", 1280),
            ligand_dim = chem_encoder.output_dim,
            attn_dim   = esm_cfg.get("attn_dim", 256),
            dropout    = mdl_cfg.get("dropout", 0.1),
        )
        family_embedding = nn.Embedding(
            fam_cfg.get("n_families", 6),
            fam_cfg.get("embed_dim",  16),
        )
        return cls(
            cross_attention   = cross_attention,
            chem_encoder      = chem_encoder,
            family_embedding  = family_embedding,
            fusion_hidden_dim = mdl_cfg.get("fusion_hidden_dim", 512),
            dropout           = mdl_cfg.get("dropout", 0.1),
        )
