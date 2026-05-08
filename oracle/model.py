"""
Full IC50 oracle: fuses protein (ESM2 per-residue cross-attention) and
ligand (GNN) embeddings to predict pIC50.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from oracle.cross_attention import CrossAttentionPooling
from oracle.ligand_encoder import LigandEncoder


class IC50Oracle(nn.Module):
    """
    Predicts pIC50 from per-residue ESM2 protein representations and a
    2D molecular graph.

    Architecture:
        protein_residues [B, L, esm_dim]  ──┐
                                              ├─ cross-attention ─> [B, attn_dim]
        ligand_emb       [B, ligand_dim]  ──┘         │
                                                       ↓
                         concat([attn_dim + ligand_dim]) ─> MLP ─> pIC50 [B]

    The ligand embedding is the attention query: the model learns which
    residues matter for each ligand rather than using a static mean-pool.

    Checkpoint format (saved by train.py):
        {
            'model_state_dict': model.state_dict(),
            'config': config_dict,
            'epoch': epoch,
            'pic50_norms': {protein: (mean, std), '__global__': (mean, std)},
        }

    Loading in GFlowNet:
        ckpt = torch.load(path)
        model = IC50Oracle.from_config(ckpt['config'])
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
    """

    def __init__(
        self,
        cross_attention: CrossAttentionPooling,
        ligand_encoder: LigandEncoder,
        fusion_hidden_dim: int = 512,
        dropout: float = 0.1,
        descriptor_dim: int = 0,
    ):
        super().__init__()
        self.cross_attention = cross_attention
        self.ligand_encoder  = ligand_encoder
        self.descriptor_dim  = descriptor_dim

        in_dim = cross_attention.output_dim + ligand_encoder.hidden_dim + descriptor_dim
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
        protein_residues: torch.Tensor,        # [B, L, esm_dim]
        protein_mask: torch.Tensor,             # [B, L] bool — True = valid residue
        ligand_batch: Batch,
        descriptors: torch.Tensor | None = None,  # [B, descriptor_dim]
    ) -> torch.Tensor:                          # [B]
        l = self.ligand_encoder(ligand_batch)                        # [B, D_l]
        p = self.cross_attention(protein_residues, protein_mask, l)  # [B, D_attn]
        parts = [p, l]
        if descriptors is not None and self.descriptor_dim > 0:
            parts.append(descriptors.to(l.dtype))
        fused = torch.cat(parts, dim=-1)
        return self.fusion_mlp(fused).squeeze(-1)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "IC50Oracle":
        """
        Reconstruct model from a config dict (e.g., loaded from a checkpoint).

        Expected config keys (matching config/default.yaml structure):
            esm.esm_dim, esm.attn_dim
            ligand_gnn.node_dim, ligand_gnn.edge_dim, ligand_gnn.hidden_dim,
            ligand_gnn.n_layers, ligand_gnn.dropout
            model.fusion_hidden_dim, model.dropout
        """
        esm_cfg   = config.get("esm", {})
        gnn_cfg   = config.get("ligand_gnn", {})
        model_cfg = config.get("model", {})

        ligand_encoder = LigandEncoder(
            node_dim  = gnn_cfg.get("node_dim", 14),
            edge_dim  = gnn_cfg.get("edge_dim", 3),
            hidden_dim= gnn_cfg.get("hidden_dim", 256),
            n_layers  = gnn_cfg.get("n_layers", 4),
            dropout   = gnn_cfg.get("dropout", 0.1),
        )
        cross_attention = CrossAttentionPooling(
            esm_dim   = esm_cfg.get("esm_dim", 1280),
            ligand_dim= ligand_encoder.hidden_dim,
            attn_dim  = esm_cfg.get("attn_dim", 256),
            dropout   = model_cfg.get("dropout", 0.1),
        )
        return cls(
            cross_attention  = cross_attention,
            ligand_encoder   = ligand_encoder,
            fusion_hidden_dim= model_cfg.get("fusion_hidden_dim", 512),
            dropout          = model_cfg.get("dropout", 0.1),
            descriptor_dim   = gnn_cfg.get("descriptor_dim", 0),
        )
