"""
Full IC50 oracle: fuses protein (ESM2) and ligand (GNN) embeddings to predict pIC50.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from oracle.ligand_encoder import LigandEncoder
from oracle.protein_encoder import ProteinEncoder


class IC50Oracle(nn.Module):
    """
    Predicts pIC50 from a precomputed ESM2 protein embedding and a
    2D molecular graph.

    Architecture:
        protein_emb [protein_dim]  ──┐
                                      ├── concat ──> MLP ──> pIC50 [1]
        ligand_emb  [ligand_dim]   ──┘

    Fusion MLP:
        Linear(protein_dim + ligand_dim, fusion_hidden_dim) -> ReLU -> Dropout
        Linear(fusion_hidden_dim, fusion_hidden_dim // 2)   -> ReLU -> Dropout
        Linear(fusion_hidden_dim // 2, 1)

    The model predicts in pIC50 space (typically 4-12). No output clamping;
    loss and metrics handle the range.

    Checkpoint format (saved by train.py):
        {
            'model_state_dict': model.state_dict(),
            'config': config_dict,
            'epoch': epoch,
            'val_loss': val_loss,
        }

    Loading in GFlowNet:
        ckpt = torch.load(path)
        model = IC50Oracle.from_config(ckpt['config'])
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
    """

    def __init__(
        self,
        protein_encoder: ProteinEncoder,
        ligand_encoder: LigandEncoder,
        fusion_hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.protein_encoder = protein_encoder
        self.ligand_encoder = ligand_encoder

        protein_dim = protein_encoder.output_dim
        ligand_dim = ligand_encoder.hidden_dim

        self.fusion_mlp = nn.Sequential(
            nn.Linear(protein_dim + ligand_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim // 2, 1),
        )

    def forward(
        self,
        protein_emb: torch.Tensor,
        ligand_batch: Batch,
    ) -> torch.Tensor:
        """
        Args:
            protein_emb:  [batch, esm_dim] precomputed ESM2 embeddings
            ligand_batch: PyG Batch of molecular graphs

        Returns:
            Tensor [batch] of predicted pIC50 values
        """
        p = self.protein_encoder(protein_emb)          # [B, protein_dim]
        l = self.ligand_encoder(ligand_batch)           # [B, ligand_dim]
        fused = torch.cat([p, l], dim=-1)               # [B, protein_dim + ligand_dim]
        return self.fusion_mlp(fused).squeeze(-1)       # [B]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "IC50Oracle":
        """
        Reconstruct model from a config dict (e.g., loaded from a checkpoint).

        Expected config keys (matching config/default.yaml structure):
            esm.esm_dim, esm.hidden_dim, esm.project, esm.dropout  (optional dropout)
            ligand_gnn.node_dim, ligand_gnn.edge_dim, ligand_gnn.hidden_dim,
            ligand_gnn.n_layers, ligand_gnn.dropout
            model.fusion_hidden_dim, model.dropout
        """
        esm_cfg = config.get("esm", {})
        gnn_cfg = config.get("ligand_gnn", {})
        model_cfg = config.get("model", {})

        protein_encoder = ProteinEncoder(
            esm_dim=esm_cfg.get("esm_dim", 1280),
            hidden_dim=esm_cfg.get("hidden_dim", 256),
            project=esm_cfg.get("project", False),
            dropout=esm_cfg.get("dropout", 0.1),
        )
        ligand_encoder = LigandEncoder(
            node_dim=gnn_cfg.get("node_dim", 14),
            edge_dim=gnn_cfg.get("edge_dim", 3),
            hidden_dim=gnn_cfg.get("hidden_dim", 256),
            n_layers=gnn_cfg.get("n_layers", 4),
            dropout=gnn_cfg.get("dropout", 0.1),
        )
        return cls(
            protein_encoder=protein_encoder,
            ligand_encoder=ligand_encoder,
            fusion_hidden_dim=model_cfg.get("fusion_hidden_dim", 512),
            dropout=model_cfg.get("dropout", 0.1),
        )
