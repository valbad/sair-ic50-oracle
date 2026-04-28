"""
Loss functions for the SAIR IC50 oracle.

Primary: Huber loss (robust to pIC50 outliers).
Optional: pairwise ranking loss within same-protein pairs (vectorised, O(B²) in ops
but avoids Python loops via broadcasting).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def huber_loss(pred: Tensor, target: Tensor, delta: float = 1.0) -> Tensor:
    """Huber loss, more robust to pIC50 outliers than MSE."""
    return F.huber_loss(pred, target, delta=delta)


def ranking_loss(
    pred: Tensor,
    target: Tensor,
    protein_ids: Tensor,
    margin: float = 0.5,
) -> Tensor:
    """
    Vectorised pairwise ranking loss within same-protein pairs.

    For all pairs (i, j) where protein_id[i] == protein_id[j] and i != j:
        loss += max(0, margin - sign(target_i - target_j) * (pred_i - pred_j))

    Teaches the model to correctly rank compounds within a protein, which is
    more important for GFlowNet reward quality than absolute pIC50 accuracy.

    Args:
        pred:        [B] predicted pIC50
        target:      [B] true pIC50
        protein_ids: [B] integer protein index per sample
        margin:      minimum ranking gap to enforce

    Returns:
        Scalar loss tensor. Returns 0 if no valid same-protein pairs exist.
    """
    same_protein = protein_ids.unsqueeze(0) == protein_ids.unsqueeze(1)  # [B, B]
    not_diagonal = ~torch.eye(len(pred), dtype=torch.bool, device=pred.device)
    mask = same_protein & not_diagonal                                    # [B, B]

    if not mask.any():
        return pred.sum() * 0.0  # differentiable zero

    diff_pred = pred.unsqueeze(0) - pred.unsqueeze(1)       # [B, B]
    diff_target = target.unsqueeze(0) - target.unsqueeze(1) # [B, B]
    signs = torch.sign(diff_target)
    losses = torch.clamp(margin - signs * diff_pred, min=0.0)
    return losses[mask].mean()


def combined_loss(
    pred: Tensor,
    target: Tensor,
    protein_ids: Tensor,
    huber_weight: float = 1.0,
    ranking_weight: float = 0.1,
    huber_delta: float = 1.0,
    ranking_margin: float = 0.5,
) -> Tensor:
    """Weighted sum of Huber loss and ranking loss."""
    h = huber_loss(pred, target, delta=huber_delta)
    r = ranking_loss(pred, target, protein_ids, margin=ranking_margin)
    return huber_weight * h + ranking_weight * r
