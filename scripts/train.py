"""
Training entry point for the SAIR IC50 oracle.

Usage:
    python scripts/train.py --config config/default.yaml

Key design decisions:
  - Model predicts normalised pIC50 (per-protein zero-mean / unit-std).
    All reported metrics (Spearman, MAE) are in raw pIC50 space — validate()
    denormalises predictions before metric computation.
  - Ranking loss compares within-protein pairs using normalised targets;
    ordering is preserved by the monotone normalisation, so ranking loss still
    measures the same thing as in raw space.
  - protein_ids for ranking loss: collate_fn assigns batch-local integer IDs
    from protein name strings.
  - Scheduler: CosineAnnealingLR stepping per batch (T_max = total_steps).
  - Mixed precision: torch.autocast — MPS without explicit dtype, CUDA float16.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy.stats import pearsonr, spearmanr
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from oracle.dataset import SAIRDataset
from oracle.loss import combined_loss
from oracle.model import IC50Oracle


# ── Collate ────────────────────────────────────────────────────────────────────

def collate_fn(batch: list) -> tuple:
    """
    Collates (residues, chem_emb, family_idx, pic50_norm, pic50_raw, protein_name).

    Residue tensors are padded via pad_sequence; mask is True for valid residues.
    Returns:
        padded      [B, L_max, D]  float16
        prot_mask   [B, L_max]     bool
        chem_embs   [B, 768]       float32
        family_idxs [B]            long
        pic50s_norm [B]            float32  (normalised, for loss)
        pic50s_raw  [B]            float32  (raw pIC50, for metrics)
        protein_ids [B]            long     (batch-local IDs, for ranking loss)
        protein_names tuple[str]            (for per-protein metric grouping)
    """
    residues_list, chem_embs, family_idxs, pic50s_norm, pic50s_raw, protein_names = zip(*batch)

    padded = torch.nn.utils.rnn.pad_sequence(
        list(residues_list), batch_first=True
    )  # [B, L_max, D] float16

    lengths   = torch.tensor([r.shape[0] for r in residues_list])
    prot_mask = torch.arange(padded.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)

    chem_embs   = torch.stack(chem_embs)                          # [B, 768]
    family_idxs = torch.tensor(family_idxs, dtype=torch.long)    # [B]
    pic50s_norm = torch.stack(pic50s_norm)                        # [B]
    pic50s_raw  = torch.stack(pic50s_raw)                         # [B]

    unique_names = {name: idx for idx, name in enumerate(dict.fromkeys(protein_names))}
    protein_ids  = torch.tensor(
        [unique_names[n] for n in protein_names], dtype=torch.long
    )

    return padded, prot_mask, chem_embs, family_idxs, pic50s_norm, pic50s_raw, protein_ids, protein_names


# ── Config helpers ─────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_entry_ids(splits_dir: Path, split: str) -> list[str]:
    txt = splits_dir / f"{split}.txt"
    return [line.strip() for line in txt.read_text().splitlines() if line.strip()]


def get_autocast_kwargs(device: torch.device) -> dict:
    if device.type == "cuda":
        return {"device_type": "cuda", "dtype": torch.float16}
    if device.type == "mps":
        return {"device_type": "mps"}
    return {"device_type": "cpu", "dtype": torch.bfloat16}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Validation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model: IC50Oracle,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    autocast_kwargs: dict,
    pic50_norms: dict[str, tuple[float, float]],
) -> dict:
    """
    Evaluate on val_loader.  Predictions are in normalised space; this function
    denormalises them using pic50_norms before computing all metrics.
    """
    model.eval()
    all_preds_norm, all_targets_raw, all_names = [], [], []

    for residues, prot_mask, chem_embs, family_idxs, _, pic50s_raw, _, protein_names in val_loader:
        residues    = residues.to(device)
        prot_mask   = prot_mask.to(device)
        chem_embs   = chem_embs.to(device)
        family_idxs = family_idxs.to(device)

        if use_amp:
            with torch.autocast(**autocast_kwargs):
                preds_norm = model(residues, prot_mask, chem_embs, family_idxs)
        else:
            preds_norm = model(residues, prot_mask, chem_embs, family_idxs)

        all_preds_norm.append(preds_norm.float().cpu())
        all_targets_raw.append(pic50s_raw.float().cpu())
        all_names.extend(protein_names)

    preds_norm_np  = torch.cat(all_preds_norm).numpy()
    targets_raw_np = torch.cat(all_targets_raw).numpy()

    # Denormalise predictions into raw pIC50 space
    _fallback = pic50_norms.get("__global__", (0.0, 1.0))
    means = np.array([pic50_norms.get(n, _fallback)[0] for n in all_names])
    stds  = np.array([pic50_norms.get(n, _fallback)[1] for n in all_names])
    preds_raw_np = preds_norm_np * stds + means

    # Global metrics in raw pIC50 space
    spearman = spearmanr(preds_raw_np, targets_raw_np).statistic
    pearson  = pearsonr(preds_raw_np, targets_raw_np)[0]
    mae      = float(abs(preds_raw_np - targets_raw_np).mean())

    # Per-protein Spearman (primary deployment metric)
    prot_preds:   dict = defaultdict(list)
    prot_targets: dict = defaultdict(list)
    for name, p, t in zip(all_names, preds_raw_np, targets_raw_np):
        prot_preds[name].append(p)
        prot_targets[name].append(t)

    per_protein_spearmans = []
    for name in prot_preds:
        if len(prot_preds[name]) >= 5:
            s = spearmanr(prot_preds[name], prot_targets[name]).statistic
            per_protein_spearmans.append(s)

    valid = np.array(per_protein_spearmans, dtype=float)
    mean_per_protein_spearman = float(np.nanmean(valid)) if len(valid) > 0 else float("nan")

    return {
        "spearman":             float(spearman),
        "pearson":              float(pearson),
        "mae":                  mae,
        "per_protein_spearman": mean_per_protein_spearman,
    }


# ── Training loop ──────────────────────────────────────────────────────────────

def train(config: dict, resume: str | None = None) -> None:
    paths     = config["paths"]
    data_cfg  = config["data"]
    train_cfg = config["training"]
    log_cfg   = config["logging"]
    esm_cfg   = config.get("esm", {})

    device = get_device()
    print(f"Device: {device}")

    use_amp = train_cfg.get("mixed_precision", False)
    autocast_kwargs = get_autocast_kwargs(device)
    if use_amp:
        print(f"Mixed precision: torch.autocast({autocast_kwargs})")

    # ── Splits ────────────────────────────────────────────────────────────────
    splits_dir = Path(paths["splits"])
    train_ids  = load_entry_ids(splits_dir, "train")
    val_ids    = load_entry_ids(splits_dir, "val")
    print(f"Split sizes — train: {len(train_ids):,}  val: {len(val_ids):,}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    dataset_kwargs = dict(
        parquet_path        = paths["parquet"],
        esm_cache_dir       = paths["esm_cache"],
        annotations_csv     = paths["annotations"],
        filter_all_passed   = data_cfg["filter_all_passed"],
        assay_filter        = data_cfg.get("assay_filter"),
        deduplicate         = data_cfg["deduplicate"],
        residue_cache_dir   = paths.get("residue_cache"),
        max_residues        = esm_cfg.get("max_residues", 1022),
        chem_emb_cache_path = paths.get("chem_emb_cache"),
        normalize_targets   = train_cfg.get("normalize_targets", True),
    )

    # Each split computes its own per-protein pIC50 normalization stats.
    # (Protein-cluster splits mean train ≠ val proteins, so train norms
    # can't be applied to val proteins directly.)
    train_ds = SAIRDataset(entry_ids=train_ids, **dataset_kwargs)
    val_ds   = SAIRDataset(entry_ids=val_ids,   **dataset_kwargs)

    train_loader = DataLoader(
        train_ds,
        batch_size  = train_cfg["batch_size"],
        shuffle     = True,
        num_workers = 0,
        collate_fn  = collate_fn,
        pin_memory  = (device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = train_cfg["batch_size"],
        shuffle     = False,
        num_workers = 0,
        collate_fn  = collate_fn,
        pin_memory  = (device.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = IC50Oracle.from_config(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = AdamW(
        model.parameters(),
        lr           = train_cfg["learning_rate"],
        weight_decay = train_cfg["weight_decay"],
    )

    total_steps = train_cfg["epochs"] * len(train_loader)
    scheduler   = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    grad_clip = train_cfg.get("grad_clip", 0.0)

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch = 1
    best_spearman = -float("inf")
    epochs_without_improvement = 0
    global_step = 0

    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_spearman = ckpt.get("best_spearman", ckpt.get("val_spearman", -float("inf")))
        epochs_without_improvement = ckpt.get("epochs_without_improvement", 0)
        global_step = ckpt.get("global_step", ckpt["epoch"] * len(train_loader))

        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        else:
            for _ in range(global_step):
                scheduler.step()
            print(f"  Scheduler fast-forwarded to step {global_step}")

        print(f"Resumed from {resume} — epoch {ckpt['epoch']}  "
              f"val_spearman={ckpt.get('val_spearman', '?'):.4f}  "
              f"continuing from epoch {start_epoch}")

    # ── Wandb (optional) ──────────────────────────────────────────────────────
    use_wandb = False
    if log_cfg.get("wandb_project"):
        try:
            import wandb
            wandb.init(
                project = log_cfg["wandb_project"],
                entity  = log_cfg.get("wandb_entity"),
                config  = config,
            )
            use_wandb = True
        except Exception as e:
            print(f"wandb init failed ({e}), continuing without logging")

    # ── Checkpoints ───────────────────────────────────────────────────────────
    ckpt_dir = Path(paths["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    patience  = train_cfg["patience"]
    log_every = log_cfg.get("log_every_n_steps", 50)

    # ── Main loop ─────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, train_cfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for residues, prot_mask, chem_embs, family_idxs, pic50s_norm, _, protein_ids, _ in pbar:
            residues    = residues.to(device)
            prot_mask   = prot_mask.to(device)
            chem_embs   = chem_embs.to(device)
            family_idxs = family_idxs.to(device)
            pic50s_norm = pic50s_norm.to(device)
            protein_ids = protein_ids.to(device)

            optimizer.zero_grad()

            loss_kwargs = dict(
                huber_weight   = train_cfg.get("huber_weight", 1.0),
                ranking_weight = train_cfg.get("ranking_weight", 0.1),
                huber_delta    = train_cfg.get("huber_delta", 1.0),
                ranking_margin = train_cfg.get("ranking_margin", 0.5),
            )
            if use_amp:
                with torch.autocast(**autocast_kwargs):
                    preds = model(residues, prot_mask, chem_embs, family_idxs)
                loss = combined_loss(preds.float(), pic50s_norm.float(), protein_ids, **loss_kwargs)
            else:
                preds = model(residues, prot_mask, chem_embs, family_idxs)
                loss  = combined_loss(preds, pic50s_norm, protein_ids, **loss_kwargs)

            loss.backward()

            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            scheduler.step()

            epoch_loss  += loss.item()
            global_step += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            if device.type == "mps" and global_step % 10 == 0:
                torch.mps.empty_cache()

            if use_wandb and global_step % log_every == 0:
                import wandb
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr":   scheduler.get_last_lr()[0],
                }, step=global_step)

        avg_train_loss = epoch_loss / len(train_loader)

        # ── Validation ────────────────────────────────────────────────────────
        metrics = validate(model, val_loader, device, use_amp, autocast_kwargs,
                           val_ds.pic50_norms)
        if device.type == "mps":
            torch.mps.empty_cache()

        print(
            f"Epoch {epoch:3d} | train_loss={avg_train_loss:.4f} | "
            f"val_spearman={metrics['spearman']:.4f} | "
            f"val_mae={metrics['mae']:.4f} | "
            f"per_prot_spearman={metrics['per_protein_spearman']:.4f}"
        )

        if use_wandb:
            import wandb
            wandb.log({
                "train/epoch_loss":         avg_train_loss,
                "val/spearman":             metrics["spearman"],
                "val/pearson":              metrics["pearson"],
                "val/mae":                  metrics["mae"],
                "val/per_protein_spearman": metrics["per_protein_spearman"],
            }, step=global_step)

        # ── Checkpoint ────────────────────────────────────────────────────────
        monitor_key = train_cfg.get("checkpoint_monitor", "per_protein_spearman")
        monitor = metrics[monitor_key]
        if monitor > best_spearman:
            best_spearman = monitor
            epochs_without_improvement = 0
            ckpt_path = ckpt_dir / "best.pt"
            torch.save({
                "model_state_dict":           model.state_dict(),
                "optimizer_state_dict":       optimizer.state_dict(),
                "scheduler_state_dict":       scheduler.state_dict(),
                "config":                     config,
                "pic50_norms":                train_ds.pic50_norms,
                "epoch":                      epoch,
                "global_step":                global_step,
                "best_spearman":              best_spearman,
                "epochs_without_improvement": 0,
                "val_spearman":               metrics["spearman"],
                "val_mae":                    metrics["mae"],
            }, ckpt_path)
            print(f"  Checkpoint saved ({monitor_key}={best_spearman:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {patience} epochs without improvement.")
                break

        gc.collect()

    # Save final checkpoint
    torch.save({
        "model_state_dict":           model.state_dict(),
        "optimizer_state_dict":       optimizer.state_dict(),
        "scheduler_state_dict":       scheduler.state_dict(),
        "config":                     config,
        "pic50_norms":                train_ds.pic50_norms,
        "epoch":                      epoch,
        "global_step":                global_step,
        "best_spearman":              best_spearman,
        "epochs_without_improvement": epochs_without_improvement,
        "val_spearman":               metrics["spearman"],
        "val_mae":                    metrics["mae"],
    }, ckpt_dir / "last.pt")

    if use_wandb:
        import wandb
        wandb.finish()

    print(f"\nDone. Best per_prot_spearman: {best_spearman:.4f}")
    print(f"Checkpoints in: {ckpt_dir}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train the SAIR IC50 oracle.")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (e.g. checkpoints/baseline/best.pt)")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, resume=args.resume)


if __name__ == "__main__":
    main()
