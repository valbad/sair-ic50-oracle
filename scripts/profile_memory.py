"""
Memory audit for the training pipeline.

Runs a forward+backward pass over 600 steps (half an epoch at batch_size=128)
and records memory at every step across four independent channels:
    1. Process RSS       — total Python process resident memory (CPU side)
    2. MPS allocated     — bytes currently live in MPS tensor buffers
    3. MPS driver total  — bytes held by the Metal driver (incl. fragmentation)
    4. Python gc objects — count of live Python objects

Prints a per-50-step table and a summary of what grew.

Usage (from repo root):
    python scripts/profile_memory.py
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import psutil
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from oracle.dataset import SAIRDataset
from oracle.loss import combined_loss
from oracle.model import IC50Oracle
from scripts.train import collate_fn
from torch.utils.data import DataLoader
from torch.optim import AdamW

PROBE_STEPS = 600   # how many steps to run
PRINT_EVERY = 50    # print a row every N steps

# ── Diagnostic mode ───────────────────────────────────────────────────────────
# Set DATA_ONLY=True to skip forward/backward and isolate whether the RSS leak
# is in the data-loading/collation path or in the model training step.
DATA_ONLY = False

# How often to call torch.mps.empty_cache() inside the training loop.
# 0 = never (original behaviour); 1 = every step; N = every N steps.
# In data-only mode this has no effect.
EMPTY_CACHE_EVERY = 10

def mb(n: int) -> float:
    return n / 1e6

def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1e6

def mps_alloc_mb() -> float:
    return torch.mps.current_allocated_memory() / 1e6

def mps_driver_mb() -> float:
    return torch.mps.driver_allocated_memory() / 1e6

def gc_obj_count() -> int:
    return len(gc.get_objects())


def main():
    device = torch.device("mps")

    print("Loading config and dataset (this may take ~30s)...")
    with open("config/default.yaml") as f:
        config = yaml.safe_load(f)
    config["training"]["batch_size"] = 128

    entry_ids = open("data/splits/train.txt").read().splitlines()

    ds = SAIRDataset(
        parquet_path="data/sair.parquet",
        esm_cache_dir="data/esm_embeddings/",
        annotations_csv="data/long_protein_annotations.csv",
        entry_ids=entry_ids,
        filter_all_passed=True,
        assay_filter="biochem",
        deduplicate=True,
    )

    loader = DataLoader(
        ds, batch_size=128, shuffle=True, num_workers=0, collate_fn=collate_fn
    )

    model = IC50Oracle.from_config(config).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    model.train()

    print(f"Mode: {'DATA_ONLY (no forward/backward)' if DATA_ONLY else 'full training step'}")

    # ── Warm up (2 steps, not measured) ──────────────────────────────────────
    it = iter(loader)
    for _ in range(2):
        embs, graphs, pic50s, pids = next(it)
        embs, graphs, pic50s, pids = (
            embs.to(device), graphs.to(device), pic50s.to(device), pids.to(device)
        )
        if not DATA_ONLY:
            preds = model(embs, graphs)
            loss = combined_loss(preds, pic50s, pids)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    torch.mps.empty_cache()
    gc.collect()

    # ── Baseline snapshot ─────────────────────────────────────────────────────
    baseline_rss      = rss_mb()
    baseline_mps      = mps_alloc_mb()
    baseline_driver   = mps_driver_mb()
    baseline_gc       = gc_obj_count()

    print(f"\nBaseline after 2 warm-up steps:")
    print(f"  RSS={baseline_rss:.0f} MB  MPS_alloc={baseline_mps:.0f} MB  "
          f"MPS_driver={baseline_driver:.0f} MB  gc_objs={baseline_gc:,}")
    print()
    print(f"{'Step':>6}  {'RSS MB':>8}  {'ΔRSS':>7}  "
          f"{'MPS_alloc':>10}  {'ΔMPS':>7}  "
          f"{'Driver MB':>10}  {'ΔDriver':>8}  "
          f"{'gc_objs':>8}  {'Δgc':>7}  {'ms/step':>8}")
    print("-" * 100)

    # ── Profiling loop ────────────────────────────────────────────────────────
    step = 0
    t0 = time.perf_counter()
    records = []

    for embs, graphs, pic50s, pids in loader:
        if step >= PROBE_STEPS:
            break

        embs   = embs.to(device)
        graphs = graphs.to(device)
        pic50s = pic50s.to(device)
        pids   = pids.to(device)

        if not DATA_ONLY:
            optimizer.zero_grad()
            preds = model(embs, graphs)
            loss  = combined_loss(preds, pic50s, pids)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if EMPTY_CACHE_EVERY > 0 and (step + 1) % EMPTY_CACHE_EVERY == 0:
                torch.mps.empty_cache()

        step += 1

        if step % PRINT_EVERY == 0:
            # Sync MPS before measuring
            torch.mps.synchronize()
            t1 = time.perf_counter()
            ms_per_step = (t1 - t0) * 1000 / PRINT_EVERY
            t0 = t1

            rss    = rss_mb()
            alloc  = mps_alloc_mb()
            driver = mps_driver_mb()
            gc_n   = gc_obj_count()

            rec = dict(step=step, rss=rss, alloc=alloc, driver=driver, gc_n=gc_n)
            records.append(rec)

            print(
                f"{step:>6}  "
                f"{rss:>8.0f}  {rss - baseline_rss:>+7.0f}  "
                f"{alloc:>10.0f}  {alloc - baseline_mps:>+7.0f}  "
                f"{driver:>10.0f}  {driver - baseline_driver:>+8.0f}  "
                f"{gc_n:>8,}  {gc_n - baseline_gc:>+7,}  "
                f"{ms_per_step:>8.0f}"
            )

    # ── Summary ───────────────────────────────────────────────────────────────
    if not records:
        print("No records collected.")
        return

    first, last = records[0], records[-1]
    print()
    print("=== SUMMARY: change from step 50 to step", last["step"], "===")
    print(f"  RSS growth:        {last['rss']    - first['rss']:>+.0f} MB")
    print(f"  MPS alloc growth:  {last['alloc']  - first['alloc']:>+.0f} MB")
    print(f"  Driver growth:     {last['driver'] - first['driver']:>+.0f} MB")
    print(f"  gc_objs growth:    {last['gc_n']   - first['gc_n']:>+,}")
    print()

    # Linear regression to estimate growth rate
    if len(records) >= 3:
        import numpy as np
        steps_arr = np.array([r["step"] for r in records], dtype=float)
        for key, label in [("rss","RSS"), ("alloc","MPS alloc"), ("driver","Driver")]:
            vals = np.array([r[key] for r in records])
            slope = np.polyfit(steps_arr, vals, 1)[0]
            total_epoch = slope * 2336   # steps in a full epoch at bs=128
            print(f"  {label:12s}: {slope:+.3f} MB/step  →  "
                  f"projected growth over one full epoch: {total_epoch:+.0f} MB")


if __name__ == "__main__":
    main()
