"""
Precompute ligand features for all unique SMILES in sair.parquet.

Two output modes (can be combined in one run):

  --output           GNN cache: {smiles: {'graph': Data, 'descriptor': Tensor[12]}}
                     Used by the baseline GNN model and GFlowNet repo.

  --chemberta-output ChemBERTa cache: {smiles: Tensor[768]}
                     Used by the ChemBERTa oracle model.

Usage:
    # GNN cache only (baseline / GFlowNet compat):
    python scripts/precompute_ligands.py \\
        --parquet data/sair.parquet \\
        --output  data/ligand_cache.pt

    # ChemBERTa cache only:
    python scripts/precompute_ligands.py \\
        --parquet           data/sair.parquet \\
        --chemberta-output  data/chem_emb_cache.pt

    # Both in one run:
    python scripts/precompute_ligands.py \\
        --parquet           data/sair.parquet \\
        --output            data/ligand_cache.pt \\
        --chemberta-output  data/chem_emb_cache.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from oracle.featurise import smiles_to_graph, smiles_to_descriptors, fallback_graph


def build_gnn_cache(unique_smiles: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    failed = 0
    for smiles in tqdm(unique_smiles, desc="GNN featurising"):
        graph = smiles_to_graph(smiles) or fallback_graph()
        desc  = smiles_to_descriptors(smiles)
        cache[smiles] = {"graph": graph, "descriptor": desc}
        if graph.x.sum() == 0 and graph.x.shape[0] == 1:
            failed += 1
    print(f"Done. {len(cache):,} entries ({failed} fallback graphs).")
    torch.save(cache, output_path)
    print(f"GNN cache saved to {output_path}  ({output_path.stat().st_size / 1e6:.0f} MB)")


def build_chemberta_cache(unique_smiles: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import numpy as np
    from transformers import AutoTokenizer, AutoModel

    model_name = "seyonec/ChemBERTa-zinc-base-v1"
    # Use CUDA if available, otherwise CPU.
    # MPS (Apple Silicon) is intentionally skipped — PyTorch's MPS allocator has a
    # known memory-leak bug (github.com/pytorch/pytorch/issues/164299) that causes
    # memory to accumulate unboundedly in long inference loops.
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Loading {model_name} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name).eval().to(device)
    print(f"GPU threads: {torch.get_num_threads()}" if device.type == "cpu"
          else f"GPU: {torch.cuda.get_device_name(0)}")

    BATCH      = 256 if device.type == "cuda" else 128
    MAX_LENGTH = 128   # >99% of drug-like SMILES tokenise to <128 tokens

    # Store as numpy float32 — smaller Python-object overhead than torch.Tensor.
    # .copy() on each row gives it independent storage so the full batch array
    # is freed after each iteration.
    cache: dict[str, np.ndarray] = {}
    batches = [unique_smiles[i : i + BATCH] for i in range(0, len(unique_smiles), BATCH)]

    with torch.no_grad():
        for batch_smiles in tqdm(batches, desc=f"ChemBERTa ({device.type})"):
            enc = tokenizer(
                batch_smiles,
                return_tensors="pt",
                max_length=MAX_LENGTH,
                truncation=True,
                padding=True,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            cls = out.last_hidden_state[:, 0, :].float().cpu().numpy()  # [B, 768]
            for smiles, emb in zip(batch_smiles, cls):
                cache[smiles] = emb.copy()

    print(f"Done. {len(cache):,} ChemBERTa embeddings computed.")
    torch.save(cache, output_path)
    print(f"ChemBERTa cache saved to {output_path}  ({output_path.stat().st_size / 1e6:.0f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Precompute ligand features.")
    parser.add_argument("--parquet", required=True, help="Path to sair.parquet")
    parser.add_argument("--output",  default=None,
                        help="Output path for GNN graph+descriptor cache (.pt)")
    parser.add_argument("--chemberta-output", default=None,
                        help="Output path for ChemBERTa CLS embedding cache (.pt)")
    args = parser.parse_args()

    if args.output is None and args.chemberta_output is None:
        parser.error("Provide at least one of --output or --chemberta-output.")

    parquet_path = Path(args.parquet).expanduser()
    print(f"Reading SMILES from {parquet_path} ...")
    df = pd.read_parquet(parquet_path, columns=["SMILES"])
    unique_smiles = df["SMILES"].dropna().unique().tolist()
    print(f"Unique SMILES: {len(unique_smiles):,}")

    if args.output:
        build_gnn_cache(unique_smiles, Path(args.output).expanduser())

    if args.chemberta_output:
        build_chemberta_cache(unique_smiles, Path(args.chemberta_output).expanduser())


if __name__ == "__main__":
    main()
