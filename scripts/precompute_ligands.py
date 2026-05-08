"""
Precompute ligand graphs and physicochemical descriptors for all unique SMILES
in sair.parquet and save them to a single cache file.

This is a one-time step. Training and evaluation load from the cache instead
of recomputing RDKit featurisation every run (~100k SMILES, several minutes).

Usage:
    python scripts/precompute_ligands.py \
        --parquet data/sair.parquet \
        --output  data/ligand_cache.pt

Output format:
    torch.save({smiles: {'graph': Data, 'descriptor': Tensor[12]}}, output)
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


def main():
    parser = argparse.ArgumentParser(description="Precompute ligand graphs and descriptors.")
    parser.add_argument("--parquet", required=True, help="Path to sair.parquet")
    parser.add_argument("--output",  required=True, help="Output .pt cache file path")
    args = parser.parse_args()

    parquet_path = Path(args.parquet).expanduser()
    output_path  = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading SMILES from {parquet_path} ...")
    df = pd.read_parquet(parquet_path, columns=["SMILES"])
    unique_smiles = df["SMILES"].dropna().unique().tolist()
    print(f"Unique SMILES: {len(unique_smiles):,}")

    cache = {}
    failed = 0
    for smiles in tqdm(unique_smiles, desc="Featurising"):
        graph = smiles_to_graph(smiles) or fallback_graph()
        desc  = smiles_to_descriptors(smiles)
        cache[smiles] = {"graph": graph, "descriptor": desc}
        if graph.x.sum() == 0 and graph.x.shape[0] == 1:
            failed += 1

    print(f"Done. {len(cache):,} entries cached  ({failed} fallback graphs).")
    print(f"Saving to {output_path} ...")
    torch.save(cache, output_path)
    size_mb = output_path.stat().st_size / 1e6
    print(f"Saved ({size_mb:.0f} MB).")


if __name__ == "__main__":
    main()
