"""
PyTorch Dataset for the SAIR IC50 oracle.

Each sample is a (protein_residues, ligand_graph, pIC50) triple.

Protein embeddings:
    Per-residue ESM2 tensors [L, 1280] float16 are pre-loaded into
    self._residue_cache (dict: protein_id -> Tensor[L, D]) at init time.
    Total memory: n_proteins × avg_seq_len × 1280 × 2 bytes (~2-3 GB for 2,859
    proteins at float16). __getitem__ does a plain dict lookup.

Ligand graphs:
    Pre-computed at init from SMILES and cached in self._graph_cache.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from oracle.featurise import smiles_to_graph, smiles_to_descriptors, fallback_graph, ATOM_FEATURE_DIM, DESCRIPTOR_DIM
from torch_geometric.data import Data


# ── Protein filter ─────────────────────────────────────────────────────────────

def load_keep_proteins(annotations_csv: str | Path) -> set[str]:
    """
    Load the set of proteins to keep from data/long_protein_annotations.csv.
    Short proteins always kept. Long proteins kept only if all annotated
    binding sites fall within the first 1022 residues (ESM2 limit).
    """
    df = pd.read_csv(Path(annotations_csv).expanduser())
    return set(df[df["keep"] == True]["protein"].tolist())


# ── Dataset ────────────────────────────────────────────────────────────────────

class SAIRDataset(Dataset):
    """
    Loads SAIR parquet, filters rows, and serves
    (protein_residues, ligand_graph, pIC50_normalized, protein_name) tuples.

    Args:
        pic50_norms: If provided, use these per-protein (mean, std) statistics
                     instead of computing from this split. Pass the training
                     dataset's .pic50_norms to the validation dataset so both
                     use the same scale.
        residue_cache_dir: Directory containing per-residue ESM2 embeddings
                           (each file: protein.pt → Tensor[L, 1280] float16).
                           Required for the cross-attention model.
        max_residues: Truncate per-residue tensors to this length. Must match
                      what was used during precompute_esm.py (default 512).
    """

    def __init__(
        self,
        parquet_path: str | Path,
        esm_cache_dir: str | Path,
        annotations_csv: str | Path = "data/long_protein_annotations.csv",
        entry_ids: Optional[list] = None,
        filter_all_passed: bool = True,
        assay_filter: Optional[str] = "biochem",
        deduplicate: bool = True,
        residue_cache_dir: Optional[str | Path] = None,
        max_residues: int = 1022,
        ligand_cache_path: Optional[str | Path] = None,
    ):
        residue_cache_dir = Path(residue_cache_dir).expanduser() if residue_cache_dir else None

        # ── Load keep-protein set ─────────────────────────────────────────────
        keep_proteins = load_keep_proteins(annotations_csv)
        print(f"Protein filter loaded: {len(keep_proteins):,} proteins allowed")

        # ── Load parquet ──────────────────────────────────────────────────────
        df = pd.read_parquet(
            Path(parquet_path).expanduser(),
            columns=[
                "entry_id", "protein", "sequence", "SMILES",
                "pIC50", "assay", "all_passed", "iptm",
            ],
        )

        # ── Filters ───────────────────────────────────────────────────────────
        if filter_all_passed:
            df = df[df["all_passed"] == True]

        if assay_filter is not None:
            df = df[df["assay"] == assay_filter]

        before = len(df)
        df = df[df["protein"].isin(keep_proteins)]
        print(f"Protein filter: {before:,} -> {len(df):,} rows "
              f"(dropped {before - len(df):,})")

        if entry_ids is not None:
            typed_ids = pd.array(entry_ids, dtype=df["entry_id"].dtype)
            df = df[df["entry_id"].isin(typed_ids)]

        # ── Deduplication ─────────────────────────────────────────────────────
        if deduplicate:
            df = (
                df.sort_values("iptm", ascending=False)
                .drop_duplicates(subset=["sequence", "SMILES"])
                .reset_index(drop=True)
            )

        df = df.dropna(subset=["pIC50", "SMILES", "sequence"]).reset_index(drop=True)

        self.df = df[["entry_id", "protein", "SMILES", "pIC50"]].copy()
        del df

        print(f"Dataset ready: {len(self.df):,} samples across "
              f"{self.df['protein'].nunique():,} proteins")


        # ── Pre-load per-residue ESM embeddings ───────────────────────────────
        unique_proteins = self.df["protein"].unique().tolist()

        if residue_cache_dir is not None:
            missing = [p for p in unique_proteins
                       if not (residue_cache_dir / f"{p}.pt").exists()]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} per-residue ESM embeddings missing in "
                    f"{residue_cache_dir}. First missing: {missing[0]}. "
                    f"Run: python scripts/precompute_esm.py --residue-dir {residue_cache_dir}"
                )
            print(f"Pre-loading {len(unique_proteins):,} per-residue ESM tensors "
                  f"(max {max_residues} residues) ... ", end="", flush=True)
            self._residue_cache: dict[str, torch.Tensor] = {
                p: torch.load(
                    residue_cache_dir / f"{p}.pt", weights_only=True
                )[:max_residues]          # [L, D] float16; max_residues=1022 = ESM2 limit
                for p in unique_proteins
            }
            total_mb = sum(
                t.numel() * t.element_size() for t in self._residue_cache.values()
            ) / 1e6
            print(f"done ({total_mb:.0f} MB)")
        else:
            # Fall back to mean-pooled embeddings (legacy / ablation mode)
            esm_cache_dir_p = Path(esm_cache_dir).expanduser()
            missing = [p for p in unique_proteins
                       if not (esm_cache_dir_p / f"{p}.pt").exists()]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} ESM embeddings missing in {esm_cache_dir_p}."
                )
            print(f"Pre-loading {len(unique_proteins):,} mean-pooled ESM embeddings "
                  f"(legacy mode) ... ", end="", flush=True)
            _mean_cache = {
                p: torch.load(esm_cache_dir_p / f"{p}.pt", weights_only=True)
                for p in unique_proteins
            }
            # Wrap [D] tensors as [1, D] so the model can treat them as 1-residue
            # sequences; mask is all-True for every sample.
            self._residue_cache = {
                p: emb.unsqueeze(0).half()  # [1, D] float16
                for p, emb in _mean_cache.items()
            }
            total_mb = sum(
                t.numel() * t.element_size() for t in self._residue_cache.values()
            ) / 1e6
            print(f"done ({total_mb:.0f} MB, legacy mean-pool mode)")

        # ── Load or compute ligand graphs and descriptors ─────────────────────
        unique_smiles = self.df["SMILES"].unique().tolist()

        if ligand_cache_path is not None:
            cache_path = Path(ligand_cache_path).expanduser()
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"Ligand cache not found at {cache_path}. "
                    f"Run: python scripts/precompute_ligands.py "
                    f"--parquet {parquet_path} --output {cache_path}"
                )
            print(f"Loading ligand cache from {cache_path} ... ", end="", flush=True)
            ligand_cache: dict = torch.load(cache_path, weights_only=False)
            self._graph_cache: dict[str, Data] = {
                s: ligand_cache[s]["graph"] for s in unique_smiles
            }
            self._descriptor_cache: dict[str, torch.Tensor] = {
                s: ligand_cache[s]["descriptor"] for s in unique_smiles
            }
            print(f"done ({len(self._graph_cache):,} entries)")
        else:
            print(f"Pre-computing {len(unique_smiles):,} ligand graphs + descriptors ... ", end="", flush=True)
            self._graph_cache = {
                s: (smiles_to_graph(s) or fallback_graph())
                for s in unique_smiles
            }
            self._descriptor_cache = {
                s: smiles_to_descriptors(s)
                for s in unique_smiles
            }
            print("done")

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        protein = row["protein"]
        smiles  = row["SMILES"]

        residues    = self._residue_cache[protein]           # [L, D] float16
        graph       = self._graph_cache[smiles]
        descriptors = self._descriptor_cache[smiles]         # [DESCRIPTOR_DIM] float32
        pic50       = torch.tensor(float(row["pIC50"]), dtype=torch.float)

        return residues, graph, descriptors, pic50, protein

    # ── Utility ───────────────────────────────────────────────────────────────

    def unique_proteins(self) -> list[str]:
        return self.df["protein"].unique().tolist()

    def pic50_stats(self) -> dict:
        s = self.df["pIC50"]
        return {
            "count": len(s),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
        }
