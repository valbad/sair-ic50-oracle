"""
Generate train/val/test splits for the SAIR dataset.

Strategy:
    1. Filter rows (all_passed=True, assay='biochem')
    2. Apply protein filter from data/long_protein_annotations.csv
       (drops long proteins with at-risk or unannotated binding sites)
    3. Deduplicate: keep one row per (sequence, SMILES), highest iptm wins
    4. Cluster proteins at 80% sequence identity with MMseqs2
    5. Assign clusters to splits (80/10/10) by protein cluster
    6. Write data/splits/train.txt, val.txt, test.txt (lists of entry_id)

Usage:
    python scripts/make_splits.py \
        --parquet /path/to/sair.parquet \
        --annotations data/long_protein_annotations.csv \
        --output-dir data/splits/ \
        --tmp-dir /tmp/mmseqs_sair/ \
        --seed 42
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def load_keep_proteins(annotations_csv: Path) -> set[str]:
    df = pd.read_csv(annotations_csv)
    keep = set(df[df["keep"] == True]["protein"].tolist())
    print(f"Protein filter: {len(keep):,} proteins allowed "
          f"({len(df)-len(keep):,} dropped)")
    return keep


def load_and_filter(parquet_path: Path, keep_proteins: set[str]) -> pd.DataFrame:
    print("Loading parquet...")
    df = pd.read_parquet(
        parquet_path,
        columns=["entry_id", "protein", "sequence", "SMILES",
                 "pIC50", "assay", "all_passed", "iptm"],
    )

    before = len(df)
    df = df[df["all_passed"] == True]
    df = df[df["assay"] == "biochem"]
    df = df.dropna(subset=["pIC50", "SMILES", "sequence"])
    print(f"After biochem + all_passed filter: {len(df):,} rows (from {before:,})")

    df = df[df["protein"].isin(keep_proteins)]
    print(f"After protein filter: {len(df):,} rows")

    df = (
        df.sort_values("iptm", ascending=False)
        .drop_duplicates(subset=["sequence", "SMILES"])
        .reset_index(drop=True)
    )
    print(f"After deduplication: {len(df):,} unique (protein, SMILES) pairs")
    print(f"Unique proteins: {df['protein'].nunique():,}")
    print(f"Unique SMILES:   {df['SMILES'].nunique():,}")
    return df


def write_fasta(proteins: pd.DataFrame, fasta_path: Path):
    with open(fasta_path, "w") as f:
        for _, row in proteins.iterrows():
            f.write(f">{row['protein']}\n{row['sequence']}\n")
    print(f"Wrote {len(proteins):,} sequences to {fasta_path}")


def run_mmseqs2_clustering(
    fasta_path: Path,
    tmp_dir: Path,
    min_seq_id: float = 0.8,
    coverage: float = 0.8,
) -> pd.DataFrame:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = tmp_dir / "clusters"
    mmseqs_tmp = tmp_dir / "mmseqs_tmp"

    cmd = [
        "mmseqs", "easy-cluster",
        str(fasta_path), str(out_prefix), str(mmseqs_tmp),
        "--min-seq-id", str(min_seq_id),
        "-c", str(coverage),
        "--cov-mode", "0",
        "--cluster-mode", "0",
        "-v", "1",
    ]

    print(f"Running MMseqs2 (min_seq_id={min_seq_id}, coverage={coverage})...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("MMseqs2 stderr:", result.stderr)
        raise RuntimeError("MMseqs2 clustering failed.")

    tsv_path = Path(str(out_prefix) + "_cluster.tsv")
    clusters = pd.read_csv(tsv_path, sep="\t", header=None,
                           names=["representative", "member"])
    print(f"MMseqs2: {clusters['representative'].nunique():,} clusters "
          f"for {len(clusters):,} proteins")
    return clusters


def assign_splits(
    clusters: pd.DataFrame,
    ratios: tuple = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    reps = clusters["representative"].unique()
    rng.shuffle(reps)

    n = len(reps)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    train_reps = set(reps[:n_train])
    val_reps = set(reps[n_train:n_train + n_val])

    split_map = {}
    for _, row in clusters.iterrows():
        rep = row["representative"]
        if rep in train_reps:
            split_map[row["member"]] = "train"
        elif rep in val_reps:
            split_map[row["member"]] = "val"
        else:
            split_map[row["member"]] = "test"

    for name in ["train", "val", "test"]:
        count = sum(1 for s in split_map.values() if s == name)
        print(f"  {name}: {count:,} proteins")

    return split_map


def write_splits(df: pd.DataFrame, split_map: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    df["split"] = df["protein"].map(split_map)
    total = len(df)
    for split_name in ["train", "val", "test"]:
        entry_ids = df[df["split"] == split_name]["entry_id"].tolist()
        out_path = output_dir / f"{split_name}.txt"
        with open(out_path, "w") as f:
            f.write("\n".join(str(e) for e in entry_ids))
        print(f"  {split_name}: {len(entry_ids):,} samples "
              f"({100*len(entry_ids)/total:.1f}%) -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, required=True)
    parser.add_argument("--annotations", type=str,
                        default="data/long_protein_annotations.csv")
    parser.add_argument("--output-dir", type=str, default="data/splits/")
    parser.add_argument("--tmp-dir", type=str, default="/tmp/mmseqs_sair/")
    parser.add_argument("--min-seq-id", type=float, default=0.8)
    parser.add_argument("--coverage", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    keep_proteins = load_keep_proteins(Path(args.annotations))
    df = load_and_filter(Path(args.parquet).expanduser(), keep_proteins)

    unique_proteins = (
        df.drop_duplicates("protein")[["protein", "sequence"]]
        .reset_index(drop=True)
    )
    fasta_path = Path(args.tmp_dir) / "proteins.fasta"
    Path(args.tmp_dir).mkdir(parents=True, exist_ok=True)
    write_fasta(unique_proteins, fasta_path)

    clusters = run_mmseqs2_clustering(
        fasta_path, Path(args.tmp_dir),
        min_seq_id=args.min_seq_id,
        coverage=args.coverage,
    )

    print("\nSplit assignment (by protein cluster):")
    split_map = assign_splits(clusters, seed=args.seed)

    print("\nWriting split files:")
    write_splits(df, split_map, Path(args.output_dir))

    print("\nDone. Commit data/splits/ and data/long_protein_annotations.csv to git.")


if __name__ == "__main__":
    main()
