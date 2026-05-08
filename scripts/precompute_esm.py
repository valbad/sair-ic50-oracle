"""
Precompute ESM2 embeddings for all unique proteins in sair.parquet.

Two output modes (can be run together in one pass):
  --output-dir   Mean-pooled [D] tensors — legacy / fast-load mode.
  --residue-dir  Per-residue [L, D] float16 tensors — required for
                 cross-attention training.

Usage (first-time, both outputs):
    python scripts/precompute_esm.py \\
        --parquet data/sair.parquet \\
        --output-dir data/esm_embeddings/ \\
        --residue-dir data/esm_residue_embeddings/ \\
        --annotations data/long_protein_annotations.csv \\
        --model esm2_t33_650M_UR50D \\
        --batch-size 16 \\
        --device auto

Usage (residue embeddings only, existing mean-pool already done):
    python scripts/precompute_esm.py \\
        --parquet data/sair.parquet \\
        --residue-dir data/esm_residue_embeddings/ \\
        --annotations data/long_protein_annotations.csv

Estimated runtime (650M model):
    - Apple M4 Max (MPS): ~15-20 minutes
    - Colab A100: ~10-15 minutes
"""

import argparse
import os

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def load_esm_model(model_name: str, device: torch.device):
    import esm
    model, alphabet = esm.pretrained.load_model_and_alphabet_hub(model_name)
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter


@torch.no_grad()
def embed_batch(
    sequences: list[tuple[str, str]],
    model,
    batch_converter,
    device: torch.device,
    repr_layer: int,
    save_residues: bool,
) -> dict[str, dict]:
    """
    Embed a batch of sequences.

    Returns dict: uniprot_id -> {"mean": Tensor[D], "residues": Tensor[L, D] float16}
    "residues" key only present when save_residues=True.
    """
    _, _, tokens = batch_converter(sequences)
    tokens = tokens.to(device)

    out = model(tokens, repr_layers=[repr_layer], return_contacts=False)
    representations = out["representations"][repr_layer]  # [B, L+2, D]

    results = {}
    for i, (label, seq) in enumerate(sequences):
        seq_len = len(seq)
        residue_embs = representations[i, 1 : seq_len + 1]  # [L, D] on device
        entry = {"mean": residue_embs.mean(dim=0).cpu()}     # [D]
        if save_residues:
            entry["residues"] = residue_embs.half().cpu()    # [L, D] float16
        results[label] = entry

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet",      type=str, required=True)
    parser.add_argument("--output-dir",   type=str, default=None,
                        help="Directory for mean-pooled [D] embeddings.")
    parser.add_argument("--residue-dir",  type=str, default=None,
                        help="Directory for per-residue [L, D] float16 embeddings.")
    parser.add_argument("--model",        type=str, default="esm2_t33_650M_UR50D")
    parser.add_argument("--batch-size",   type=int, default=16)
    parser.add_argument("--device",       type=str, default="auto")
    parser.add_argument("--annotations",  type=str, default=None,
                        help="Path to long_protein_annotations.csv.")
    parser.add_argument("--max-seq-len",  type=int, default=1022,
                        help="Truncate sequences longer than this (ESM2 limit).")
    args = parser.parse_args()

    if not args.output_dir and not args.residue_dir:
        parser.error("At least one of --output-dir or --residue-dir must be provided.")

    output_dir  = Path(args.output_dir).expanduser()  if args.output_dir  else None
    residue_dir = Path(args.residue_dir).expanduser() if args.residue_dir else None
    save_residues = residue_dir is not None

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    if residue_dir:
        residue_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    print(f"Using device: {device}")
    print(f"Mean-pool output : {output_dir or '(skipped)'}")
    print(f"Residue output   : {residue_dir or '(skipped)'}")

    # ── Load unique proteins ───────────────────────────────────────────────────
    print(f"Loading parquet: {args.parquet}")
    df = pd.read_parquet(
        Path(args.parquet).expanduser(),
        columns=["protein", "sequence"],
    )
    proteins = (
        df.drop_duplicates(subset=["protein"])
        .dropna(subset=["sequence"])
        [["protein", "sequence"]]
        .reset_index(drop=True)
    )
    print(f"Unique proteins: {len(proteins):,}")

    if args.annotations:
        ann  = pd.read_csv(args.annotations)
        keep = set(ann[ann["keep"] == True]["protein"])
        proteins = proteins[proteins["protein"].isin(keep)].reset_index(drop=True)
        print(f"After annotation filter: {len(proteins):,} proteins to embed")

    # Skip proteins already computed in ALL requested outputs
    def already_done(protein: str) -> bool:
        if output_dir  and not (output_dir  / f"{protein}.pt").exists():
            return False
        if residue_dir and not (residue_dir / f"{protein}.pt").exists():
            return False
        return True

    proteins = proteins[~proteins["protein"].apply(already_done)].reset_index(drop=True)
    print(f"Remaining to embed: {len(proteins):,}")

    if len(proteins) == 0:
        print("All embeddings already computed. Nothing to do.")
        return

    # ── Load ESM2 ─────────────────────────────────────────────────────────────
    print(f"Loading ESM2 model: {args.model}")
    model, batch_converter = load_esm_model(args.model, device)

    repr_layer = int(args.model.split("_t")[1].split("_")[0])
    print(f"Extracting representations from layer {repr_layer}")

    # ── Embed in batches ──────────────────────────────────────────────────────
    sequences = [
        (row["protein"], row["sequence"][: args.max_seq_len])
        for _, row in proteins.iterrows()
    ]
    n_batches = (len(sequences) + args.batch_size - 1) // args.batch_size

    def save_results(results: dict):
        for uniprot_id, entry in results.items():
            if output_dir:
                torch.save(entry["mean"], output_dir / f"{uniprot_id}.pt")
            if residue_dir and "residues" in entry:
                torch.save(entry["residues"], residue_dir / f"{uniprot_id}.pt")

    for i in tqdm(range(n_batches), desc="Embedding proteins"):
        batch = sequences[i * args.batch_size : (i + 1) * args.batch_size]
        try:
            results = embed_batch(
                batch, model, batch_converter, device, repr_layer, save_residues
            )
            save_results(results)
        except RuntimeError:
            print(f"\nOOM on batch {i}, retrying one by one...")
            for label, seq in batch:
                try:
                    results = embed_batch(
                        [(label, seq)], model, batch_converter, device,
                        repr_layer, save_residues
                    )
                    save_results(results)
                except RuntimeError:
                    print(f"  Skipping {label} (seq len {len(seq)}): still OOM")
        finally:
            if device.type == "mps":
                torch.mps.empty_cache()

    if output_dir:
        print(f"\nMean-pool embeddings: {len(list(output_dir.glob('*.pt'))):,} files in {output_dir}")
    if residue_dir:
        print(f"Residue embeddings  : {len(list(residue_dir.glob('*.pt'))):,} files in {residue_dir}")


if __name__ == "__main__":
    main()
