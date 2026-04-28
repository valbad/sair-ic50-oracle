"""
Precompute ESM2 embeddings for all unique proteins in sair.parquet.

Saves one .pt file per protein (UniProt ID) containing a 1D tensor of shape [D]
(mean-pooled over residues). This is a one-time cost: ~5,149 proteins.

Usage:
    python scripts/precompute_esm.py \
        --parquet /path/to/sair.parquet \
        --output-dir /path/to/esm_embeddings/ \
        --model esm2_t33_650M_UR50D \
        --batch-size 16 \
        --device auto

Estimated runtime (650M model):
    - Apple M4 Max (MPS): ~15-20 minutes
    - Colab A100: ~10-15 minutes
"""

import argparse
import os

# Must be set before torch is imported — MPS reads this at backend init.
# 0.0 disables the memory pool high-watermark, forcing MPS to release GPU
# memory aggressively. Prevents the GPU from starving WindowServer under load.
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
    sequences: list[tuple[str, str]],  # list of (label, sequence)
    model,
    batch_converter,
    device: torch.device,
    repr_layer: int,
) -> dict[str, torch.Tensor]:
    """Embed a batch of sequences. Returns dict: uniprot_id -> embedding [D]."""
    _, _, tokens = batch_converter(sequences)
    tokens = tokens.to(device)

    out = model(tokens, repr_layers=[repr_layer], return_contacts=False)
    representations = out["representations"][repr_layer]  # [B, L+2, D]

    results = {}
    for i, (label, seq) in enumerate(sequences):
        # Mean pool over actual residues (exclude BOS/EOS tokens)
        seq_len = len(seq)
        emb = representations[i, 1 : seq_len + 1].mean(dim=0).cpu()  # [D]
        results[label] = emb

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="esm2_t33_650M_UR50D")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--annotations",
        type=str,
        default=None,
        help="Path to long_protein_annotations.csv. If provided, only embed "
             "proteins marked keep=True.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=1022,
        help="Truncate sequences longer than this (ESM2 limit is 1022 residues).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    print(f"Using device: {device}")

    # ── Load unique proteins from parquet ─────────────────────────────────────
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
        ann = pd.read_csv(args.annotations)
        keep = set(ann[ann["keep"] == True]["protein"])
        proteins = proteins[proteins["protein"].isin(keep)].reset_index(drop=True)
        print(f"After annotation filter: {len(proteins):,} proteins to embed")

    # Skip already computed
    already_done = {p.stem for p in output_dir.glob("*.pt")}
    proteins = proteins[~proteins["protein"].isin(already_done)]
    print(f"Remaining to embed: {len(proteins):,}")

    if len(proteins) == 0:
        print("All embeddings already computed. Nothing to do.")
        return

    # ── Load ESM2 ─────────────────────────────────────────────────────────────
    print(f"Loading ESM2 model: {args.model}")
    model, batch_converter = load_esm_model(args.model, device)

    # Determine which layer to extract from
    repr_layer = int(args.model.split("_t")[1].split("_")[0])
    print(f"Extracting representations from layer {repr_layer}")

    # ── Embed in batches ──────────────────────────────────────────────────────
    sequences = [
        (row["protein"], row["sequence"][: args.max_seq_len])
        for _, row in proteins.iterrows()
    ]

    n_batches = (len(sequences) + args.batch_size - 1) // args.batch_size

    for i in tqdm(range(n_batches), desc="Embedding proteins"):
        batch = sequences[i * args.batch_size : (i + 1) * args.batch_size]
        try:
            embeddings = embed_batch(batch, model, batch_converter, device, repr_layer)
            for uniprot_id, emb in embeddings.items():
                torch.save(emb, output_dir / f"{uniprot_id}.pt")
        except RuntimeError as e:
            # OOM on a long sequence: fall back to one at a time
            print(f"\nOOM on batch {i}, retrying one by one...")
            for label, seq in batch:
                try:
                    embeddings = embed_batch(
                        [(label, seq)], model, batch_converter, device, repr_layer
                    )
                    torch.save(embeddings[label], output_dir / f"{label}.pt")
                except RuntimeError:
                    print(f"  Skipping {label} (seq len {len(seq)}): still OOM")
        finally:
            # Release MPS memory after every batch so the GPU doesn't accumulate
            # a large pool that starves WindowServer under sustained load.
            if device.type == "mps":
                torch.mps.empty_cache()

    print(f"\nDone. Embeddings saved to {output_dir}")
    print(f"Total files: {len(list(output_dir.glob('*.pt'))):,}")


if __name__ == "__main__":
    main()
