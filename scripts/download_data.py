"""
Download sair.parquet from HuggingFace.

Usage:
    python scripts/download_data.py --output-dir /path/to/sair-data/

Authentication:
    Run `huggingface-cli login` first, or set the HF_TOKEN environment variable.
    You must have requested and been granted access to SandboxAQ/SAIR on HF.
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "SandboxAQ/SAIR"
FILENAME = "sair.parquet"


def download(output_dir: str) -> Path:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN")  # optional, login via CLI is preferred

    print(f"Downloading {FILENAME} from {REPO_ID} ...")
    print(f"Destination: {output_dir}")

    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        repo_type="dataset",
        local_dir=str(output_dir),
        token=token,
    )

    print(f"Done. File saved to: {local_path}")
    return Path(local_path)


def main():
    parser = argparse.ArgumentParser(description="Download SAIR parquet file.")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where sair.parquet will be saved.",
    )
    args = parser.parse_args()
    download(args.output_dir)


if __name__ == "__main__":
    main()
