"""
Download and extract a configurable subset of the ASVspoof5 dataset from
Hugging Face.

Source: https://huggingface.co/datasets/jungjee/asvspoof5
Public dataset, no gating/token required as of this writing — if you hit a
401/403, run `huggingface-cli login` first.

The full dataset is ~142GB (train+dev+eval), which is impractical to pull
just to validate a pipeline. This script is shard-aware: by default it only
grabs the shards listed in configs/data.yaml (`shards_to_download`), so you
can validate everything end to end on a fraction of the data before
committing to a full download.

Usage:

    python data/download_asvspoof5.py --config configs/data.yaml --split train
    python data/download_asvspoof5.py --config configs/data.yaml --split train --all-shards
    python data/download_asvspoof5.py --config configs/data.yaml --protocols-only
"""
import argparse
import tarfile
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _download_file(repo_id: str, repo_type: str, filename: str, local_dir: str) -> Path:
    print(f"[download] {filename} ...")
    path = hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=filename,
        local_dir=local_dir,
    )
    print(f"[download] done -> {path}")
    return Path(path)


def _extract_tar(tar_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {tar_path.name} -> {dest_dir} ...")
    with tarfile.open(tar_path) as tf:
        tf.extractall(dest_dir)
    print(f"[extract] done: {tar_path.name}")


def download_protocols(cfg: dict) -> Path:
    """Download and extract the protocols tar (small — ~93MB), plus the codec
    config csv if the configured repo actually has one. Not every ASVspoof5
    mirror ships this file, and its absence shouldn't block getting the
    protocols (the actual labels) downloaded and extracted."""
    repo_id = cfg["hf_repo_id"]
    repo_type = cfg["hf_repo_type"]
    raw_dir = cfg["paths"]["raw_download_dir"]

    protocols_tar = _download_file(repo_id, repo_type, cfg["protocols_archive"], raw_dir)

    codec_config_file = cfg.get("codec_config_file")
    if codec_config_file:
        try:
            _download_file(repo_id, repo_type, codec_config_file, raw_dir)
        except Exception as e:
            print(
                f"[protocols] Note: '{codec_config_file}' isn't in this HF repo "
                f"({type(e).__name__}) — skipping it. It's not required to proceed; "
                "set codec_config_file to null in configs/data.yaml to silence this."
            )

    extracted = Path(cfg["paths"]["extracted_dir"]) / "protocols"
    _extract_tar(protocols_tar, extracted)

    print(f"\n[protocols] Extracted to {extracted}")
    print("[protocols] Before writing anything downstream, confirm the real")
    print("[protocols] column layout matches configs/data.yaml's `protocol_columns`:")
    print(f"[protocols]   pd.read_csv('<extracted_tsv>', sep='\\t', header=None).head()")
    return extracted


def download_split(cfg: dict, split: str, all_shards: bool = False) -> list[Path]:
    """Download (and extract) the configured tar shards for a split ('train' or 'dev')."""
    repo_id = cfg["hf_repo_id"]
    repo_type = cfg["hf_repo_type"]
    raw_dir = cfg["paths"]["raw_download_dir"]
    extracted_dir = Path(cfg["paths"]["extracted_dir"])

    split_cfg = cfg["splits"][split]
    letters = split_cfg["available_shards"] if all_shards else split_cfg["shards_to_download"]
    total_available = len(split_cfg["available_shards"])

    if not all_shards and len(letters) < total_available:
        print(
            f"[{split}] Downloading {len(letters)}/{total_available} shards "
            f"({letters}). Pass --all-shards for the full split once the "
            "pipeline is validated."
        )

    extracted_paths = []
    for letter in letters:
        filename = split_cfg["shard_pattern"].format(letter=letter)
        tar_path = _download_file(repo_id, repo_type, filename, raw_dir)
        dest = extracted_dir
        _extract_tar(tar_path, dest)
        extracted_paths.append(dest)

    return extracted_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--split", choices=["train", "dev"], default=None)
    parser.add_argument("--all-shards", action="store_true", help="Download every shard for the split, not just the configured subset")
    parser.add_argument("--protocols-only", action="store_true", help="Only download the protocol/label files, not audio")
    args = parser.parse_args()

    cfg = load_config(args.config)

    download_protocols(cfg)

    if args.protocols_only:
        return

    if args.split is None:
        parser.error("--split is required unless --protocols-only is set")

    download_split(cfg, args.split, all_shards=args.all_shards)


if __name__ == "__main__":
    main()
