"""
Evaluates a trained checkpoint against a real, properly held-out split
(dev, by default) — as opposed to train_detector.py's in-loop validation,
which is just a random slice of the *train* shard. That in-loop number is
useful for picking a checkpoint during training, but it's optimistic: a
random split of train isn't speaker-disjoint the way the real train/dev
partition is (see configs/data.yaml note on ASVspoof5's speaker-disjoint
partitions), so the classifier can pick up on speaker/channel cues that
won't transfer to genuinely new speakers. This script gives the number
that actually matters.

Usage:
    python training/extract_embeddings.py --split dev   # run once first
    python training/evaluate_detector.py --data-config configs/data.yaml --model-config configs/model.yaml --split dev
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import load_configs
from eval.metrics import compute_eer
from models.detector.classifier import Detector
from training.train_detector import CachedEmbeddingDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--split", default="dev", help="which cached embeddings to evaluate against")
    parser.add_argument("--checkpoint", default=None, help="defaults to <checkpoint_dir>/detector_best.pt")
    args = parser.parse_args()

    data_cfg, model_cfg = load_configs(args.data_config, args.model_config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[evaluate] device: {device}")

    cache_dir = model_cfg["paths"]["embeddings_cache_dir"]
    cache_prefix = os.path.join(cache_dir, args.split)
    if not os.path.exists(f"{cache_prefix}_meta.json"):
        raise FileNotFoundError(
            f"{cache_prefix}_meta.json not found — run training/extract_embeddings.py --split {args.split} first."
        )
    dataset = CachedEmbeddingDataset(cache_prefix)
    print(f"[evaluate] loaded {len(dataset)} cached '{args.split}' embeddings")

    checkpoint_path = args.checkpoint or os.path.join(model_cfg["paths"]["checkpoint_dir"], "detector_best.pt")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Detector(
        frontend=None,
        hidden_dim=ckpt["hidden_dim"],
        mlp_hidden=ckpt["classifier_config"]["mlp_hidden"],
        dropout=ckpt["classifier_config"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[evaluate] loaded checkpoint from {checkpoint_path} (trained epoch {ckpt['epoch']}, in-loop val EER {ckpt['val_eer']:.2%})")

    loader = DataLoader(dataset, batch_size=model_cfg["training"]["batch_size"], shuffle=False)
    all_labels, all_scores = [], []
    with torch.no_grad():
        for embeddings, labels in loader:
            embeddings = embeddings.to(device)
            scores = torch.sigmoid(model(embeddings))
            all_labels.append(labels.numpy())
            all_scores.append(scores.cpu().numpy())
    all_labels = np.concatenate(all_labels)
    all_scores = np.concatenate(all_scores)

    eer, threshold = compute_eer(all_labels, all_scores)
    accuracy = float(((all_scores > 0.5).astype(int) == all_labels).mean())
    bonafide_fraction = float(all_labels.mean())

    print(f"\n[evaluate] === {args.split} split — {len(dataset)} utterances, {bonafide_fraction:.1%} bonafide ===")
    print(f"[evaluate] EER: {eer:.2%}  (threshold: {threshold:.4f})")
    print(f"[evaluate] accuracy @ 0.5: {accuracy:.2%}")
    print(f"\n[evaluate] compare this against the in-loop train-val EER above — a real gap between the two "
          f"is expected and tells you how much the in-loop number was overfit to train-shard-specific cues.")


if __name__ == "__main__":
    main()
