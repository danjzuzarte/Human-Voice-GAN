"""
Trains the detector's classifier head. Fast path (default, recommended):
trains on embeddings cached by extract_embeddings.py — this is a small
model over small tensors, so a full training run takes minutes on a T4.

Usage:
    python training/extract_embeddings.py --split train   # run once first
    python training/train_detector.py --data-config configs/data.yaml --model-config configs/model.yaml
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import load_configs
from eval.metrics import compute_eer
from models.detector.classifier import Detector


class CachedEmbeddingDataset(Dataset):
    """Reads embeddings from the on-disk memory-mapped cache extract_embeddings.py
    writes (`{prefix}_embeddings.dat` + `{prefix}_meta.json` + `{prefix}_labels.pt`)
    rather than loading the whole split into RAM at once — the earlier
    single-.pt-file, all-in-RAM format didn't scale past ~1 shard. `cache_prefix`
    is the shared prefix, e.g. '.../train' for '.../train_embeddings.dat' etc."""

    def __init__(self, cache_prefix: str):
        with open(f"{cache_prefix}_meta.json") as f:
            meta = json.load(f)
        self.n = meta["n"]
        self.frames = meta["frames"]
        self.hidden_dim = meta["hidden_dim"]
        dtype = np.float16 if meta["dtype"] == "float16" else np.float32
        # mode="r": read-only, memory-mapped — the OS pages data in from disk
        # on demand as __getitem__ touches it, instead of this process ever
        # holding the whole split resident in RAM.
        self.embeddings = np.memmap(
            f"{cache_prefix}_embeddings.dat", dtype=dtype, mode="r",
            shape=(self.n, self.frames, self.hidden_dim),
        )
        self.labels = torch.load(f"{cache_prefix}_labels.pt")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # np.array(...) copies this ONE utterance's slice out of the memmap
        # into an owned, writable array — memmap slices are read-only views
        # and shouldn't be handed to torch/DataLoader collation directly.
        # Only ~0.3MB materializes per item, not the whole split.
        item = np.array(self.embeddings[idx], dtype=np.float32)
        return torch.from_numpy(item), self.labels[idx]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device) -> tuple:
    model.eval()
    all_labels, all_scores = [], []
    with torch.no_grad():
        for embeddings, labels in loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            logits = model(embeddings)
            scores = torch.sigmoid(logits)
            all_labels.append(labels.cpu().numpy())
            all_scores.append(scores.cpu().numpy())
    all_labels = np.concatenate(all_labels)
    all_scores = np.concatenate(all_scores)
    eer, threshold = compute_eer(all_labels, all_scores)
    accuracy = float(((all_scores > 0.5).astype(int) == all_labels).mean())
    return eer, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    args = parser.parse_args()

    data_cfg, model_cfg = load_configs(args.data_config, args.model_config)
    train_cfg = model_cfg["training"]
    set_seed(train_cfg["seed"])

    if not train_cfg.get("use_cached_embeddings", True):
        raise NotImplementedError(
            "use_cached_embeddings: false (training with the frontend inline) isn't "
            "implemented yet — that's a future fine-tuning path, not needed for the "
            "current MVP. Run extract_embeddings.py and keep this true for now."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device: {device}")

    cache_dir = model_cfg["paths"]["embeddings_cache_dir"]
    cache_prefix = os.path.join(cache_dir, "train")
    if not os.path.exists(f"{cache_prefix}_meta.json"):
        raise FileNotFoundError(
            f"{cache_prefix}_meta.json not found — run training/extract_embeddings.py --split train first."
        )
    full_dataset = CachedEmbeddingDataset(cache_prefix)
    print(f"[train] loaded {len(full_dataset)} cached embeddings")

    val_fraction = train_cfg["val_fraction"]
    val_size = int(len(full_dataset) * val_fraction)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(train_cfg["seed"]),
    )
    print(f"[train] split: {train_size} train / {val_size} val")

    batch_size = train_cfg["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    hidden_dim = full_dataset.embeddings.shape[-1]
    clf_cfg = model_cfg["classifier"]
    model = Detector(
        frontend=None,  # cached-embeddings fast path — see models/detector/classifier.py
        hidden_dim=hidden_dim,
        mlp_hidden=clf_cfg["mlp_hidden"],
        dropout=clf_cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"]
    )
    criterion = nn.BCEWithLogitsLoss()

    checkpoint_dir = model_cfg["paths"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_eer = float("inf")

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(embeddings)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * embeddings.size(0)

        train_loss = total_loss / train_size
        eer, accuracy = evaluate(model, val_loader, device)
        print(f"[train] epoch {epoch:02d}/{train_cfg['epochs']} | loss={train_loss:.4f} | val EER={eer:.2%} | val acc={accuracy:.2%}")

        if eer < best_eer:
            best_eer = eer
            ckpt_path = os.path.join(checkpoint_dir, "detector_best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "hidden_dim": hidden_dim,
                "classifier_config": clf_cfg,
                "epoch": epoch,
                "val_eer": eer,
            }, ckpt_path)
            print(f"[train]   -> new best, saved to {ckpt_path}")

    print(f"[train] done. best val EER: {best_eer:.2%}")


if __name__ == "__main__":
    main()
