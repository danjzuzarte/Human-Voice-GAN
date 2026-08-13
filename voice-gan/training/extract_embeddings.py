"""
One-time pass: run the frozen frontend over every matched utterance in a
split and cache the resulting frame embeddings to a single consolidated
tensor file (plus labels + filenames), so train_detector.py can train the
classifier head in minutes instead of re-running a 95M-parameter transformer
every epoch.

Cached to local disk by default (configs/model.yaml `paths.embeddings_cache_dir`),
not Drive — regenerating this takes minutes, not worth spending Drive quota on.
Only the trained classifier checkpoint (much smaller) goes to Drive.

Usage:
    python training/extract_embeddings.py --data-config configs/data.yaml --model-config configs/model.yaml --split train
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import ASVspoof5Dataset, load_configs
from models.detector.frontend import SpeechFrontend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    data_cfg, model_cfg = load_configs(args.data_config, args.model_config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[extract] device: {device}")

    dataset = ASVspoof5Dataset(data_cfg, model_cfg, split=args.split, random_crop=False)
    print(f"[extract] {len(dataset)} utterances matched to downloaded audio for split='{args.split}'")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    frontend_cfg = model_cfg["frontend"]
    frontend = SpeechFrontend(checkpoint=frontend_cfg["checkpoint"], freeze=frontend_cfg["freeze"]).to(device)
    frontend.eval()

    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for waveforms, labels in tqdm(loader, desc="extracting embeddings"):
            waveforms = waveforms.to(device)
            embeddings = frontend(waveforms)  # [batch, frames, hidden_dim]
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)

    embeddings = torch.cat(all_embeddings, dim=0)  # [N, frames, hidden_dim]
    labels = torch.cat(all_labels, dim=0)  # [N]

    cache_dir = model_cfg["paths"]["embeddings_cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, f"{args.split}_embeddings.pt")
    torch.save({"embeddings": embeddings, "labels": labels}, out_path)

    size_gb = embeddings.element_size() * embeddings.nelement() / 1e9
    print(f"[extract] saved {embeddings.shape} embeddings ({size_gb:.2f}GB) + {labels.shape} labels -> {out_path}")


if __name__ == "__main__":
    main()
