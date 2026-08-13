"""
One-time pass: run the frozen frontend over every matched utterance in a
split and cache the resulting frame embeddings, so train_detector.py can
train the classifier head in minutes instead of re-running a 95M-parameter
transformer every epoch.

Streamed straight to an on-disk memory-mapped file, one batch at a time —
NOT accumulated in a Python list and concatenated at the end. That in-RAM
approach worked for a single ~36k-utterance shard (~11GB in float16) but
doesn't scale: the full 5-shard train set (~182k utterances) would be
~55GB, more than a standard Colab runtime's system RAM. Streaming to a
memmap keeps peak RAM at roughly one batch's worth, independent of how
many shards you pull — cost is disk space instead (plan for ~0.3MB per
utterance in float16 at the default 4s/wav2vec2-base settings), which is
what /content's local disk is for.

Writes three files per split (all under configs/model.yaml
`paths.embeddings_cache_dir`, local disk not Drive — regenerating this
takes minutes, not worth spending Drive quota on; only the much smaller
trained classifier checkpoint goes to Drive):
    {split}_embeddings.dat   raw memory-mapped tensor, float16, shape (N, frames, hidden_dim)
    {split}_meta.json        {"n", "frames", "hidden_dim", "dtype"} needed to reopen the .dat
    {split}_labels.pt        small — just N floats, loaded normally via torch.load

Usage:
    python training/extract_embeddings.py --data-config configs/data.yaml --model-config configs/model.yaml --split train
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    n = len(dataset)
    print(f"[extract] {n} utterances matched to downloaded audio for split='{args.split}'")
    # shuffle=False: iteration order matches dataset row order, so batch i's
    # rows land at the same offsets in the memmap as they'd occupy in the
    # protocol dataframe — no separate index-tracking needed downstream.
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    frontend_cfg = model_cfg["frontend"]
    frontend = SpeechFrontend(checkpoint=frontend_cfg["checkpoint"], freeze=frontend_cfg["freeze"]).to(device)
    frontend.eval()

    cache_dir = model_cfg["paths"]["embeddings_cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)
    mmap_path = os.path.join(cache_dir, f"{args.split}_embeddings.dat")
    meta_path = os.path.join(cache_dir, f"{args.split}_meta.json")
    labels_path = os.path.join(cache_dir, f"{args.split}_labels.pt")

    # Allocated lazily on the first batch, once the real frame count and
    # hidden_dim are known (frame count is deterministic given the fixed-
    # duration crop/pad in dataset.py, but reading it off a real batch is
    # simpler and safer than re-deriving wav2vec2's conv-stride math here).
    mmap_array = None
    write_idx = 0
    all_labels = []
    with torch.no_grad():
        for waveforms, labels in tqdm(loader, desc="extracting embeddings"):
            waveforms = waveforms.to(device)
            embeddings = frontend(waveforms)  # [batch, frames, hidden_dim]
            # .half(): float16 storage, same reasoning as before — halves
            # both this batch's transient RAM and the on-disk file size.
            # CachedEmbeddingDataset upconverts back to float32 per-item when
            # reading, so the classifier head still trains in float32.
            batch_np = embeddings.half().cpu().numpy()

            if mmap_array is None:
                frames, hidden_dim = batch_np.shape[1], batch_np.shape[2]
                mmap_array = np.memmap(mmap_path, dtype=np.float16, mode="w+", shape=(n, frames, hidden_dim))
                with open(meta_path, "w") as f:
                    json.dump({"n": n, "frames": frames, "hidden_dim": hidden_dim, "dtype": "float16"}, f)
                print(f"[extract] allocated on-disk cache: {n} x {frames} x {hidden_dim} (float16) -> {mmap_path}")

            batch_n = batch_np.shape[0]
            mmap_array[write_idx:write_idx + batch_n] = batch_np
            write_idx += batch_n
            all_labels.append(labels)

            # Flush periodically rather than only at the very end, so dirty
            # pages get written back to disk incrementally instead of piling
            # up for one huge flush — keeps the OS's page-cache pressure
            # smoother over a long extraction run (full 5-shard train set is
            # ~180k utterances / thousands of batches).
            if write_idx % (args.batch_size * 250) == 0:
                mmap_array.flush()

    mmap_array.flush()
    del mmap_array  # release the memmap handle, ensures everything is flushed to disk
    assert write_idx == n, f"wrote {write_idx} rows but expected {n} — dataset length mismatch"

    labels = torch.cat(all_labels, dim=0)  # tiny — N floats, fine to hold in RAM
    torch.save(labels, labels_path)

    size_gb = os.path.getsize(mmap_path) / 1e9
    print(f"[extract] saved {write_idx} utterances ({size_gb:.2f}GB, float16, memory-mapped) -> {mmap_path}")
    print(f"[extract] labels -> {labels_path}")


if __name__ == "__main__":
    main()
