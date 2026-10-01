"""
Harden the detector against the current round's fine-tuned generator — the
`harden_detector` step of the LangGraph adversarial loop (see graph/nodes.py,
ARCHITECTURE.md).

Generates fresh hard-negative clips with this round's fine-tuned F5-TTS
generator (via GeneratorWrapper's normal full-quality 32-step inference —
NOT the differentiable few-step sampler, which is training-only for the
*generator* side, see models/generator/differentiable_sampling.py), embeds
them with the frozen frontend, and fine-tunes the detector's classifier head
(warm-started from the previous round's checkpoint, not from scratch) on a
mix of:
  - the fresh hard negatives (label=spoof)
  - a resampled "replay" batch of real audio (label=whatever it actually
    is), so hardening against this round's generator doesn't overwrite what
    the detector already learned from the real ASVspoof5 train set
    (catastrophic forgetting) — see configs/adversarial.yaml
    `detector_hardening.n_replay_real`.

Only the classifier head is trained here, same as train_detector.py — the
frontend stays frozen throughout the whole project (configs/model.yaml
`frontend.freeze`).

Acoustic-domain augmentation (configs/adversarial.yaml's `augmentation`
section, see data/augmentation.py): when enabled, both the fresh hard
negatives and the replayed real audio are randomly perturbed (additive
noise, gain, optional lowpass) before being embedded, so the detector
doesn't just learn "clean recording = bonafide." This is a "loop-only"
augmentation branch — `detector_best.pt` stays the shared, unaugmented
starting checkpoint across every branch; only what happens inside this loop
changes.

When augmentation is DISABLED (the default, and what every real Colab run
before this branch used), replay uses the original fast path — reading
directly from `extract_embeddings.py`'s pre-cached train-embedding memmap,
no raw audio or frontend forward pass needed. When augmentation is ENABLED,
replay instead re-loads raw audio for the sampled rows via
`data.dataset.ASVspoof5Dataset` and re-embeds it live through the frozen
frontend each round (still cheap — a frozen-frontend forward pass over
`n_replay_real` short clips, not training) — this requires the real
ASVspoof5 `train` split's audio to be downloaded/extracted locally, not just
its cached embeddings; see the adversarial-loop notebook's updated
data-setup cell.

Reports the new fooling rate (fraction of a held-out set of freshly
generated clips the hardened detector now scores as bonafide) at the end,
which is what graph/nodes.py's conditional edge reads to decide whether
another harden_generator round is needed.

Per-side conditional routing (configs/adversarial.yaml's
`routing.conditional`, see graph/graph.py's decide_side_node): when a
round's graph/nodes.py decides the *generator* is this round's active side
(i.e. harden_generator ran, but the detector shouldn't be touched this
round), this script is called with `--eval-only`. That skips the
classifier-head training loop completely — the detector's weights are
carried over unmodified — but still generates fresh hard negatives from
--generator-checkpoint and evaluates the CURRENT, untouched detector against
them (plus a fresh replay mix), so fooling_rate_history/val_eer_history
still get a real per-round reading either way, and
route_after_harden_detector (graph/graph.py) doesn't need to know or care
which mode produced it. See `main()`'s `if args.eval_only:` branch. This is
the "loop-only" per-side routing branch — same relationship to the
always-both baseline as the augmentation flag: disabled (the default)
reproduces the original always-train behavior bit-for-bit.

Usage:
    python training/harden_detector.py --data-config configs/data.yaml \\
        --model-config configs/model.yaml --generator-config configs/generator.yaml \\
        --adversarial-config configs/adversarial.yaml \\
        --generator-checkpoint <path to this round's fine-tuned transformer, from finetune_generator.py> \\
        --detector-checkpoint-in <path to the previous round's detector_best.pt> \\
        --detector-checkpoint-out <path to save the hardened detector> \\
        --round 1
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.augmentation import augment_batch
from data.dataset import ASVspoof5Dataset, load_configs
from eval.metrics import compute_eer
from models.detector.classifier import Detector
from models.detector.frontend import SpeechFrontend
from models.generator.f5_tts_wrapper import GeneratorWrapper
from training.generate_samples import load_target_sentences, sample_reference_rows


class MixedEmbeddingDataset(Dataset):
    """In-memory dataset over a small mix of (embedding, label) tensors —
    unlike CachedEmbeddingDataset (training/train_detector.py), this is
    freshly computed each round from a handful of hard negatives plus a
    replay sample, so it's small enough to just hold in RAM rather than
    memory-map to disk."""

    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor):
        assert embeddings.shape[0] == labels.shape[0]
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def _load_and_fix_length(path: str, target_sr: int, max_samples: int) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    waveform = waveform.squeeze(0)
    n = waveform.shape[-1]
    if n == max_samples:
        return waveform
    if n > max_samples:
        return waveform[:max_samples]
    return torch.nn.functional.pad(waveform, (0, max_samples - n))


def generate_hard_negatives(
    generator: GeneratorWrapper, data_cfg: dict, gen_cfg: dict, det_cfg: dict,
    output_dir: str, n: int, rng: random.Random,
) -> "list[str]":
    """Clones n fresh utterances with this round's fine-tuned generator,
    reusing generate_samples.py's reference-sampling / target-sentence
    machinery rather than reimplementing it — same reference_split
    convention (dev, disjoint from the detector's original train-split
    embeddings)."""
    reference_split_cfg = {
        **gen_cfg,
        "cloning": {**gen_cfg["cloning"], "reference_split": det_cfg["reference_split"], "n_speakers": n},
    }
    reference_rows = sample_reference_rows(data_cfg, reference_split_cfg)
    sentences = load_target_sentences(gen_cfg["cloning"]["target_sentences_file"])

    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i, ref in enumerate(reference_rows):
        gen_text = sentences[rng.randrange(len(sentences))]
        out_path = os.path.join(output_dir, f"hard_negative_{i:04d}.wav")
        generator.clone(ref["path"], gen_text, output_path=out_path)
        paths.append(out_path)
    return paths


def _embed_waveforms(waveforms: torch.Tensor, frontend: SpeechFrontend, device: str, batch_size: int = 16) -> torch.Tensor:
    """Runs the frozen frontend over an already-loaded (and, if applicable,
    already-augmented) batch of fixed-length waveforms, shape
    [n, samples]. Shared by embed_clips (fresh hard negatives, loaded from
    disk) and sample_and_embed_replay (real audio, loaded via
    ASVspoof5Dataset) so both paths embed identically."""
    frontend.eval()
    all_embeddings = []
    with torch.no_grad():
        for start in range(0, waveforms.shape[0], batch_size):
            batch = waveforms[start:start + batch_size].to(device)
            embeddings = frontend(batch)  # [batch, frames, hidden_dim]
            all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)


def embed_clips(
    paths: "list[str]", frontend: SpeechFrontend, device: str, target_sr: int, max_samples: int,
    batch_size: int = 16, augment_cfg: "dict | None" = None, rng: "random.Random | None" = None,
) -> torch.Tensor:
    """Runs the frozen frontend over freshly generated clips, matching the
    exact preprocessing extract_embeddings.py applies to real ASVspoof5
    audio (resample, mono-mix, fixed-length crop/pad) so the resulting
    embeddings are directly comparable to / mixable with the cached ones.

    If augment_cfg is given (configs/adversarial.yaml's `augmentation`
    section, with `enabled: true`), each clip is randomly perturbed (see
    data/augmentation.py) before embedding — applied to the hard negatives
    too, not just the replayed real audio, so the detector also learns to
    spot this round's generator under noisy/channel-varied conditions, not
    only its clean default output."""
    waveforms = torch.stack([_load_and_fix_length(p, target_sr, max_samples) for p in paths])
    if augment_cfg and augment_cfg.get("enabled"):
        waveforms = augment_batch(waveforms, target_sr, rng, augment_cfg)
    return _embed_waveforms(waveforms, frontend, device, batch_size)


def sample_replay_embeddings(cache_prefix: str, n: int, rng: random.Random) -> "tuple[torch.Tensor, torch.Tensor]":
    """Draws n random rows from the cached train-embeddings memmap
    (training/extract_embeddings.py's output) as the catastrophic-forgetting
    guard — see module docstring. Fast path, used when augmentation is
    disabled (the default, and what every prior real Colab run used) — see
    sample_and_embed_replay for the augmentation-enabled equivalent."""
    with open(f"{cache_prefix}_meta.json") as f:
        meta = json.load(f)
    total = meta["n"]
    dtype = np.float16 if meta["dtype"] == "float16" else np.float32
    embeddings = np.memmap(
        f"{cache_prefix}_embeddings.dat", dtype=dtype, mode="r",
        shape=(total, meta["frames"], meta["hidden_dim"]),
    )
    labels = torch.load(f"{cache_prefix}_labels.pt")

    idx = rng.sample(range(total), min(n, total))
    replay_embeddings = torch.from_numpy(np.array(embeddings[idx], dtype=np.float32))
    replay_labels = labels[idx]
    return replay_embeddings, replay_labels


def sample_and_embed_replay(
    data_cfg: dict, model_cfg: dict, frontend: SpeechFrontend, device: str,
    n: int, rng: random.Random, augment_cfg: dict,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Augmentation-enabled equivalent of sample_replay_embeddings: instead
    of reading pre-cached (unaugmented) embeddings, re-loads raw audio for n
    random `train`-split rows via ASVspoof5Dataset, perturbs it (see
    data/augmentation.py), and embeds it live through the frozen frontend.
    Still cheap — a frozen-frontend forward pass over n short clips, not
    training — but requires the real ASVspoof5 `train` split's audio to be
    downloaded/extracted locally (not just its cached embeddings); raises a
    clear error via ASVspoof5Dataset's own check if it isn't."""
    dataset = ASVspoof5Dataset(data_cfg, model_cfg, split="train", random_crop=False)
    idx = rng.sample(range(len(dataset)), min(n, len(dataset)))
    waveforms, labels = zip(*(dataset[i] for i in idx))
    waveforms = augment_batch(torch.stack(waveforms), dataset.target_sr, rng, augment_cfg)
    replay_embeddings = _embed_waveforms(waveforms, frontend, device)
    replay_labels = torch.stack(labels)
    return replay_embeddings, replay_labels


def evaluate(model: Detector, loader: DataLoader, device: str) -> tuple:
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
    parser.add_argument("--generator-config", default="configs/generator.yaml")
    parser.add_argument("--adversarial-config", default="configs/adversarial.yaml")
    parser.add_argument("--generator-checkpoint", required=True, help="this round's fine-tuned transformer, from finetune_generator.py")
    parser.add_argument("--detector-checkpoint-in", required=True, help="previous round's detector_best.pt (or the original trained detector, for round 1)")
    parser.add_argument("--detector-checkpoint-out", required=True)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Per-side conditional routing (see graph/graph.py's decide_side_node): skip the "
             "classifier-head training loop entirely — the detector's weights are carried over UNCHANGED "
             "this round. Still generates fresh hard negatives from --generator-checkpoint (the round's "
             "generator, whether or not it was itself just fine-tuned this round) and reports a real "
             "fooling_rate/val_eer reading against the current, untouched detector, so the loop still has a "
             "genuine per-round metric to route on even on a round where only the generator side was active.",
    )
    args = parser.parse_args()

    data_cfg, model_cfg = load_configs(args.data_config, args.model_config)
    with open(args.generator_config) as f:
        gen_cfg = yaml.safe_load(f)
    with open(args.adversarial_config) as f:
        full_adv_cfg = yaml.safe_load(f)
    det_cfg = full_adv_cfg["detector_hardening"]
    # Augmentation section is optional — missing entirely (every config from
    # before this branch) or `enabled: false` both mean "exact original
    # behavior," so the augmentation-off baseline stays bit-for-bit
    # comparable to every prior real Colab run. See data/augmentation.py.
    augment_cfg = full_adv_cfg.get("augmentation", {"enabled": False})
    top_seed = full_adv_cfg.get("seed", 42)
    rng = random.Random(top_seed + 1000 + args.round)  # +1000: distinct stream from finetune_generator.py's rng, same round

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[harden-detector] round {args.round} | device: {device} | augmentation: "
          f"{'enabled' if augment_cfg.get('enabled') else 'disabled'} | mode: "
          f"{'EVAL-ONLY (detector weights unchanged this round)' if args.eval_only else 'train'}")

    # --- this round's fine-tuned generator (inference only, full 32-step quality) ---
    generator = GeneratorWrapper(
        model_name=gen_cfg["model"]["name"],
        target_sample_rate=gen_cfg["model"]["target_sample_rate"],
        device=device,
        transformer_checkpoint=args.generator_checkpoint,
    )
    hard_negative_paths = generate_hard_negatives(
        generator, data_cfg, gen_cfg, det_cfg,
        full_adv_cfg["paths"]["hard_negatives_dir"], det_cfg["n_hard_negatives"], rng,
    )
    print(f"[harden-detector] generated {len(hard_negative_paths)} hard negatives -> {full_adv_cfg['paths']['hard_negatives_dir']}")

    # --- frontend + previous detector checkpoint (warm start, not from scratch) ---
    frontend = SpeechFrontend(
        checkpoint=model_cfg["frontend"]["checkpoint"], freeze=model_cfg["frontend"]["freeze"],
    ).to(device)
    target_sr = model_cfg["frontend"]["target_sample_rate"]
    max_samples = int(model_cfg["frontend"]["max_duration_sec"] * target_sr)

    ckpt = torch.load(args.detector_checkpoint_in, map_location=device, weights_only=False)
    model = Detector(
        frontend=None,  # trained on precomputed embeddings below, same fast path as train_detector.py
        hidden_dim=ckpt["hidden_dim"],
        mlp_hidden=ckpt["classifier_config"]["mlp_hidden"],
        dropout=ckpt["classifier_config"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"[harden-detector] warm-started from {args.detector_checkpoint_in} (epoch {ckpt['epoch']}, prior val EER {ckpt.get('val_eer', float('nan')):.2%})")

    # --- build this round's small mixed training set ---
    # Hard negatives get augmented (if enabled) right here, after F5-TTS
    # synthesis and before embedding — the detector then also learns to spot
    # this round's generator under noisy/channel-varied conditions, not just
    # its clean default output.
    hard_negative_embeddings = embed_clips(
        hard_negative_paths, frontend, device, target_sr, max_samples, augment_cfg=augment_cfg, rng=rng,
    )
    hard_negative_labels = torch.zeros(hard_negative_embeddings.shape[0])  # spoof = 0

    if augment_cfg.get("enabled"):
        # Augmentation-enabled path: re-load raw `train`-split audio and embed
        # it live (perturbed) each round, instead of reading the unaugmented
        # cached embeddings — see sample_and_embed_replay's docstring for why.
        replay_embeddings, replay_labels = sample_and_embed_replay(
            data_cfg, model_cfg, frontend, device, det_cfg["n_replay_real"], rng, augment_cfg,
        )
    else:
        cache_prefix = os.path.join(model_cfg["paths"]["embeddings_cache_dir"], "train")
        if not os.path.exists(f"{cache_prefix}_meta.json"):
            raise FileNotFoundError(
                f"{cache_prefix}_meta.json not found — run training/extract_embeddings.py --split train first "
                "(needed for the real-embedding replay sample, see module docstring)."
            )
        replay_embeddings, replay_labels = sample_replay_embeddings(cache_prefix, det_cfg["n_replay_real"], rng)

    all_embeddings = torch.cat([hard_negative_embeddings, replay_embeddings], dim=0)
    all_labels = torch.cat([hard_negative_labels, replay_labels], dim=0)
    print(
        f"[harden-detector] training set: {hard_negative_embeddings.shape[0]} fresh hard negatives "
        f"+ {replay_embeddings.shape[0]} replayed real embeddings = {all_embeddings.shape[0]} total"
    )

    if args.eval_only:
        # Per-side conditional routing: this round's generator side was the
        # one that got attention (see finetune_generator.py / graph/graph.py's
        # decide_side_node) — the detector's weights stay exactly as loaded
        # from --detector-checkpoint-in. We still evaluate the CURRENT,
        # untouched detector against this round's freshly-built mixed set
        # (all of it, not a train/val split — there's no training to hold
        # anything out from) so val_eer_history still gets a real, comparable
        # per-round reading instead of a gap or a stale carried-over number.
        #
        # Same "All-NaN slice" failure mode the training path's stratified
        # split guards against below can hit here too, if this round's small
        # mix happens to have only one class — compute_eer's ROC curve is
        # undefined with no positive (or no negative) examples. Real
        # production round sizes (40 hard negatives + hundreds of replayed
        # real clips, mixed spoof/bonafide) make this very unlikely in
        # practice, but it costs nothing to guard rather than let a rare
        # unlucky round crash the whole loop.
        if len(set(all_labels.numpy().tolist())) > 1:
            full_loader = DataLoader(
                MixedEmbeddingDataset(all_embeddings, all_labels), batch_size=det_cfg["batch_size"], shuffle=False,
            )
            best_eer, eval_only_accuracy = evaluate(model, full_loader, device)
            print(
                f"[harden-detector] eval-only round — detector weights unchanged | "
                f"eval EER={best_eer:.2%} | eval acc={eval_only_accuracy:.2%} (against this round's fresh "
                f"{all_embeddings.shape[0]}-clip mix, no training)"
            )
        else:
            best_eer = ckpt.get("val_eer", float("nan"))
            print(
                f"[harden-detector] eval-only round — this round's mixed set has only one class present, "
                f"EER is undefined here; carrying forward the prior checkpoint's val_eer ({best_eer:.2%}) instead "
                "of crashing on an all-NaN ROC curve."
            )

        os.makedirs(os.path.dirname(args.detector_checkpoint_out), exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "hidden_dim": ckpt["hidden_dim"],
            "classifier_config": ckpt["classifier_config"],
            "epoch": ckpt["epoch"],  # unchanged — no training epochs ran this round
            "val_eer": best_eer,
            "round": args.round,
            "hardened_against_generator_checkpoint": args.generator_checkpoint,
            "eval_only": True,
        }, args.detector_checkpoint_out)
        print(f"[harden-detector] saved carried-over (unmodified) detector -> {args.detector_checkpoint_out}")
    else:
        # Stratified, not a plain random slice — this round's training set is
        # small (tens to low hundreds of rows, unlike the full cached train
        # split), so a naive random val split has a real chance of landing on a
        # single-class val set by chance, which makes compute_eer's ROC curve
        # undefined (caught via real-execution testing: "All-NaN slice
        # encountered" from a val split that came up all-spoof). Falls back to a
        # plain random split only in the degenerate case where the whole mixed
        # set has just one class already (val EER is meaningless either way then
        # — not something stratification can fix).
        label_values = all_labels.numpy()
        if len(set(label_values.tolist())) > 1:
            train_idx, val_idx = train_test_split(
                range(all_embeddings.shape[0]), test_size=det_cfg["val_fraction"],
                stratify=label_values, random_state=top_seed + args.round,
            )
        else:
            rng_split = random.Random(top_seed + args.round)
            idx = list(range(all_embeddings.shape[0]))
            rng_split.shuffle(idx)
            val_size = max(1, int(len(idx) * det_cfg["val_fraction"]))
            val_idx, train_idx = idx[:val_size], idx[val_size:]
        train_embeddings, val_embeddings = all_embeddings[train_idx], all_embeddings[val_idx]
        train_labels, val_labels = all_labels[train_idx], all_labels[val_idx]

        train_loader = DataLoader(
            MixedEmbeddingDataset(train_embeddings, train_labels), batch_size=det_cfg["batch_size"], shuffle=True,
        )
        val_loader = DataLoader(
            MixedEmbeddingDataset(val_embeddings, val_labels), batch_size=det_cfg["batch_size"], shuffle=False,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=det_cfg["learning_rate"])
        criterion = nn.BCEWithLogitsLoss()

        best_eer = float("inf")
        best_state_dict = model.state_dict()
        for epoch in range(1, det_cfg["epochs"] + 1):
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

            train_loss = total_loss / train_embeddings.shape[0]
            eer, accuracy = evaluate(model, val_loader, device)
            print(f"[harden-detector] epoch {epoch:02d}/{det_cfg['epochs']} | loss={train_loss:.4f} | val EER={eer:.2%} | val acc={accuracy:.2%}")
            if eer < best_eer:
                best_eer = eer
                best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state_dict)

        os.makedirs(os.path.dirname(args.detector_checkpoint_out), exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "hidden_dim": ckpt["hidden_dim"],
            "classifier_config": ckpt["classifier_config"],
            "epoch": ckpt["epoch"] + det_cfg["epochs"],
            "val_eer": best_eer,
            "round": args.round,
            "hardened_against_generator_checkpoint": args.generator_checkpoint,
            "eval_only": False,
        }, args.detector_checkpoint_out)
        print(f"[harden-detector] saved hardened detector -> {args.detector_checkpoint_out} (val EER {best_eer:.2%})")

    # --- report the new fooling rate: a FRESH held-out batch from the same
    # generator (not the clips just trained on) scored by the just-hardened
    # detector — this is what graph/nodes.py's conditional edge reads. ---
    eval_dir = full_adv_cfg["paths"]["hard_negatives_dir"] + "_eval"
    eval_paths = generate_hard_negatives(
        generator, data_cfg, gen_cfg, det_cfg, eval_dir, det_cfg["n_hard_negatives"], rng,
    )
    eval_detector = Detector(
        frontend=frontend, hidden_dim=ckpt["hidden_dim"],
        mlp_hidden=ckpt["classifier_config"]["mlp_hidden"], dropout=ckpt["classifier_config"]["dropout"],
    ).to(device)
    eval_detector.load_state_dict(model.state_dict(), strict=False)
    eval_detector.eval()
    scores = []
    with torch.no_grad():
        for path in eval_paths:
            waveform = _load_and_fix_length(path, target_sr, max_samples).unsqueeze(0).to(device)
            scores.append(torch.sigmoid(eval_detector(waveform)).item())
    fooling_rate = sum(1 for s in scores if s > 0.5) / len(scores) if scores else float("nan")

    result_path = args.detector_checkpoint_out + ".round_result.json"
    with open(result_path, "w") as f:
        json.dump({
            "round": args.round, "val_eer": best_eer, "fooling_rate": fooling_rate,
            "n_eval_clips": len(scores), "mean_p_bonafide_eval": sum(scores) / len(scores) if scores else float("nan"),
        }, f, indent=2)

    print(
        f"[harden-detector] round {args.round} new fooling rate (fresh held-out clips from this round's "
        f"generator, scored by the just-hardened detector): {fooling_rate:.1%} -> {result_path}"
    )


if __name__ == "__main__":
    main()
