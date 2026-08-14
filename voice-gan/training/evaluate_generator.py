"""
Scores `generate_samples.py`'s output against the trained detector — the
baseline "how detectable is an off-the-shelf generator" number. See
ARCHITECTURE.md for the full roadmap.

Unlike training/evaluate_detector.py (which scores pre-cached embeddings),
this runs the real frontend inline over each generated .wav, since these
clips were never part of the embeddings cache. Also scores a same-sized
sample of real dev bonafide and real dev spoof clips for comparison, so the
generated-clip numbers have real context instead of a number in isolation.

Usage:
    python training/generate_samples.py --generator-config configs/generator.yaml   # run first
    python training/evaluate_generator.py --data-config configs/data.yaml \\
        --model-config configs/model.yaml --generator-config configs/generator.yaml
"""
import argparse
import json
import os
import random
import sys

import torch
import torchaudio
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import _index_flac_files, load_configs, load_protocol_df
from models.detector.classifier import Detector
from models.detector.frontend import SpeechFrontend


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


def score_clips(paths: "list[str]", model: Detector, device: str, target_sr: int, max_samples: int, batch_size: int = 16) -> "list[float]":
    """Returns P(bonafide) for each path, via the real frontend + classifier."""
    scores = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start:start + batch_size]
            batch = torch.stack([_load_and_fix_length(p, target_sr, max_samples) for p in batch_paths]).to(device)
            logits = model(batch)
            scores.extend(torch.sigmoid(logits).cpu().tolist())
    return scores


def sample_comparison_clips(data_cfg: dict, split: str, label_value, n: int, seed: int) -> "list[str]":
    """n real clips of the given label (spoof or bonafide) from `split`,
    for context alongside the generated-clip scores."""
    df = load_protocol_df(data_cfg, split=split)
    label_col = data_cfg["label_column"]
    subset = df[df[label_col] == label_value]

    extracted_dir = data_cfg["paths"]["extracted_dir"]
    flac_index = _index_flac_files(extracted_dir)
    matched = subset[subset["filename"].isin(flac_index.keys())]
    if len(matched) == 0:
        return []
    sampled = matched.sample(min(n, len(matched)), random_state=seed)
    return [flac_index[fn] for fn in sampled["filename"]]


def summarize(name: str, scores: "list[float]") -> dict:
    if not scores:
        return {"name": name, "n": 0}
    fooled = sum(1 for s in scores if s > 0.5) / len(scores)
    mean_score = sum(scores) / len(scores)
    return {
        "name": name,
        "n": len(scores),
        "mean_p_bonafide": mean_score,
        "fraction_scored_above_0.5": fooled,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--generator-config", default="configs/generator.yaml")
    parser.add_argument("--checkpoint", default=None, help="defaults to <checkpoint_dir>/detector_best.pt")
    args = parser.parse_args()

    data_cfg, model_cfg = load_configs(args.data_config, args.model_config)
    with open(args.generator_config) as f:
        gen_cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[evaluate-generator] device: {device}")

    manifest_path = os.path.join(gen_cfg["paths"]["output_dir"], "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"{manifest_path} not found — run training/generate_samples.py first.")
    with open(manifest_path) as f:
        manifest = json.load(f)
    generated_paths = [item["output_path"] for item in manifest]
    print(f"[evaluate-generator] loaded manifest: {len(generated_paths)} generated clips")

    target_sr = model_cfg["frontend"]["target_sample_rate"]
    max_samples = int(model_cfg["frontend"]["max_duration_sec"] * target_sr)

    frontend = SpeechFrontend(
        checkpoint=model_cfg["frontend"]["checkpoint"],
        freeze=model_cfg["frontend"]["freeze"],
    ).to(device)

    checkpoint_path = args.checkpoint or os.path.join(model_cfg["paths"]["checkpoint_dir"], "detector_best.pt")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Detector(
        frontend=frontend,
        hidden_dim=ckpt["hidden_dim"],
        mlp_hidden=ckpt["classifier_config"]["mlp_hidden"],
        dropout=ckpt["classifier_config"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)  # strict=False: checkpoint has no frontend weights (it was trained frontend=None)
    print(f"[evaluate-generator] loaded checkpoint from {checkpoint_path} (trained epoch {ckpt['epoch']})")

    generated_scores = score_clips(generated_paths, model, device, target_sr, max_samples)

    seed = gen_cfg["cloning"]["seed"]
    n_compare = len(generated_paths)
    reference_split = gen_cfg["cloning"]["reference_split"]
    real_bonafide_paths = sample_comparison_clips(data_cfg, reference_split, data_cfg["bonafide_value"], n_compare, seed)
    real_spoof_paths = sample_comparison_clips(data_cfg, reference_split, "spoof", n_compare, seed)
    real_bonafide_scores = score_clips(real_bonafide_paths, model, device, target_sr, max_samples) if real_bonafide_paths else []
    real_spoof_scores = score_clips(real_spoof_paths, model, device, target_sr, max_samples) if real_spoof_paths else []

    results = [
        summarize("generated (F5-TTS clones)", generated_scores),
        summarize(f"real bonafide ({reference_split})", real_bonafide_scores),
        summarize(f"real spoof ({reference_split})", real_spoof_scores),
    ]

    print(f"\n[evaluate-generator] === baseline generator detectability ===")
    for r in results:
        if r["n"] == 0:
            print(f"[evaluate-generator] {r['name']}: no clips available for comparison")
            continue
        print(
            f"[evaluate-generator] {r['name']:30s} n={r['n']:3d}  "
            f"mean P(bonafide)={r['mean_p_bonafide']:.3f}  "
            f"scored-as-bonafide={r['fraction_scored_above_0.5']:.1%}"
        )
    gen_fooled = results[0]["fraction_scored_above_0.5"] if results[0]["n"] else float("nan")
    print(
        f"\n[evaluate-generator] fooling rate (fraction of off-the-shelf F5-TTS clones the detector "
        f"scored as bonafide): {gen_fooled:.1%} — this is the baseline number. Compare it "
        f"against the real-bonafide/real-spoof rows above for context: a detector with no useful "
        f"signal at all would score generated clips similarly to real bonafide; a detector that's "
        f"actually catching this generator will score them closer to real spoof."
    )


if __name__ == "__main__":
    main()
