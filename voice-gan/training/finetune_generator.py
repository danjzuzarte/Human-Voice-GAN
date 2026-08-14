"""
Real, weight-space adversarial fine-tuning of the F5-TTS generator against
the current detector — the `harden_generator` step of the LangGraph
adversarial loop (see graph/nodes.py, ARCHITECTURE.md).

This is NOT the standard reconstruction-loss fine-tuning F5-TTS's own
train/finetune_cli.py does. It backprops a loss derived from the detector's
own judgment through generation and into the F5-TTS transformer's weights,
using a custom differentiable few-step sampler (models/generator/
differentiable_sampling.py) since F5-TTS's official sampler is
@torch.no_grad() and gives no path for this. Two loss terms per step:

  - adversarial_loss: BCE pushing the frozen detector's score on the
    generated audio toward "bonafide" (1) — this is the actual adversarial
    pressure.
  - anchor_loss: MSE between the trainable transformer's flow prediction and
    a frozen reference copy of the *original* pretrained transformer's
    prediction, at every ODE step — keeps fine-tuning from collapsing into
    audio that merely fools the detector without still sounding like real
    cloned speech.

total_loss = adversarial_loss + anchor_weight * anchor_loss

Only the transformer is updated (via AdamW). The mel-spec extractor, vocoder,
and the entire detector are frozen throughout — gradients flow through the
frozen detector (needed to reach the generator) but never update it; that
happens separately in training/harden_detector.py.

Usage:
    python training/finetune_generator.py --data-config configs/data.yaml \\
        --model-config configs/model.yaml --generator-config configs/generator.yaml \\
        --adversarial-config configs/adversarial.yaml \\
        --detector-checkpoint <path to this round's detector_best.pt> \\
        --generator-checkpoint-out <path to save the fine-tuned transformer> \\
        [--generator-checkpoint-in <path to a previous round's checkpoint to resume from>] \\
        --round 1
"""
import argparse
import json
import os
import random
import sys

import torch
import torch.nn as nn
import torchaudio
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import load_configs
from models.detector.classifier import Detector
from models.detector.frontend import SpeechFrontend
from models.generator.differentiable_sampling import differentiable_sample
from models.generator.f5_tts_finetune import (
    clone_frozen_reference,
    load_f5tts_for_finetuning,
    load_transformer_checkpoint,
)
from training.generate_samples import load_target_sentences, sample_reference_rows


def _load_fixed_length(path: str, target_sr: int, duration_sec: float) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    waveform = waveform.squeeze(0)
    target_samples = int(duration_sec * target_sr)
    n = waveform.shape[-1]
    if n >= target_samples:
        return waveform[:target_samples]
    return torch.nn.functional.pad(waveform, (0, target_samples - n))


def _load_and_fix_length_for_detector(waveform: torch.Tensor, native_sr: int, target_sr: int, max_samples: int) -> torch.Tensor:
    """Resamples a generated waveform (native F5-TTS rate, e.g. 24kHz) to the
    detector's expected rate and fixes it to the detector's expected window
    length — kept differentiable (no .detach()/.numpy() round trip)."""
    if native_sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, native_sr, target_sr)
    n = waveform.shape[-1]
    if n >= max_samples:
        return waveform[:max_samples]
    return torch.nn.functional.pad(waveform, (0, max_samples - n))


def build_batch(reference_rows: list, sentences: list, batch_size: int, reference_duration_sec: float,
                 native_sr: int, rng: random.Random) -> "tuple[torch.Tensor, list]":
    """Samples `batch_size` (reference audio, ref_text placeholder, gen_text)
    items. Reference audio auto-transcription (ref_text="") happens once per
    item via F5-TTS's own ASR utility — cached by F5-TTS internally per
    reference file, so repeated rounds over the same references are cheap
    after the first."""
    from f5_tts.infer.utils_infer import preprocess_ref_audio_text

    chosen = rng.sample(reference_rows, min(batch_size, len(reference_rows)))
    cond_list, text_list = [], []
    for i, ref in enumerate(chosen):
        gen_text = sentences[rng.randrange(len(sentences))]
        ref_path, ref_text = preprocess_ref_audio_text(ref["path"], "", show_info=lambda *a, **k: None)
        cond_list.append(_load_fixed_length(ref_path, native_sr, reference_duration_sec))
        text_list.append(ref_text + gen_text)

    cond = torch.stack(cond_list)
    return cond, text_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--generator-config", default="configs/generator.yaml")
    parser.add_argument("--adversarial-config", default="configs/adversarial.yaml")
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument("--generator-checkpoint-out", required=True)
    parser.add_argument("--generator-checkpoint-in", default=None)
    parser.add_argument("--round", type=int, default=1)
    args = parser.parse_args()

    data_cfg, model_cfg = load_configs(args.data_config, args.model_config)
    with open(args.generator_config) as f:
        gen_cfg = yaml.safe_load(f)
    with open(args.adversarial_config) as f:
        full_adv_cfg = yaml.safe_load(f)
    adv_cfg = full_adv_cfg["generator_finetuning"]
    top_seed = full_adv_cfg.get("seed", 42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[finetune-generator] round {args.round} | device: {device}")

    # --- generator (trainable) ---
    cfm_model, vocoder, mel_spec_type, native_sr = load_f5tts_for_finetuning(gen_cfg["model"]["name"], device)
    reference_transformer = clone_frozen_reference(cfm_model)  # anchor: always the ORIGINAL pretrained weights
    if args.generator_checkpoint_in:
        load_transformer_checkpoint(cfm_model, args.generator_checkpoint_in, device)
        print(f"[finetune-generator] resumed generator from {args.generator_checkpoint_in}")
    cfm_model.transformer.train()
    for p in cfm_model.transformer.parameters():
        p.requires_grad = True

    # --- detector (frozen, but stays in the autograd graph) ---
    frontend = SpeechFrontend(
        checkpoint=model_cfg["frontend"]["checkpoint"], freeze=model_cfg["frontend"]["freeze"],
    ).to(device)
    ckpt = torch.load(args.detector_checkpoint, map_location=device, weights_only=False)
    detector = Detector(
        frontend=frontend,
        hidden_dim=ckpt["hidden_dim"],
        mlp_hidden=ckpt["classifier_config"]["mlp_hidden"],
        dropout=ckpt["classifier_config"]["dropout"],
    ).to(device)
    detector.load_state_dict(ckpt["model_state_dict"], strict=False)
    detector.eval()
    for p in detector.parameters():
        p.requires_grad = False
    print(f"[finetune-generator] loaded detector from {args.detector_checkpoint} (epoch {ckpt['epoch']})")

    # --- data ---
    reference_rows = sample_reference_rows(data_cfg, gen_cfg)
    sentences = load_target_sentences(gen_cfg["cloning"]["target_sentences_file"])
    rng = random.Random(top_seed + args.round)

    detector_sr = model_cfg["frontend"]["target_sample_rate"]
    detector_max_samples = int(model_cfg["frontend"]["max_duration_sec"] * detector_sr)
    reference_duration_sec = adv_cfg["reference_duration_sec"]
    generation_duration_sec = adv_cfg["generation_duration_sec"]

    from f5_tts.infer.utils_infer import hop_length
    duration = int(generation_duration_sec * native_sr / hop_length)

    optimizer = torch.optim.AdamW(cfm_model.transformer.parameters(), lr=adv_cfg["generator_lr"])
    bce = nn.BCEWithLogitsLoss()

    history = []
    for step in range(1, adv_cfg["adversarial_steps"] + 1):
        cond, text_list = build_batch(
            reference_rows, sentences, adv_cfg["batch_size"], reference_duration_sec, native_sr, rng,
        )
        cond = cond.to(device)

        from f5_tts.model.utils import convert_char_to_pinyin
        pinyin_text = convert_char_to_pinyin(text_list)

        mel, cond_seq_len, anchor_loss = differentiable_sample(
            cfm_model, cond, pinyin_text, duration,
            steps=adv_cfg["nfe_step"], reference_transformer=reference_transformer,
        )
        generated_mel = mel[:, cond_seq_len:, :].permute(0, 2, 1)
        generated_mel = generated_mel.to(torch.float32)
        generated_wave = vocoder.decode(generated_mel)  # [batch, samples] at native_sr, gradient-tracked

        batch = generated_wave.shape[0]
        detector_input = torch.stack([
            _load_and_fix_length_for_detector(generated_wave[i], native_sr, detector_sr, detector_max_samples)
            for i in range(batch)
        ])

        logits = detector(detector_input)
        target = torch.ones_like(logits)  # push toward "bonafide"
        adversarial_loss = bce(logits, target)

        total_loss = adversarial_loss + adv_cfg["anchor_weight"] * anchor_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(cfm_model.transformer.parameters(), adv_cfg.get("max_grad_norm", 1.0))
        optimizer.step()

        with torch.no_grad():
            mean_p_bonafide = torch.sigmoid(logits).mean().item()
        history.append({
            "step": step, "adversarial_loss": adversarial_loss.item(),
            "anchor_loss": anchor_loss.item(), "total_loss": total_loss.item(),
            "mean_p_bonafide": mean_p_bonafide,
        })
        print(
            f"[finetune-generator] step {step:03d}/{adv_cfg['adversarial_steps']} | "
            f"adv_loss={adversarial_loss.item():.4f} | anchor_loss={anchor_loss.item():.4f} | "
            f"mean P(bonafide)={mean_p_bonafide:.3f}"
        )

    os.makedirs(os.path.dirname(args.generator_checkpoint_out), exist_ok=True)
    torch.save(cfm_model.transformer.state_dict(), args.generator_checkpoint_out)

    meta_path = args.generator_checkpoint_out + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump({"round": args.round, "history": history}, f, indent=2)

    print(f"[finetune-generator] done. saved -> {args.generator_checkpoint_out} (meta: {meta_path})")


if __name__ == "__main__":
    main()