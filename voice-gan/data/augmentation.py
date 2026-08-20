"""
Acoustic-domain augmentation for detector training audio — the noise/gain/
bandwidth perturbations applied to the adversarial loop's per-round training
batches when configs/adversarial.yaml's `augmentation.enabled` is true (see
training/harden_detector.py).

Motivation: the detector was trained only on ASVspoof5's studio-quality
real+spoof recordings, and one plausible reason F5-TTS's clean, studio-
quality clones fooled it 100% of the time in the baseline evaluation is that
the detector never learned to separate "clean recording" from "genuine
human speech" — if clean audio always meant bonafide during training, any
sufficiently clean generator output looks bonafide too, regardless of
whether it's actually synthetic. Randomizing the acoustic conditions the
detector trains on (background noise, channel gain, reduced bandwidth)
forces it to stop leaning on recording cleanliness as a shortcut.

This is deliberately simple, synthetic augmentation — built entirely from
torchaudio's real, installed `functional` API, verified directly against it
(see the module's own smoke test below) rather than assumed:
  - additive Gaussian noise at a randomized SNR (torchaudio.functional.add_noise)
  - a randomized gain adjustment (torchaudio.functional.gain)
  - an optional lowpass filter (torchaudio.functional.lowpass_biquad),
    simulating a narrower, non-studio recording/transmission channel
No external noise corpus (e.g. MUSAN) is used — not available in this
pipeline; synthetic Gaussian noise is the cheap, dependency-free stand-in.

Applied independently per clip, each round, at a configurable probability —
not to every clip — so the detector still sees a mix of clean and perturbed
audio, matching standard data-augmentation practice rather than replacing
the training distribution outright.

Scope note: this is a "loop-only" augmentation branch — it perturbs the
adversarial loop's per-round hard negatives AND its replayed real audio
(training/harden_detector.py), not the original detector-training run.
`detector_best.pt` (the shared starting checkpoint for every branch) is
untouched.
"""
import random

import torch
import torchaudio


def augment_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    rng: random.Random,
    apply_prob: float = 0.5,
    snr_db_range: "tuple[float, float]" = (5.0, 20.0),
    gain_db_range: "tuple[float, float]" = (-6.0, 6.0),
    lowpass_prob: float = 0.3,
    lowpass_cutoff_hz_range: "tuple[float, float]" = (2000.0, 7000.0),
) -> torch.Tensor:
    """Randomized acoustic-domain augmentation for a single mono waveform,
    shape [samples]. No-ops (returns the input unchanged) with probability
    `1 - apply_prob`, so not every clip is perturbed.

    All randomized *choices* (whether to apply, SNR, gain, cutoff frequency)
    are drawn from `rng` — pass the caller's own per-round `random.Random` so
    augmentation stays reproducible round-over-round the same way everything
    else in the adversarial loop already is (see harden_detector.py's `rng`
    construction, offset by `--round`). The actual noise *samples* are drawn
    from torch's global RNG (unseeded per call) — consistent with how the
    rest of this project already treats torch-level stochastic ops (e.g.
    dropout); only the hyperparameter choices are made reproducible.
    """
    if rng.random() > apply_prob:
        return waveform

    out = waveform.unsqueeze(0)  # torchaudio.functional ops expect [..., time]

    snr_db = rng.uniform(*snr_db_range)
    noise = torch.randn_like(out)
    out = torchaudio.functional.add_noise(out, noise, torch.tensor([snr_db]))

    gain_db = rng.uniform(*gain_db_range)
    out = torchaudio.functional.gain(out, gain_db)

    if rng.random() < lowpass_prob:
        cutoff_hz = rng.uniform(*lowpass_cutoff_hz_range)
        out = torchaudio.functional.lowpass_biquad(out, sample_rate, cutoff_hz)

    return out.squeeze(0)


def augment_batch(waveforms: torch.Tensor, sample_rate: int, rng: random.Random, cfg: dict) -> torch.Tensor:
    """Applies augment_waveform independently to every row of a batched
    waveform tensor, shape [batch, samples]. `cfg` is
    configs/adversarial.yaml's `augmentation` section (already resolved to a
    dict — see harden_detector.py's main())."""
    augmented = [
        augment_waveform(
            waveforms[i], sample_rate, rng,
            apply_prob=cfg.get("apply_prob", 0.5),
            snr_db_range=tuple(cfg.get("snr_db_range", (5.0, 20.0))),
            gain_db_range=tuple(cfg.get("gain_db_range", (-6.0, 6.0))),
            lowpass_prob=cfg.get("lowpass_prob", 0.3),
            lowpass_cutoff_hz_range=tuple(cfg.get("lowpass_cutoff_hz_range", (2000.0, 7000.0))),
        )
        for i in range(waveforms.shape[0])
    ]
    return torch.stack(augmented)


if __name__ == "__main__":
    # Quick real-execution smoke test (real torchaudio, toy audio) — run
    # directly with `python data/augmentation.py`. Not a pytest suite (this
    # project doesn't have one), matches the project's existing pattern of
    # verifying against real installed library code rather than assuming API
    # shape.
    rng = random.Random(0)
    sr = 16000
    wave = torch.randn(sr * 3)  # 3s of "audio"
    out = augment_waveform(wave, sr, rng, apply_prob=1.0, lowpass_prob=1.0)
    assert out.shape == wave.shape, f"shape changed: {wave.shape} -> {out.shape}"
    assert not torch.isnan(out).any(), "augmentation produced NaNs"
    assert not torch.equal(out, wave), "apply_prob=1.0 but waveform is unchanged"

    batch = torch.randn(4, sr * 3)
    cfg = {"apply_prob": 1.0, "lowpass_prob": 1.0}
    out_batch = augment_batch(batch, sr, rng, cfg)
    assert out_batch.shape == batch.shape
    assert not torch.isnan(out_batch).any()

    # apply_prob=0.0 must be a true no-op (bit-identical), not just "close"
    out_noop = augment_waveform(wave, sr, rng, apply_prob=0.0)
    assert torch.equal(out_noop, wave), "apply_prob=0.0 should be an exact no-op"

    print("[augmentation] smoke test passed: shapes preserved, no NaNs, apply_prob=0/1 behave correctly")
