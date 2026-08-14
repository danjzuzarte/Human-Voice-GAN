"""
Wraps a pretrained self-supervised speech model (wav2vec2 or WavLM) as a
frame-level feature extractor for the detector's back-end classifier.

MVP default: frozen frontend (no fine-tuning) — this makes the expensive
part (running a ~95M-parameter transformer over every utterance) a one-time
cost via training/extract_embeddings.py, rather than something that happens
on every training step. See configs/model.yaml `frontend.freeze`.

Correction (found via real-execution testing, not just import/--help smoke
tests): `freeze` must mean "frozen weights" (requires_grad=False on this
module's own parameters, already set below), NOT "no gradient flows through
this module at all". Early callers only ever invoked this frozen frontend
from inside a caller-level `torch.no_grad()` (extract_embeddings.py,
evaluate_generator.py's score_clips, harden_detector.py's embed_clips), so
those two meanings were indistinguishable until the adversarial loop needed
the second one: harden_generator (training/finetune_generator.py) backprops
an adversarial loss through the frozen detector — frontend included — to
reach the generator's weights, with no no_grad() at the call site. An
earlier version of forward() below wrapped the frozen path in its own
internal `torch.no_grad()`, which silently zeroed every gradient reaching
the generator in that path (confirmed by a real backward() call during
verification: `detector_input.requires_grad` was True going in,
`logits.requires_grad` was False coming out). Fixed by dropping the internal
no_grad() and relying solely on requires_grad=False — correct for both the
original frozen-embedding use case (frontend's own weights still don't
accumulate grad or update) and the adversarial loop (upstream gradient can
still pass through).
"""
import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, Wav2Vec2Model, WavLMModel

_MODEL_CLASSES = {
    "wav2vec2": Wav2Vec2Model,
    "wavlm": WavLMModel,
}


def _infer_model_family(checkpoint: str) -> str:
    return "wavlm" if "wavlm" in checkpoint.lower() else "wav2vec2"


class SpeechFrontend(nn.Module):
    """Raw waveform -> frame-level embeddings [batch, frames, hidden_dim]."""

    def __init__(self, checkpoint: str = "facebook/wav2vec2-base", freeze: bool = True):
        super().__init__()
        family = _infer_model_family(checkpoint)
        model_cls = _MODEL_CLASSES[family]
        self.model = model_cls.from_pretrained(checkpoint)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(checkpoint)
        self.hidden_dim = self.model.config.hidden_size
        self.freeze = freeze
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: [batch, samples] raw audio at 16kHz. Callers should run
        it through self.feature_extractor's normalization first (dataset.py
        does this) — the frontend itself expects already-normalized input,
        matching how HF speech models are normally used.

        No internal torch.no_grad() here even when frozen — see the module
        docstring's correction above. `freeze` already made this module's
        own parameters requires_grad=False in __init__, so it never
        accumulates gradient or gets updated by an optimizer either way;
        callers that want to skip building the autograd graph entirely for
        speed (eval-only inference, embedding extraction) should wrap the
        call in their own `with torch.no_grad():`, same as every existing
        caller already does."""
        return self.model(waveform).last_hidden_state  # [batch, frames, hidden_dim]

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            # Keep the frozen frontend in eval mode (dropout/batchnorm off)
            # even when the wrapping module is switched to train mode.
            self.model.eval()
        return self
