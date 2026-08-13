"""
Wraps a pretrained self-supervised speech model (wav2vec2 or WavLM) as a
frame-level feature extractor for the detector's back-end classifier.

MVP default: frozen frontend (no fine-tuning) — this makes the expensive
part (running a ~95M-parameter transformer over every utterance) a one-time
cost via training/extract_embeddings.py, rather than something that happens
on every training step. See configs/model.yaml `frontend.freeze`.
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
        matching how HF speech models are normally used."""
        if self.freeze:
            with torch.no_grad():
                out = self.model(waveform)
        else:
            out = self.model(waveform)
        return out.last_hidden_state  # [batch, frames, hidden_dim]

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            # Keep the frozen frontend in eval mode (dropout/batchnorm off)
            # even when the wrapping module is switched to train mode.
            self.model.eval()
        return self
