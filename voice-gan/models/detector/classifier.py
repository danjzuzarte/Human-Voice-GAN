"""
Back-end classifier: attentive pooling over frame embeddings + a small MLP
head. This is the MVP backbone — a lighter alternative to a full AASIST
graph-attention network, chosen to get a working, trainable detector fast.
Swap in a fancier back end later without touching the frontend or training
script, as long as it keeps the same [batch, frames, hidden_dim] -> logit
interface.

Label convention used throughout this project: 1 = bonafide, 0 = spoof.
sigmoid(logit) = P(bonafide).
"""
import torch
import torch.nn as nn


class AttentivePooling(nn.Module):
    """Learns a per-frame attention weight, then takes a weighted sum over
    frames — lets the model learn to focus on the most discriminative parts
    of an utterance instead of averaging everything equally."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # embeddings: [batch, frames, hidden_dim]
        weights = torch.softmax(self.attention(embeddings), dim=1)  # [batch, frames, 1]
        pooled = (embeddings * weights).sum(dim=1)  # [batch, hidden_dim]
        return pooled


class ClassifierHead(nn.Module):
    """Pooled embedding -> single logit."""

    def __init__(self, hidden_dim: int, mlp_hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.pool = AttentivePooling(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(embeddings)
        logit = self.mlp(pooled).squeeze(-1)  # [batch]
        return logit


class Detector(nn.Module):
    """Full model: optional frontend + classifier head.

    Pass `frontend=None` for the fast path — training directly on cached
    embeddings (see training/extract_embeddings.py), which is the default
    for the detector MVP. Pass a real SpeechFrontend once you need to run on
    raw waveforms (inference on new audio, or fine-tuning the frontend
    later)."""

    def __init__(self, frontend: "nn.Module | None", hidden_dim: int, mlp_hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.frontend = frontend
        self.head = ClassifierHead(hidden_dim, mlp_hidden, dropout)

    def forward(self, waveform_or_embeddings: torch.Tensor) -> torch.Tensor:
        if self.frontend is not None:
            embeddings = self.frontend(waveform_or_embeddings)
        else:
            embeddings = waveform_or_embeddings
        return self.head(embeddings)
