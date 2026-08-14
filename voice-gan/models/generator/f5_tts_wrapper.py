"""
Thin wrapper around the pretrained F5-TTS zero-shot voice-cloning model
(github.com/SWivid/F5-TTS, checkpoint via Hugging Face `SWivid/F5-TTS`).

MVP: off-the-shelf inference — clone a reference speaker's voice and have it
say new text, to get a baseline "how detectable is an off-the-shelf
generator" number against the trained detector. See ARCHITECTURE.md for the
full roadmap.

`transformer_checkpoint` lets this wrapper load an adversarially-fine-tuned
transformer (see training/finetune_generator.py,
models/generator/f5_tts_finetune.py) on top of the pretrained base, so a
given round's hardened generator can be evaluated with F5-TTS's own
full-quality 32-step inference path — the differentiable few-step sampler
used during training (models/generator/differentiable_sampling.py) is only
for backprop, never for the audio that actually gets scored/listened to.

The `f5-tts` package (`pip install f5-tts`) is imported lazily inside
`GeneratorWrapper.__init__`, not at module load time, so this file can be
imported (and its surrounding logic tested) in environments — like a
sandbox with no GPU — where the package isn't installed and doesn't need
to be. Actual generation requires it.
"""
import os

import torch
import torchaudio


class GeneratorWrapper:
    """Loads a pretrained F5-TTS checkpoint and exposes a single `clone()`
    call: reference audio (+ optional reference text) + target text ->
    a waveform tensor at `target_sample_rate`, resampled from whatever F5-TTS
    natively outputs (24kHz as of the v1 base checkpoint) to match the
    detector's expected input rate."""

    def __init__(
        self,
        model_name: str = "F5TTS_v1_Base",
        target_sample_rate: int = 16000,
        device: "str | None" = None,
        transformer_checkpoint: "str | None" = None,
    ):
        from f5_tts.api import F5TTS  # lazy import — see module docstring

        self.target_sample_rate = target_sample_rate
        self._f5tts = F5TTS(model=model_name, device=device)

        if transformer_checkpoint is not None:
            state_dict = torch.load(transformer_checkpoint, map_location=device)
            self._f5tts.ema_model.transformer.load_state_dict(state_dict)

    def clone(self, ref_audio_path: str, gen_text: str, ref_text: str = "", output_path: "str | None" = None) -> torch.Tensor:
        """ref_text="" (the default) triggers F5-TTS's built-in ASR step to
        auto-transcribe the reference audio — used here since ASVspoof5's
        protocol files carry no text transcripts to pass in directly.

        Returns a [samples] float32 waveform tensor at `self.target_sample_rate`.
        If `output_path` is given, also writes the (resampled) audio there."""
        wav, sr, _spec = self._f5tts.infer(
            ref_file=ref_audio_path,
            ref_text=ref_text,
            gen_text=gen_text,
            file_wave=None,
            file_spec=None,
        )
        waveform = torch.as_tensor(wav, dtype=torch.float32)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if sr != self.target_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sample_rate)
        waveform = waveform.squeeze(0)

        if output_path is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(output_path, waveform.unsqueeze(0), self.target_sample_rate)

        return waveform
