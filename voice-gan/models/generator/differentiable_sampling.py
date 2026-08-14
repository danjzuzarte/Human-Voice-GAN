"""
Custom, gradient-enabled few-step ODE sampler for F5-TTS's CFM model.

F5-TTS's own `CFM.sample()` (confirmed by reading the actual source —
github.com/SWivid/F5-TTS, src/f5_tts/model/cfm.py) is decorated
`@torch.no_grad()`. It's inference-only by design: the library ships a
standard reconstruction-loss trainer (train/train.py) but no path to
backprop a custom loss through generation. There is no official way to
fine-tune F5-TTS against a detector's judgment using the library as-is.

This reimplements the same fixed-step Euler ODE integration CFM.sample()
does in its non-classifier-free-guidance branch (`cfg_strength < 1e-5`),
WITHOUT no_grad, and with far fewer steps than the usual 32 (`nfe_step`,
e.g. 8) — specifically so training/finetune_generator.py can backprop an
adversarial loss (derived from the detector's score on the generated audio)
all the way back into the transformer's weights. Trading step count for
gradient-trackability is a standard technique in adversarial diffusion /
flow-matching fine-tuning (e.g. adversarial diffusion distillation) — this
function is only ever used during training; real inference (generate_samples.py)
still uses F5-TTS's own full-quality 32-step sampler unmodified.

Because keeping every ODE step's activations in the autograd graph is what
makes backprop possible here, GPU memory scales with `steps` — this is the
actual reason `steps` is kept small (8, not 32) for training, not a
correctness requirement.
"""
import torch
import torch.nn.functional as F


def differentiable_sample(cfm_model, cond, text, duration, steps=8, reference_transformer=None):
    """
    cfm_model: a loaded f5_tts.model.CFM instance (models/generator/f5_tts_finetune.py).
        cfm_model.transformer must be in train() mode with requires_grad=True
        params — this is the model being fine-tuned.
    cond: [batch, samples] raw reference waveform, already at the model's
        target sample rate (mel-spec extraction happens inside).
    text: list[str], already pinyin-converted (see f5_tts.model.utils.convert_char_to_pinyin),
        matching the exact contract CFM.sample()'s own `text` argument expects.
    duration: int, target total mel-frame length (reference + generated).
    steps: number of Euler ODE steps — kept small (default 8) for training;
        real inference uses 32 via F5-TTS's own sampler.
    reference_transformer: optional frozen copy of the *original* pretrained
        transformer (see f5_tts_finetune.clone_frozen_reference). If given,
        also returns an anchor/distillation loss — the mean-squared
        difference between the trainable transformer's flow prediction and
        the frozen reference's, at every ODE step. This is what keeps
        adversarial fine-tuning from collapsing into audio that merely fools
        the detector without still sounding like real cloned speech; the
        reference side is always called under torch.no_grad() since it's
        frozen and only serves as a fixed target.

    Returns: (mel, cond_seq_len, anchor_loss)
        mel: [batch, frames, mel_dim] — full sequence (reference + generated
             portion), gradient-tracked w.r.t. cfm_model.transformer's params.
        cond_seq_len: int, the reference portion's frame count — callers
             slice mel[:, cond_seq_len:, :] to get just the generated audio.
        anchor_loss: scalar tensor (0.0 if reference_transformer is None).
    """
    from f5_tts.model.utils import lens_to_mask

    if cond.ndim == 2:
        cond = cfm_model.mel_spec(cond)
        cond = cond.permute(0, 2, 1)
        assert cond.shape[-1] == cfm_model.num_channels

    cond = cond.to(next(cfm_model.transformer.parameters()).dtype)

    batch, cond_seq_len, device = *cond.shape[:2], cond.device
    lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

    if isinstance(text, list):
        if cfm_model.vocab_char_map is not None:
            from f5_tts.model.utils import list_str_to_idx
            text = list_str_to_idx(text, cfm_model.vocab_char_map).to(device)
        else:
            from f5_tts.model.utils import list_str_to_tensor
            text = list_str_to_tensor(text).to(device)
        assert text.shape[0] == batch

    cond_mask = lens_to_mask(lens)

    duration_t = torch.full((batch,), duration, device=device, dtype=torch.long)
    duration_t = torch.maximum(torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration_t)
    max_duration = int(duration_t.amax().item())

    cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
    cond_mask_padded = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
    cond_mask_padded = cond_mask_padded.unsqueeze(-1)
    step_cond = torch.where(cond_mask_padded, cond, torch.zeros_like(cond))

    mask = lens_to_mask(duration_t, length=max_duration)

    x = torch.randn(batch, max_duration, cfm_model.num_channels, device=device, dtype=step_cond.dtype)
    t = torch.linspace(0, 1, steps + 1, device=device, dtype=step_cond.dtype)
    dt = 1.0 / steps

    anchor_loss = torch.zeros((), device=device, dtype=step_cond.dtype)

    for i in range(steps):
        dx = cfm_model.transformer(
            x=x, cond=step_cond, text=text, time=t[i], mask=mask,
            drop_audio_cond=False, drop_text=False,
        )
        if reference_transformer is not None:
            with torch.no_grad():
                dx_ref = reference_transformer(
                    x=x, cond=step_cond, text=text, time=t[i], mask=mask,
                    drop_audio_cond=False, drop_text=False,
                )
            anchor_loss = anchor_loss + F.mse_loss(dx, dx_ref.detach())
        x = x + dx * dt

    if reference_transformer is not None:
        anchor_loss = anchor_loss / steps

    mel = torch.where(cond_mask_padded, cond, x)
    return mel, cond_seq_len, anchor_loss
