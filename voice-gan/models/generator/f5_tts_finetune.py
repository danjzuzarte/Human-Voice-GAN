"""
Low-level F5-TTS model loading for adversarial fine-tuning.

Bypasses the F5TTS API class in f5_tts_wrapper.py — that class is
inference-only (its .infer() calls CFM.sample(), which is @torch.no_grad()
and gives no access to the underlying transformer for gradient-based
training). This loads the exact same CFM model + Vocos vocoder the
inference API loads (same config, same pretrained checkpoint, same
auto-download from Hugging Face — mirrors what F5TTS.__init__ does
internally, confirmed by reading the actual source), but returns the raw
objects so training/finetune_generator.py can put the transformer in
train() mode, optimize it directly, and keep a frozen reference copy for
the anchor/distillation loss (see models/generator/differentiable_sampling.py).

Imports are lazy (inside functions), matching f5_tts_wrapper.py's pattern,
so this file can be imported without the f5-tts package installed.
"""
import copy

import torch


def load_f5tts_for_finetuning(model_name: str = "F5TTS_v1_Base", device: "str | None" = None):
    """Returns (cfm_model, vocoder, mel_spec_type, target_sample_rate).
    cfm_model.transformer is the trainable part; cfm_model.mel_spec and
    vocoder are used but not fine-tuned here — only the flow-matching
    transformer is treated as the adversarial "generator" being hardened,
    matching how a GAN generator is usually one specific trainable network,
    not the whole synthesis stack."""
    from importlib.resources import files

    from cached_path import cached_path
    from hydra.utils import get_class
    from omegaconf import OmegaConf

    from f5_tts.infer.utils_infer import load_model, load_vocoder

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{model_name}.yaml")))
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
    target_sample_rate = model_cfg.model.mel_spec.target_sample_rate

    vocoder = load_vocoder(mel_spec_type, False, "", device, None)
    vocoder.eval()
    for p in vocoder.parameters():
        p.requires_grad = False

    repo_name, ckpt_step, ckpt_type = "F5-TTS", 1250000, "safetensors"
    ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{model_name}/model_{ckpt_step}.{ckpt_type}"))

    cfm_model = load_model(
        model_cls, model_arc, ckpt_file, mel_spec_type, "", "euler", True, device
    )
    # load_model()/load_checkpoint() auto-casts to float16 whenever dtype=None is passed
    # (our case — mel_spec_type is "vocos", not "bigvgan") and the GPU supports it (compute
    # capability >= 7). That's fine for F5-TTS's own inference-only usage, but this module
    # exists specifically to enable real backprop through the transformer, and training in
    # pure float16 with no gradient scaling is a well-known instability source: found on a
    # real GPU run, adv_loss/anchor_loss went to NaN on step 2 (immediately after the first
    # optimizer.step()) and stayed NaN for the rest of the round, since NaN in a single
    # backward pass poisons AdamW's exp_avg/exp_avg_sq state permanently. Force bfloat16
    # instead — same ~2x memory savings vs float32, but bf16 keeps float32's full exponent
    # range (just less mantissa precision), so it doesn't share fp16's overflow fragility and
    # needs no loss scaler. Requires real bf16 support (Ampere/Ada or newer, compute
    # capability >= 8 — the A10G/L4/A100 runtimes Colab typically assigns all qualify); on an
    # older T4 (compute 7.5) it still runs correctly via software emulation, just slower.
    cfm_model = cfm_model.to(torch.bfloat16)
    return cfm_model, vocoder, mel_spec_type, target_sample_rate


def clone_frozen_reference(cfm_model):
    """A frozen, deep-copied transformer for the anchor/distillation loss —
    call this BEFORE loading any previous fine-tuning round's weights onto
    cfm_model.transformer, so the reference always anchors back to the
    original pretrained generator rather than accumulating drift across
    rounds. Keeps the fine-tuned model from degenerating into audio that
    merely fools the detector without still sounding like real cloned
    speech."""
    reference = copy.deepcopy(cfm_model.transformer)
    for p in reference.parameters():
        p.requires_grad = False
    reference.eval()
    return reference


def load_transformer_checkpoint(cfm_model, checkpoint_path: str, device: "str | None" = None):
    """Loads a previous adversarial-fine-tuning round's transformer weights
    onto cfm_model.transformer in place (e.g. resuming round N from round
    N-1's output). No-op helper, kept separate from load_f5tts_for_finetuning
    so callers can choose whether to start a round from the pretrained base
    or from a prior round's checkpoint."""
    state_dict = torch.load(checkpoint_path, map_location=device or "cpu")
    cfm_model.transformer.load_state_dict(state_dict)
    return cfm_model