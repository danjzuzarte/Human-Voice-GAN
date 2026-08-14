"""
Generates a baseline set of voice-cloned "fake" utterances with a
pretrained, off-the-shelf F5-TTS checkpoint (no fine-tuning) — the starting
line for "how detectable is an off-the-shelf generator". See ARCHITECTURE.md
for the full roadmap.

For each of `cloning.n_speakers` sampled bonafide reference utterances from
the ASVspoof5 `cloning.reference_split` (dev by default — deliberately not
train, so the generated set doesn't overlap anything the detector trained
on), clones that speaker's voice reading a sentence drawn from a small fixed
pool of generic target sentences (`cloning.target_sentences_file` —
ASVspoof5's protocol files carry no text transcripts, so target text can't
come from the dataset itself). Writes each generated .wav plus a
`manifest.json` describing what was generated from what.

Usage:
    python training/generate_samples.py --data-config configs/data.yaml \\
        --model-config configs/model.yaml --generator-config configs/generator.yaml
"""
import argparse
import json
import os
import random
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import _index_flac_files, load_configs, load_protocol_df


def load_target_sentences(path: str) -> list:
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        raise ValueError(f"{path} contained no non-empty lines.")
    return lines


def sample_reference_rows(data_cfg: dict, gen_cfg: dict) -> "list[dict]":
    """Picks `n_speakers` bonafide rows (one per distinct speaker where
    possible) from `reference_split`, matched against actually-downloaded
    audio. Returns a list of {speaker_id, filename, path} dicts."""
    split = gen_cfg["cloning"]["reference_split"]
    n = gen_cfg["cloning"]["n_speakers"]
    seed = gen_cfg["cloning"]["seed"]

    df = load_protocol_df(data_cfg, split=split)
    label_col = data_cfg["label_column"]
    bonafide_val = data_cfg["bonafide_value"]
    bonafide_df = df[df[label_col] == bonafide_val]

    extracted_dir = data_cfg["paths"]["extracted_dir"]
    flac_index = _index_flac_files(extracted_dir)
    matched = bonafide_df[bonafide_df["filename"].isin(flac_index.keys())]
    if len(matched) == 0:
        raise ValueError(
            f"No bonafide '{split}' protocol rows matched any downloaded .flac "
            f"file under {extracted_dir}. Has the '{split}' shard been downloaded?"
        )

    # Prefer one row per distinct speaker_id, so references aren't all the
    # same voice — falls back to repeating speakers only if there aren't
    # enough distinct ones in the matched (downloaded) subset.
    #
    # Note: this deliberately avoids `groupby(...).apply(lambda g: g.sample(1))`
    # to pick one row per group — as of pandas 3.0, groupby.apply excludes the
    # grouping column from what's passed to/returned by the function by
    # default (and can no longer be overridden via include_groups=True), which
    # silently drops speaker_id from the result. Shuffle-then-drop-duplicates
    # achieves the same "one random row per speaker" result without relying on
    # that version-fragile behavior.
    shuffled = matched.sample(frac=1, random_state=seed)
    by_speaker = shuffled.drop_duplicates(subset="speaker_id", keep="first")

    rng = random.Random(seed)
    speaker_ids = list(by_speaker["speaker_id"])
    rng.shuffle(speaker_ids)

    if len(by_speaker) >= n:
        chosen_ids = set(speaker_ids[:n])
        chosen = by_speaker[by_speaker["speaker_id"].isin(chosen_ids)]
    else:
        # Not enough distinct speakers in the matched subset — top up with
        # extra rows (possibly repeating a speaker) rather than erroring.
        chosen = by_speaker
        remaining = matched[~matched.index.isin(by_speaker.index)]
        extra_needed = n - len(chosen)
        if extra_needed > 0 and len(remaining) > 0:
            extra = remaining.sample(min(extra_needed, len(remaining)), random_state=seed)
            chosen = pd.concat([chosen, extra], ignore_index=True)

    rows = []
    for _, row in chosen.iterrows():
        rows.append({
            "speaker_id": row["speaker_id"],
            "filename": row["filename"],
            "path": flac_index[row["filename"]],
        })
    return rows[:n]


def build_generator(gen_cfg: dict):
    """Factory, isolated so tests can monkeypatch this instead of loading
    the real (heavy, GPU/HF-download-requiring) F5-TTS checkpoint."""
    from models.generator.f5_tts_wrapper import GeneratorWrapper
    return GeneratorWrapper(
        model_name=gen_cfg["model"]["name"],
        target_sample_rate=gen_cfg["model"]["target_sample_rate"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--generator-config", default="configs/generator.yaml")
    args = parser.parse_args()

    data_cfg, model_cfg = load_configs(args.data_config, args.model_config)
    import yaml
    with open(args.generator_config) as f:
        gen_cfg = yaml.safe_load(f)

    reference_rows = sample_reference_rows(data_cfg, gen_cfg)
    sentences = load_target_sentences(gen_cfg["cloning"]["target_sentences_file"])
    print(f"[generate] {len(reference_rows)} reference speakers, {len(sentences)} target sentences")

    generator = build_generator(gen_cfg)

    output_dir = gen_cfg["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    manifest = []
    for i, ref in enumerate(reference_rows):
        gen_text = sentences[i % len(sentences)]
        out_name = f"gen_{i:03d}_{ref['speaker_id']}.wav"
        out_path = os.path.join(output_dir, out_name)

        generator.clone(ref_audio_path=ref["path"], gen_text=gen_text, output_path=out_path)

        manifest.append({
            "index": i,
            "speaker_id": ref["speaker_id"],
            "reference_file": ref["path"],
            "gen_text": gen_text,
            "output_path": out_path,
        })
        print(f"[generate] {i + 1}/{len(reference_rows)} -> {out_path}")

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[generate] done. {len(manifest)} clips -> {output_dir} (manifest: {manifest_path})")


if __name__ == "__main__":
    main()
