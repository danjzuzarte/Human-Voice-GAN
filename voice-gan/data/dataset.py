"""
PyTorch Dataset for ASVspoof5 utterances: joins the protocol dataframe
(speaker/filename/label) with the actual extracted .flac files, loads audio,
and applies a fixed-duration crop/pad so every item comes out the same
length — this keeps embedding caching and batching simple (no padding logic
needed downstream).

Shared by training/extract_embeddings.py and training/train_detector.py
(the latter only needs this directly when NOT using cached embeddings).
"""
import glob
import os
import random

import pandas as pd
import torch
import torchaudio
import yaml
from torch.utils.data import Dataset


def load_protocol_df(data_cfg: dict, split: str = "train") -> pd.DataFrame:
    """Load and correctly select the protocol TSV for a split, using the
    same delimiter-detection + train/dev/eval file-selection logic used in
    the setup notebook — duplicated here since notebooks aren't part of
    this package."""
    protocol_dir = os.path.join(data_cfg["paths"]["extracted_dir"], "protocols")
    all_tsvs = sorted(glob.glob(f"{protocol_dir}/**/*.tsv", recursive=True))
    split_tsvs = [f for f in all_tsvs if split in os.path.basename(f).lower()]
    if not split_tsvs:
        raise FileNotFoundError(
            f"No protocol TSV found for split='{split}' under {protocol_dir}. "
            f"Found: {all_tsvs}"
        )
    path = split_tsvs[0]

    columns = data_cfg["protocol_columns"]
    delimiter = data_cfg.get("protocol_delimiter", "\\s+")
    engine = "python" if delimiter != "\t" else None
    df = pd.read_csv(path, sep=delimiter, header=None, names=columns, engine=engine)
    assert len(df.columns) == len(columns), (
        f"Loaded {len(df.columns)} columns from {path}, but config lists "
        f"{len(columns)} in protocol_columns — config/data mismatch."
    )
    return df


def _index_flac_files(extracted_dir: str) -> dict:
    """filename (without extension) -> full path, for every .flac under extracted_dir."""
    index = {}
    for path in glob.glob(f"{extracted_dir}/**/*.flac", recursive=True):
        stem = os.path.splitext(os.path.basename(path))[0]
        index[stem] = path
    return index


class ASVspoof5Dataset(Dataset):
    """Yields (waveform, label) pairs. label: 1 = bonafide, 0 = spoof.

    waveform is always exactly `max_duration_sec * target_sample_rate`
    samples long — shorter clips are zero-padded, longer clips are cropped
    (randomly during training, from the start during eval, controlled by
    `random_crop`).
    """

    def __init__(self, data_cfg: dict, model_cfg: dict, split: str = "train", random_crop: bool = True):
        self.df = load_protocol_df(data_cfg, split=split)
        self.filename_col = "filename"
        self.label_col = data_cfg["label_column"]
        self.bonafide_val = data_cfg["bonafide_value"]

        extracted_dir = data_cfg["paths"]["extracted_dir"]
        flac_index = _index_flac_files(extracted_dir)

        # Only keep protocol rows we actually have audio for — with a single
        # shard downloaded, most rows in the full split protocol won't have
        # a matching file yet, and that's expected, not an error.
        matched_rows = self.df[self.df[self.filename_col].isin(flac_index.keys())]
        if len(matched_rows) == 0:
            raise ValueError(
                f"None of {len(self.df)} protocol rows for split='{split}' matched "
                f"any of {len(flac_index)} indexed .flac files under {extracted_dir}. "
                "Likely cause: audio for this split hasn't been downloaded yet, or the "
                "protocol split doesn't match the downloaded audio split."
            )
        self.rows = matched_rows.reset_index(drop=True)
        self.flac_index = flac_index

        self.target_sr = model_cfg["frontend"]["target_sample_rate"]
        self.max_samples = int(model_cfg["frontend"]["max_duration_sec"] * self.target_sr)
        self.random_crop = random_crop

    def __len__(self) -> int:
        return len(self.rows)

    def _fix_length(self, waveform: torch.Tensor) -> torch.Tensor:
        n = waveform.shape[-1]
        if n == self.max_samples:
            return waveform
        if n > self.max_samples:
            if self.random_crop:
                start = random.randint(0, n - self.max_samples)
            else:
                start = 0
            return waveform[..., start:start + self.max_samples]
        # shorter than target: zero-pad at the end
        pad = self.max_samples - n
        return torch.nn.functional.pad(waveform, (0, pad))

    def __getitem__(self, idx: int):
        row = self.rows.iloc[idx]
        path = self.flac_index[row[self.filename_col]]
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # mono
        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)
        waveform = self._fix_length(waveform.squeeze(0))

        label = 1 if row[self.label_col] == self.bonafide_val else 0
        return waveform, torch.tensor(label, dtype=torch.float32)


def load_configs(data_config_path: str, model_config_path: str) -> tuple:
    with open(data_config_path) as f:
        data_cfg = yaml.safe_load(f)
    with open(model_config_path) as f:
        model_cfg = yaml.safe_load(f)
    return data_cfg, model_cfg
