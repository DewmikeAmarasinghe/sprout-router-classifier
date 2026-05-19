"""
CSV → HuggingFace Dataset with tokenization.

load_and_tokenize() converts train and val DataFrames into datasets.Dataset
objects with input_ids, attention_mask, and labels ready for HuggingFace Trainer.

TEST SET POLICY:
    test.csv is never loaded here. It is loaded only once in the final evaluation
    phase (phase_8_evaluate.py). All development uses train + val splits only.

WHY max_length=64:
    95th percentile of Sprout queries is < 50 tokens.
    64 covers all but the most extreme edge cases.
    128 wastes ~30% more GPU memory and slows training by ~30% with negligible gain.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedTokenizerBase

log = logging.getLogger(__name__)


def load_and_tokenize(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    hf_model_name: str,
    max_length: int = 64,
) -> tuple:
    """Tokenize train and val DataFrames into HuggingFace Datasets.

    Args:
        train_df: DataFrame with columns ["text", "label"].
        val_df: DataFrame with columns ["text", "label"].
        hf_model_name: HuggingFace hub model ID (e.g. "xlm-roberta-base").
        max_length: Token sequence length. Default 64 covers Sprout P95.

    Returns:
        Tuple of (train_dataset, val_dataset, tokenizer).
        Datasets have columns: input_ids, attention_mask, labels.
    """
    from datasets import Dataset
    from transformers import AutoTokenizer

    log.info(f"Loading tokenizer: {hf_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

    train_df = validate_and_clean(train_df, split="train")
    val_df = validate_and_clean(val_df, split="val")

    log.info(f"Tokenizing train ({len(train_df):,}) and val ({len(val_df):,}) sets ...")

    train_ds = Dataset.from_pandas(train_df[["text", "label"]].reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df[["text", "label"]].reset_index(drop=True))

    def tokenize(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    train_ds = train_ds.map(tokenize, batched=True, desc="Tokenising train")
    val_ds = val_ds.map(tokenize, batched=True, desc="Tokenising val")

    # HuggingFace Trainer expects the label column to be named "labels"
    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")

    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    log.info(
        f"Tokenization done. "
        f"Train features: {train_ds.column_names}  "
        f"Val features: {val_ds.column_names}"
    )
    return train_ds, val_ds, tokenizer


def load_single_split(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 64,
    split_name: str = "test",
) -> Dataset:
    """Tokenize a single DataFrame split with an already-loaded tokenizer.

    Used when loading the test set in isolation during final evaluation.
    """
    from datasets import Dataset

    df = validate_and_clean(df, split=split_name)
    ds = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))

    def tokenize(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    ds = ds.map(tokenize, batched=True, desc=f"Tokenising {split_name}")
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return ds


def validate_and_clean(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Ensure required columns exist and drop null rows."""
    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[{split}] DataFrame missing columns: {missing}")

    before = len(df)
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.strip() != ""]
    after = len(df)
    if after < before:
        log.warning(f"[{split}] Dropped {before - after} null/empty rows.")

    df = df.copy()
    df["label"] = df["label"].astype(int)
    return df
