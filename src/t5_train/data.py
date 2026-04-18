"""Dataset loading and tokenization for T5 simplification.

Loads WikiLarge from the Hugging Face hub and MultiCochrane splits from CSV files under a
directory you pass in. Provides batched preprocessing that builds input_ids, attention_mask,
and labels with pad tokens masked to negative one hundred for loss computation.
"""
from __future__ import annotations

import os
from typing import Any

from datasets import DatasetDict, load_dataset
from transformers import PreTrainedTokenizer


def load_wikilarge(prefix: str) -> DatasetDict:
    """Load WikiLarge and add a fixed text prefix column plus input and target column names.

    Renames Normal and Simple columns to input_text and target_text. Every row in each split
    receives the same prefix string for T5 conditioning.
    """
    wiki = load_dataset("bogdancazan/wikilarge-text-simplification")
    for split in wiki:
        n = len(wiki[split])
        wiki[split] = wiki[split].add_column("prefix", [prefix] * n)
        wiki[split] = wiki[split].rename_column("Normal", "input_text")
        wiki[split] = wiki[split].rename_column("Simple", "target_text")
    return wiki


def load_multicochrane_csv(base_path: str) -> DatasetDict:
    """Load train, validation, and test CSVs from base_path if the files exist.

    Expects train0_en.csv, val0_en.csv, and test0_en.csv. Maps Expert and Simple columns to
    input_text and target_text when present. Raises if train or validation is missing.
    """
    data_files = {
        "train": os.path.join(base_path, "train0_en.csv"),
        "test": os.path.join(base_path, "test0_en.csv"),
        "validation": os.path.join(base_path, "val0_en.csv"),
    }
    data_files = {k: v for k, v in data_files.items() if os.path.exists(v)}
    if not data_files.get("train") or not data_files.get("validation"):
        raise FileNotFoundError("Need train and validation CSV under base_path.")
    raw = load_dataset("csv", data_files=data_files)
    out = DatasetDict()
    for split in raw:
        ds = raw[split]
        cols = ds.column_names
        if "Expert" in cols and "Simple" in cols:
            ds = ds.rename_column("Expert", "input_text")
            ds = ds.rename_column("Simple", "target_text")
        elif "input_text" not in cols or "target_text" not in cols:
            raise ValueError(f"Unexpected columns in {split}: {cols}")
        out[split] = ds
    return out


def add_prefix_column(ds: DatasetDict, prefix: str) -> DatasetDict:
    """Ensure each split has a prefix column set to the given string (mapped if missing)."""
    def _add(ex):
        ex["prefix"] = prefix
        return ex

    for split in ds:
        if "prefix" not in ds[split].column_names:
            ds[split] = ds[split].map(_add)
    return ds


def preprocess_wiki_examples(
    examples: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    max_input_length: int,
    max_target_length: int,
    mask_pad_with: str = "pad_id",
) -> dict[str, Any]:
    """Tokenize WikiLarge batches: prefix plus input as source, target as labels.

    mask_pad_with equals zero to replace label id zero with ignore index, or pad_id to mask
    the tokenizer pad token id instead.
    """
    input_txt = examples["input_text"]
    target_txt = examples["target_text"]
    prefixes = examples["prefix"]
    inputs = [p + inp for p, inp in zip(prefixes, input_txt)]
    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        padding="max_length",
        truncation=True,
    )
    labels = tokenizer(
        target_txt,
        max_length=max_target_length,
        padding="max_length",
        truncation=True,
    ).input_ids
    labels_with_ignore: list[list[int]] = []
    for row in labels:
        if mask_pad_with == "zero":
            labels_with_ignore.append([t if t != 0 else -100 for t in row])
        else:
            pad = tokenizer.pad_token_id
            labels_with_ignore.append([t if t != pad else -100 for t in row])
    model_inputs["labels"] = labels_with_ignore
    return model_inputs


def preprocess_medical_examples(
    examples: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> dict[str, Any]:
    """Tokenize MultiCochrane batches with prefix plus expert text and simple targets.

    Labels mask padding with negative one hundred for the language modeling loss.
    """
    inputs = [p + inp for p, inp in zip(examples["prefix"], examples["input_text"])]
    model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True)
    labels = tokenizer(
        examples["target_text"],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    ).input_ids
    pad = tokenizer.pad_token_id
    model_inputs["labels"] = [[t if t != pad else -100 for t in row] for row in labels]
    return model_inputs


def encode_wikilarge(
    wiki: DatasetDict,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    mask_pad_with: str,
    num_proc: int = 1,
) -> tuple[DatasetDict, Any, Any, Any]:
    """Map tokenization over train, validation, and test splits. Returns encoded splits and wiki."""
    def _fn(batch):
        return preprocess_wiki_examples(batch, tokenizer, max_length, max_length, mask_pad_with)

    train_ds = wiki["train"]
    val_ds = wiki["validation"]
    test_ds = wiki["test"]
    enc_train = train_ds.map(_fn, batched=True, remove_columns=train_ds.column_names, num_proc=num_proc)
    enc_val = val_ds.map(_fn, batched=True, remove_columns=val_ds.column_names, num_proc=num_proc)
    enc_test = test_ds.map(_fn, batched=True, remove_columns=test_ds.column_names, num_proc=num_proc)
    return enc_train, enc_val, enc_test, wiki


def encode_multicochrane(
    ds: DatasetDict,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    num_proc: int = 1,
) -> DatasetDict:
    """Encode all splits with preprocess_medical_examples and drop empty rows per split."""
    encoded = DatasetDict()
    for split, part in ds.items():
        cols = part.column_names
        encoded[split] = part.map(
            lambda b: preprocess_medical_examples(b, tokenizer, max_length),
            batched=True,
            remove_columns=cols,
            num_proc=num_proc,
            load_from_cache_file=True,
        )
        n0 = len(encoded[split])
        encoded[split] = encoded[split].filter(
            lambda ex: len(ex["input_ids"]) > 0 and len(ex["labels"]) > 0
        )
        if len(encoded[split]) < n0:
            print(f"Filtered {n0 - len(encoded[split])} empty rows from {split}.")
    return encoded
