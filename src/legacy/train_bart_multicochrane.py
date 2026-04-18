#!/usr/bin/env python
# coding: utf-8

"""
Simplified script to fine-tune BART-large
on the medical text simplification dataset.
Aims for faster training (~1-2 hours).
"""

import os
import logging
import sys
from pathlib import Path
import random
import evaluate
import nltk
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import bootstrap_paths  # noqa: F401
from paths import MULTICOHRANE_EN_UNFILTERED, OUTPUT_ROOT

# --- Configuration ---
MODEL_CHECKPOINT = "facebook/bart-large"
MEDICAL_DATASET_PATH = str(MULTICOHRANE_EN_UNFILTERED)
OUTPUT_DIR = str(OUTPUT_ROOT / "bart_large_medical_output")
TRUST_REMOTE_CODE = False

# Training Config (Adjusted for Robustness)
NUM_TRAIN_EPOCHS = 3  # Increased epochs
PER_DEVICE_TRAIN_BATCH_SIZE = 32  # Reduced batch size
PER_DEVICE_EVAL_BATCH_SIZE = 32  # Reduce eval batch size
GRADIENT_ACCUMULATION_STEPS = 4  # Add gradient accumulation
LEARNING_RATE = 1e-5  # Adjusted learning rate
WEIGHT_DECAY = 0.05  # Increased weight decay
WARMUP_RATIO = 0.05  # Slightly increase warmup
LR_SCHEDULER_TYPE = "cosine"  # Changed scheduler
LOGGING_STEPS = 50
SAVE_STRATEGY = "epoch"
EVAL_STRATEGY = "epoch"
MIXED_PRECISION = "bf16"
SEED = 42
METRIC_FOR_BEST_MODEL = "sari_combined"  # Custom metric
GREATER_IS_BETTER = True

# Generation Config (For Aggressive Simplification)
NUM_BEAMS = 10  # Increase beams
LENGTH_PENALTY = 0.8  # Encourage shorter sentences
EARLY_STOPPING = True
NO_REPEAT_NGRAM_SIZE = 2  # Prevent repetition
MAX_LENGTH = 256
MIN_LENGTH = 32

# Data Config
INPUT_COLUMN = "input_text"
TARGET_COLUMN = "target_text"
PREFIX = "simplify: "
MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 256
IGNORE_PAD_TOKEN_FOR_LOSS = True

# --- Setup ---
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)

# NLTK download
try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    nltk.download("punkt", quiet=True)

# --- Hugging Face Login ---
def hf_login():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        logger.info("HF_TOKEN not set; skipping Hugging Face Hub login.")
        return
    try:
        login(token=token)
        logger.info("Successfully logged into Hugging Face Hub.")
    except Exception as e:
        logger.error(f"Failed to log into Hugging Face Hub: {e}")

# --- Data Loading and Preprocessing ---
def load_and_prepare_medical_dataset(tokenizer):
    logger.info(f"Loading medical dataset from: {MEDICAL_DATASET_PATH}")
    if not os.path.isdir(MEDICAL_DATASET_PATH):
        raise FileNotFoundError(f"Medical dataset path not found or not a directory: {MEDICAL_DATASET_PATH}")
    data_files = {
        "train": os.path.join(MEDICAL_DATASET_PATH, "train0_en.csv"),
        "validation": os.path.join(MEDICAL_DATASET_PATH, "val0_en.csv"),
        "test": os.path.join(MEDICAL_DATASET_PATH, "test0_en.csv"),
    }
    for split, file_path in data_files.items():
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Medical data file not found: {file_path}")

    raw_datasets = load_dataset("csv", data_files=data_files)
    logger.info(f"Medical dataset loaded: {raw_datasets}")

    def preprocess_function(examples):
        inputs = [PREFIX + doc for doc in examples[INPUT_COLUMN]]
        targets = [doc for doc in examples[TARGET_COLUMN]]
        model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)

        labels = tokenizer(text_target=targets, max_length=MAX_TARGET_LENGTH, truncation=True)

        if IGNORE_PAD_TOKEN_FOR_LOSS:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        original_inputs = tokenizer(examples[INPUT_COLUMN], max_length=MAX_INPUT_LENGTH, truncation=True)
        model_inputs["original_inputs"] = original_inputs["input_ids"]

        return model_inputs

    column_names = raw_datasets["train"].column_names
    processed_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=column_names, desc="Running tokenizer on dataset"
    )
    logger.info(f"Processed dataset structure: {processed_datasets}")
    return processed_datasets

# --- Model Loading ---
def load_model_and_tokenizer():
    logger.info(f"Loading model and tokenizer for '{MODEL_CHECKPOINT}'...")
    config = AutoConfig.from_pretrained(MODEL_CHECKPOINT, trust_remote_code=TRUST_REMOTE_CODE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=True, trust_remote_code=TRUST_REMOTE_CODE)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT, config=config, trust_remote_code=TRUST_REMOTE_CODE)
    return model, tokenizer

# --- Metrics ---
def get_compute_metrics_fn(tokenizer):
    sari_metric = evaluate.load("sari")
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')  # load semantic model once.

    def compute_metrics(eval_preds):
        preds, labels, original_inputs = eval_preds.predictions, eval_preds.label_ids, eval_preds.inputs

        if original_inputs is None:
            return {"sari": 0.0, "jaccard": 0.0, "semantic_similarity": 1.0, "diff_words": 0, "sari_substitution": 0.0, "sari_combined": 0.0}

        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, max_length=MAX_LENGTH, min_length=MIN_LENGTH)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, max_length=MAX_LENGTH, min_length=MIN_LENGTH)

        original_inputs = np.where(original_inputs != -100, original_inputs, tokenizer.pad_token_id)
        decoded_sources = tokenizer.batch_decode(original_inputs, skip_special_tokens=True, max_length=MAX_LENGTH, min_length=MIN_LENGTH)

        all_refs = [[label] for label in decoded_labels]

        sari_results = sari_metric.compute(sources=decoded_sources, predictions=decoded_preds, references=all_refs)
        sari = sari_results["sari"]

        # Calculate a custom "substitution-focused" SARI component
        sari_substitution = sari_results["sari_Ngram_F1_p"][:, 1].mean()  # Focus on 1-gram precision for addition


        jaccard_scores = []
        semantic_similarities = []
        diff_words_percentages = []

        for source, pred in zip(decoded_sources, decoded_preds):
            source_words = set(source.lower().split())
            pred_words = set(pred.lower().split())
            jaccard_scores.append(1 - nltk.jaccard_distance(source_words, pred_words))
            diff_words_percentages.append(
                (len(source_words.symmetric_difference(pred_words)) / len(source_words.union(pred_words))) * 100
            )
            source_embedding = sentence_model.encode(source, convert_to_tensor=True)
            pred_embedding = sentence_model.encode(pred, convert_to_tensor=True)
            source_embedding = source_embedding.unsqueeze(0)  # added line
            pred_embedding = pred_embedding.unsqueeze(0)  # added line
            semantic_similarity = F.cosine_similarity(source_embedding, pred_embedding).item()
            semantic_similarities.append(semantic_similarity)

        jaccard = np.mean(jaccard_scores)
        semantic_similarity = np.mean(semantic_similarities)
        diff_words = np.mean(diff_words_percentages)

        # Custom combined metric to reward substitution and penalize similarity
        sari_combined = sari_substitution - (2.0 * semantic_similarity) + (0.5 * diff_words) #Experiment with weights!

        if random.random() < 0.05:
            idx = random.randint(0, len(decoded_sources) - 1)
            logger.info(
                f"Eval Sample:\nSource: {decoded_sources[idx]}\nTarget: {decoded_labels[idx]}\nPrediction: {decoded_preds[idx]}\nSARI: {sari:.4f}\nSubstitution SARI: {sari_substitution:.4f}\nJaccard: {jaccard:.4f}\nSemantic: {semantic_similarity:.4f}\nDiff Words: {diff_words:.2f}%\nCombined Metric: {sari_combined:.4f}"
            )

        return {"sari": sari, "jaccard": jaccard, "semantic_similarity": semantic_similarity, "diff_words": diff_words, "sari_substitution": sari_substitution, "sari_combined": sari_combined}

    return compute_metrics

# --- Main ---
def main():
    hf_login()
    set_seed(SEED)

    model, tokenizer = load_model_and_tokenizer()
    processed_datasets = load_and_prepare_medical_dataset(tokenizer)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100 if IGNORE_PAD_TOKEN_FOR_LOSS else tokenizer.pad_token_id,
        pad_to_multiple_of=8 if MIXED_PRECISION != "no" else None,
    )

    compute_metrics_fn = get_compute_metrics_fn(tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        eval_strategy=EVAL_STRATEGY,
        save_strategy=SAVE_STRATEGY,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        logging_steps=LOGGING_STEPS,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=METRIC_FOR_BEST_MODEL,
        greater_is_better=GREATER_IS_BETTER,
        predict_with_generate=True,
        generation_max_length=MAX_LENGTH,
        report_to="tensorboard",
        seed=SEED,
        fp16=MIXED_PRECISION == "fp16",
        bf16=MIXED_PRECISION == "bf16",
        optim="adamw_torch",
        include_inputs_for_metrics=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_datasets["train"],
        eval_dataset=processed_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )

    logger.info("Starting training...")
    train_result = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    logger.info(f"Training complete. Best model saved to {OUTPUT_DIR}")

    logger.info("Evaluating model on validation set...")
    metrics = trainer.evaluate(
        metric_key_prefix="eval",
        generation_num_beams=NUM_BEAMS,  # Pass generation params to evaluate
        generation_length_penalty=LENGTH_PENALTY,
        generation_early_stopping=EARLY_STOPPING,
        generation_no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
        max_length=MAX_LENGTH,
        min_length=MIN_LENGTH,
    )
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    logger.info("Script finished.")

if __name__ == "__main__":
    main()