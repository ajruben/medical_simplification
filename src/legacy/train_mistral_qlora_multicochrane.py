#!/usr/bin/env python
# coding: utf-8

"""
Simplified script to fine-tune Mistral-7B using QLoRA
on the medical text simplification dataset.
"""

import os
import logging
import math
import sys
from pathlib import Path

import datasets as hf_datasets
import evaluate
import nltk
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from datasets import DatasetDict, load_dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling, # Use this for Causal LM
    DataCollatorForSeq2Seq,
    SchedulerType,
    Trainer,
    TrainingArguments,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from huggingface_hub import login
import bootstrap_paths  # noqa: F401
from paths import MULTICOHRANE_EN_UNFILTERED, OUTPUT_ROOT

# --- Configuration ---
MODEL_CHECKPOINT = "mistralai/Mistral-7B-v0.1"
MEDICAL_DATASET_PATH = str(MULTICOHRANE_EN_UNFILTERED)
OUTPUT_DIR = str(OUTPUT_ROOT / "mistral_qlora_medical_simple_output")
TRUST_REMOTE_CODE = True # Needed for Mistral

# QLoRA Config
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
# Auto-detect target modules later

# Training Config
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 2
PER_DEVICE_EVAL_BATCH_SIZE = 4 # Can be larger for eval
GRADIENT_ACCUMULATION_STEPS = 8 # Effective batch size = 2 * 8 = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
LR_SCHEDULER_TYPE = "cosine"
LOGGING_STEPS = 25
SAVE_STRATEGY = "epoch"
EVAL_STRATEGY = "epoch"
MIXED_PRECISION = "bf16" # Use 'bf16' for RTX 4090
SEED = 42

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
        "test": os.path.join(MEDICAL_DATASET_PATH, "test0_en.csv")
    }
    for split, file_path in data_files.items():
        if not os.path.isfile(file_path):
             raise FileNotFoundError(f"Medical data file not found: {file_path}")

    raw_datasets = load_dataset("csv", data_files=data_files)
    logger.info(f"Medical dataset loaded: {raw_datasets}")

    # --- Preprocessing for Causal LM ---
    # Combine input and target, tokenize together
    def preprocess_function(examples):
        # Combine prefix, input, and target text. Add EOS token.
        combined_texts = [
            PREFIX + inp + " " + tgt + tokenizer.eos_token
            for inp, tgt in zip(examples[INPUT_COLUMN], examples[TARGET_COLUMN])
        ]

        # Tokenize the combined text
        # Use a longer max_length if needed, sum of input/target + prefix/separators
        combined_max_length = MAX_INPUT_LENGTH + MAX_TARGET_LENGTH
        tokenized_examples = tokenizer(
            combined_texts,
            max_length=combined_max_length,
            truncation=True,
            padding="max_length" # Pad explicitly here
        )

        # For Causal LM, labels are typically the same as input_ids
        tokenized_examples["labels"] = tokenized_examples["input_ids"].copy()

        # Optional: Mask prompt tokens in labels if desired (set label to -100)
        # This requires knowing the length of the tokenized prompt part.
        # Example (needs careful implementation):
        # prompt_inputs = tokenizer([PREFIX + inp for inp in examples[INPUT_COLUMN]], max_length=MAX_INPUT_LENGTH, truncation=True)
        # prompt_lengths = [len(p) for p in prompt_inputs["input_ids"]]
        # for i, label_ids in enumerate(tokenized_examples["labels"]):
        #     prompt_len = prompt_lengths[i]
        #     tokenized_examples["labels"][i][:prompt_len] = [-100] * prompt_len

        return tokenized_examples

    column_names = raw_datasets["train"].column_names
    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )
    logger.info(f"Processed dataset structure: {processed_datasets}")
    return processed_datasets

# --- Model Loading ---
def load_model_and_tokenizer_qlora():
    logger.info(f"Loading model and tokenizer for '{MODEL_CHECKPOINT}' with QLoRA...")

    # Quantization Config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if MIXED_PRECISION == 'bf16' else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    logger.info("Using 4-bit quantization (BitsAndBytesConfig)")

    # Load Config
    config = AutoConfig.from_pretrained(
        MODEL_CHECKPOINT,
        trust_remote_code=TRUST_REMOTE_CODE,
    )

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CHECKPOINT,
        use_fast=True,
        trust_remote_code=TRUST_REMOTE_CODE,
    )
    if tokenizer.pad_token is None:
        logger.warning("Tokenizer does not have a pad token. Setting pad_token = eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        config.pad_token_id = config.eos_token_id

    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CHECKPOINT,
        config=config,
        quantization_config=quantization_config,
        trust_remote_code=TRUST_REMOTE_CODE,
        # device_map="auto" # Let Trainer handle device placement
    )

    # --- PEFT (QLoRA) Setup ---
    logger.info("Applying QLoRA...")
    model = prepare_model_for_kbit_training(model)

    # Define LoRA target modules
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    logger.info(f"Using target modules for Mistral: {target_modules}")

    peft_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, peft_config)
    logger.info("PEFT model created.")
    model.print_trainable_parameters()

    return model, tokenizer

# --- Metrics ---
def compute_metrics(eval_preds):
    # Simple compute metrics for Causal LM - just return loss
    # Perplexity can be calculated if needed: math.exp(eval_preds.metrics["eval_loss"])
    return {}

# --- Main ---
def main():
    hf_login()
    set_seed(SEED)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer_qlora()

    # Load and prepare dataset
    processed_datasets = load_and_prepare_medical_dataset(tokenizer)

    # Data collator
    logger.info("Using DataCollatorForLanguageModeling for Causal LM.")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False # Ensure causal language modeling format
    )

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        eval_strategy=EVAL_STRATEGY,  # Correct argument name
        save_strategy=SAVE_STRATEGY,  # Correct argument name
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        logging_steps=LOGGING_STEPS,
        save_total_limit=1, # Keep only the best checkpoint
        load_best_model_at_end=True,
        metric_for_best_model="loss", # Evaluate based on loss for Causal LM
        greater_is_better=False,
        report_to="tensorboard",
        seed=SEED,
        fp16=MIXED_PRECISION == "fp16",
        bf16=MIXED_PRECISION == "bf16",
        optim="paged_adamw_8bit" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else "adamw_torch", # Use paged optimizer if available
        gradient_checkpointing=True, # Enable gradient checkpointing for memory saving
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_datasets["train"],
        eval_dataset=processed_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics, # Simple loss/perplexity metric
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()
    trainer.save_model() # Saves the adapter model
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    logger.info(f"Training complete. Best model saved to {OUTPUT_DIR}")

    # Evaluate
    logger.info("Evaluating model on validation set...")
    metrics = trainer.evaluate(metric_key_prefix="eval")
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    logger.info("Script finished.")

if __name__ == "__main__":
    main()
