#!/usr/bin/env python
# coding: utf-8

"""
Fine-tunes various language models (Encoder-Decoder and Decoder-only)
for text simplification using Hugging Face Transformers, PEFT, bitsandbytes,
and optionally Optuna for hyperparameter optimization.

Prioritizes medical domain data by default and supports WikiLarge.
Includes options for full fine-tuning, LoRA, and QLoRA.
"""

import os
import argparse
import logging
import math
import random
import sys
from pathlib import Path

import datasets as hf_datasets
import evaluate
import nltk
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from datasets import DatasetDict, load_dataset, concatenate_datasets
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    SchedulerType,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
    default_data_collator,
    get_scheduler,
    set_seed,
)
import optuna
from huggingface_hub import login

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)

# NLTK download for metrics
try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    nltk.download("punkt", quiet=True)

import bootstrap_paths  # noqa: F401
from paths import MULTICOHRANE_EN_UNFILTERED, OUTPUT_ROOT

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLMs for Text Simplification")

    # Model Args
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Hugging Face model identifier (e.g., 'mistralai/Mistral-7B-v0.1', 'facebook/bart-large').")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code for model loading (needed for some models).")

    # Data Args
    parser.add_argument("--dataset_name", type=str, default="medical", choices=["medical", "wikilarge", "combined"], help="Dataset to use.")
    parser.add_argument("--medical_dataset_path", type=str, default=str(MULTICOHRANE_EN_UNFILTERED), help="Path to the directory containing medical CSV files (train0_en.csv, val0_en.csv, test0_en.csv).")
    parser.add_argument("--wikilarge_dataset_name", type=str, default="bogdancazan/wikilarge-text-simplification", help="Hugging Face dataset identifier for WikiLarge.")
    parser.add_argument("--input_column", type=str, default="input_text", help="Column name for source text in datasets.")
    parser.add_argument("--target_column", type=str, default="target_text", help="Column name for target text in datasets.")
    parser.add_argument("--prefix", type=str, default="simplify: ", help="Prefix added to source text.")
    parser.add_argument("--max_input_length", type=int, default=256, help="Maximum input sequence length after tokenization.")
    parser.add_argument("--max_target_length", type=int, default=256, help="Maximum target sequence length after tokenization.")
    parser.add_argument("--ignore_pad_token_for_loss", type=bool, default=True, help="Whether to ignore the tokens corresponding to padded labels in the loss computation.")

    # Fine-tuning Strategy Args
    parser.add_argument("--use_qlora", action="store_true", help="Enable QLoRA (4-bit base model + LoRA adapters).")
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA (FP16/BF16/INT8 base model + LoRA adapters).")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank (r).")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=None, help="List of module names to apply LoRA to (e.g., 'q_proj' 'v_proj'). If None, common modules for the model type might be inferred.")

    # Efficiency Args
    parser.add_argument("--load_in_4bit", action="store_true", help="Load base model in 4-bit (implies --use_qlora if not already set).")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load base model in 8-bit.")
    parser.add_argument("--use_8bit_optimizer", action="store_true", help="Use bitsandbytes 8-bit AdamW optimizer.")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision training. 'bf16' recommended for Ampere+ GPUs.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save memory.")

    # Training Args (Trainer)
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_ROOT / "simplification_output"), help="Output directory for checkpoints and logs.")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory.")
    parser.add_argument("--do_train", action="store_true", default=True, help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", default=True, help="Whether to run evaluation on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument("--evaluation_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"], help="Evaluation strategy.")
    parser.add_argument("--eval_steps", type=int, default=None, help="Evaluate every N steps if evaluation_strategy='steps'.")
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"], help="Checkpoint saving strategy.")
    parser.add_argument("--save_steps", type=int, default=None, help="Save checkpoint every N steps if save_strategy='steps'.")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Limit the total number of checkpoints. Deletes older checkpoints.")
    parser.add_argument("--load_best_model_at_end", action="store_true", default=True, help="Load the best model found during training at the end.")
    parser.add_argument("--metric_for_best_model", type=str, default="sari", help="Metric to use to compare models.")
    parser.add_argument("--greater_is_better", type=bool, default=True, help="Whether the metric_for_best_model should be maximized.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per GPU for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size per GPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--label_smoothing_factor", type=float, default=0.0, help="Label smoothing factor (only for Seq2Seq models).")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Linear warmup over warmup_ratio fraction of total steps.")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="cosine", choices=[s.value for s in SchedulerType], help="Learning rate scheduler type.")
    # Seq2Seq specific args
    parser.add_argument("--predict_with_generate", action="store_true", default=True, help="Use generate for predictions (only for Seq2Seq models).")
    parser.add_argument("--generation_max_length", type=int, default=None, help="Max length for generation during evaluation (only for Seq2Seq models). Defaults to max_target_length.")
    parser.add_argument("--generation_num_beams", type=int, default=1, help="Number of beams for generation during evaluation (only for Seq2Seq models).")
    # General args
    parser.add_argument("--logging_strategy", type=str, default="steps", choices=["no", "steps", "epoch"], help="Logging strategy.")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every N steps.")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Integrations to report results to (e.g., 'tensorboard', 'wandb', 'none').")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # HPO Args (Optuna)
    parser.add_argument("--run_hpo", action="store_true", help="Run Optuna hyperparameter optimization instead of a single training run.")
    parser.add_argument("--num_hpo_trials", type=int, default=5, help="Number of Optuna trials to run.")
    parser.add_argument("--hpo_output_dir", type=str, default=str(OUTPUT_ROOT / "hpo_simplification"), help="Output directory for HPO results.")

    args = parser.parse_args()

    # --- Argument Validation and Post-processing ---
    if args.load_in_4bit:
        args.use_qlora = True # 4bit loading implies QLoRA usage

    if args.use_qlora and args.use_lora:
        logger.warning("Both --use_qlora and --use_lora specified. Prioritizing QLoRA.")
        args.use_lora = False

    if args.use_qlora and args.load_in_8bit:
        logger.warning("Both --use_qlora (4-bit) and --load_in_8bit specified. Prioritizing QLoRA (4-bit).")
        args.load_in_8bit = False

    if args.load_in_4bit and args.load_in_8bit:
         logger.warning("Both --load_in_4bit and --load_in_8bit specified. Prioritizing 4-bit.")
         args.load_in_8bit = False

    if args.generation_max_length is None:
        args.generation_max_length = args.max_target_length

    # Ensure output dirs exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.run_hpo:
        Path(args.hpo_output_dir).mkdir(parents=True, exist_ok=True)

    return args

# --- Data Loading and Preprocessing ---
def load_and_prepare_datasets(args, tokenizer):
    """Loads datasets based on args and preprocesses them."""
    raw_datasets = DatasetDict()

    # Load Medical Data (CSV)
    if args.dataset_name in ["medical", "combined"]:
        logger.info(f"Loading medical dataset from: {args.medical_dataset_path}")
        if not os.path.isdir(args.medical_dataset_path):
             raise FileNotFoundError(f"Medical dataset path not found or not a directory: {args.medical_dataset_path}")
        data_files = {
            "train": os.path.join(args.medical_dataset_path, "train0_en.csv"),
            "validation": os.path.join(args.medical_dataset_path, "val0_en.csv"),
            "test": os.path.join(args.medical_dataset_path, "test0_en.csv")
        }
        for split, file_path in data_files.items():
            if not os.path.isfile(file_path):
                 raise FileNotFoundError(f"Medical data file not found: {file_path}")

        medical_ds = load_dataset("csv", data_files=data_files)
        # Basic validation of expected columns
        for split in medical_ds:
            if args.input_column not in medical_ds[split].column_names or args.target_column not in medical_ds[split].column_names:
                 raise ValueError(f"Medical dataset split '{split}' missing required columns '{args.input_column}' or '{args.target_column}'. Found: {medical_ds[split].column_names}")
        raw_datasets["medical"] = medical_ds
        logger.info(f"Medical dataset loaded: {medical_ds}")


    # Load WikiLarge Data (Hugging Face Hub)
    if args.dataset_name in ["wikilarge", "combined"]:
        logger.info(f"Loading WikiLarge dataset: {args.wikilarge_dataset_name}")
        wikilarge_ds = load_dataset(args.wikilarge_dataset_name)
        # Rename columns to be consistent
        column_mapping = {"Normal": args.input_column, "Simple": args.target_column}
        wikilarge_ds = wikilarge_ds.rename_columns(column_mapping)
         # Basic validation of expected columns
        for split in wikilarge_ds:
            if args.input_column not in wikilarge_ds[split].column_names or args.target_column not in wikilarge_ds[split].column_names:
                 raise ValueError(f"WikiLarge dataset split '{split}' missing required columns '{args.input_column}' or '{args.target_column}' after renaming. Found: {wikilarge_ds[split].column_names}")
        raw_datasets["wikilarge"] = wikilarge_ds
        logger.info(f"WikiLarge dataset loaded: {wikilarge_ds}")

    # Combine datasets if needed
    if args.dataset_name == "combined":
        if "medical" not in raw_datasets or "wikilarge" not in raw_datasets:
            raise ValueError("Both medical and wikilarge datasets must be loaded for 'combined' mode.")
        logger.info("Combining medical and WikiLarge datasets.")
        combined_ds = DatasetDict()
        for split in ["train", "validation", "test"]:
            # Ensure consistent features before concatenating
            # Keep only input_text and target_text for simplicity
            med_split = raw_datasets["medical"][split].remove_columns([col for col in raw_datasets["medical"][split].column_names if col not in [args.input_column, args.target_column]])
            wiki_split = raw_datasets["wikilarge"][split].remove_columns([col for col in raw_datasets["wikilarge"][split].column_names if col not in [args.input_column, args.target_column]])
            combined_ds[split] = concatenate_datasets([med_split, wiki_split])
        final_raw_ds = combined_ds
    elif args.dataset_name == "medical":
        final_raw_ds = raw_datasets["medical"]
    elif args.dataset_name == "wikilarge":
        final_raw_ds = raw_datasets["wikilarge"]
    else:
        raise ValueError(f"Invalid dataset_name: {args.dataset_name}") # Should not happen

    logger.info(f"Final raw dataset structure: {final_raw_ds}")

    # --- Preprocessing ---
    def preprocess_function(examples):
        inputs = [args.prefix + doc for doc in examples[args.input_column]]
        model_inputs = tokenizer(inputs, max_length=args.max_input_length, truncation=True)

        # Setup the tokenizer for targets
        labels = tokenizer(text_target=examples[args.target_column], max_length=args.max_target_length, truncation=True)

        if args.ignore_pad_token_for_loss:
            # Replace tokenizer.pad_token_id in the labels by -100 when we want to ignore padding in the loss.
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        # Add original inputs if needed for metrics like SARI
        # We need to tokenize them separately as they shouldn't have the prefix
        original_inputs = tokenizer(examples[args.input_column], max_length=args.max_input_length, truncation=True)
        model_inputs["original_inputs"] = original_inputs["input_ids"]

        return model_inputs

    # Apply preprocessing
    column_names = final_raw_ds["train"].column_names
    processed_datasets = final_raw_ds.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    logger.info(f"Processed dataset structure: {processed_datasets}")
    logger.info(f"Example processed input_ids: {processed_datasets['train'][0]['input_ids']}")
    logger.info(f"Example processed labels: {processed_datasets['train'][0]['labels']}")
    logger.info(f"Example processed original_inputs: {processed_datasets['train'][0]['original_inputs']}")


    return processed_datasets

# --- Metrics Computation ---
def get_compute_metrics_fn(tokenizer):
    """Returns a function to compute metrics."""
    sari_metric = evaluate.load("sari")

    def compute_metrics_seq2seq(eval_preds):
        # This function is for Seq2Seq models and relies on generated predictions
        preds, labels, original_inputs = eval_preds.predictions, eval_preds.label_ids, eval_preds.inputs

        if original_inputs is None:
             logger.error("Original inputs not found in eval_preds for SARI calculation. Make sure 'original_inputs' is included in the dataset and TrainingArguments has include_inputs_for_metrics=True.")
             return {"sari": 0.0}

        # Decode predictions (generated by predict_with_generate)
        if isinstance(preds, tuple): preds = preds[0]
        # preds are generated token IDs. Replace -100 if present (shouldn't be for generation)
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Decode labels (ground truth)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Decode original inputs (sources for SARI)
        original_inputs = np.where(original_inputs != -100, original_inputs, tokenizer.pad_token_id)
        decoded_sources = tokenizer.batch_decode(original_inputs, skip_special_tokens=True)


        # Prepare references for SARI
        all_refs = [[label] for label in decoded_labels]

        # Compute SARI
        try:
            sari_results = sari_metric.compute(sources=decoded_sources, predictions=decoded_preds, references=all_refs)
            result = {"sari": sari_results["sari"]}
        except Exception as e:
            logger.error(f"Error computing SARI: {e}")
            result = {"sari": 0.0}

        # Log a few examples
        if random.random() < 0.01:
            idx = random.randint(0, len(decoded_sources) - 1)
            logger.info(f"Seq2Seq Eval Sample:\nSource: {decoded_sources[idx]}\nTarget: {decoded_labels[idx]}\nPrediction: {decoded_preds[idx]}\nSARI: {result.get('sari', 0.0):.4f}")

        return result

    def compute_metrics_causal(eval_preds):
        # For Causal LM, standard evaluation often uses perplexity.
        # SARI requires generation, which isn't default in Trainer's evaluate.
        # We'll just return the loss for now.
        # Perplexity calculation:
        # try:
        #     loss = eval_preds.metrics["eval_loss"]
        #     perplexity = math.exp(loss)
        #     return {"perplexity": perplexity, "loss": loss}
        # except KeyError:
        #     return {} # Return empty if loss not available
        return {} # Return empty for now, focus on training loss

    # Return the appropriate function based on model type (determined later)
    return compute_metrics_seq2seq, compute_metrics_causal

# --- Model Loading ---
def load_model_and_tokenizer(args):
    """Loads model and tokenizer based on args, applying quantization and PEFT if needed."""

    # Quantization Config (for QLoRA or 4/8bit loading)
    quantization_config = None
    if args.load_in_4bit or args.use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.mixed_precision == 'bf16' else torch.float16, # Match compute dtype to mixed precision
            bnb_4bit_use_double_quant=True, # Recommended for stability
        )
        logger.info("Using 4-bit quantization (BitsAndBytesConfig)")
    elif args.load_in_8bit:
         quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
         logger.info("Using 8-bit quantization (BitsAndBytesConfig)")

    # Load Config
    config = AutoConfig.from_pretrained(
        args.model_checkpoint,
        trust_remote_code=args.trust_remote_code,
    )

    # --- Determine Model Type (Seq2Seq or CausalLM) ---
    is_encoder_decoder = config.is_encoder_decoder
    model_class = AutoModelForSeq2SeqLM if is_encoder_decoder else AutoModelForCausalLM
    logger.info(f"Model type detected: {'Encoder-Decoder' if is_encoder_decoder else 'Decoder-Only (CausalLM)'}")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_checkpoint,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    # Special padding handling for Causal LMs
    if not is_encoder_decoder and tokenizer.pad_token is None:
        logger.warning("Tokenizer does not have a pad token. Setting pad_token = eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        config.pad_token_id = config.eos_token_id # Ensure model config also knows

    # Load Model
    model = model_class.from_pretrained(
        args.model_checkpoint,
        config=config,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if args.mixed_precision == 'bf16' and not (args.load_in_4bit or args.load_in_8bit) else (torch.float16 if args.mixed_precision == 'fp16' and not (args.load_in_4bit or args.load_in_8bit) else None), # Set dtype only if not quantizing
        trust_remote_code=args.trust_remote_code,
        # device_map="auto" # Often needed for large models, especially quantized
        # Consider adding device_map='auto' if running into OOM issues even with quantization/PEFT
    )

    # --- PEFT (LoRA / QLoRA) Setup ---
    if args.use_lora or args.use_qlora:
        logger.info(f"Applying {'QLoRA' if args.use_qlora else 'LoRA'}...")

        # Prepare model for k-bit training if quantized
        if args.load_in_4bit or args.load_in_8bit:
            logger.info("Preparing model for k-bit training.")
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=args.gradient_checkpointing # Use Trainer's arg
            )

        # Define LoRA target modules (can be model-specific)
        target_modules = args.lora_target_modules
        if target_modules is None:
            if "mistral" in args.model_checkpoint.lower() or "llama" in args.model_checkpoint.lower():
                 target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                 logger.info(f"Auto-detected target modules for Mistral/LLaMA: {target_modules}")
            elif "t5" in args.model_checkpoint.lower() or "bart" in args.model_checkpoint.lower():
                 target_modules = ["q", "v"] # Common for T5/BART attention
                 logger.info(f"Auto-detected target modules for T5/BART: {target_modules}")
            else:
                 logger.warning("Could not auto-detect LoRA target modules. LoRA might not be applied effectively. Specify --lora_target_modules.")
                 target_modules = [] # Avoid error, but LoRA won't do much

        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none", # Usually recommended
            task_type=TaskType.SEQ_2_SEQ_LM if is_encoder_decoder else TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, peft_config)
        logger.info("PEFT model created.")
        model.print_trainable_parameters()

    # Resize token embeddings if necessary (e.g., after adding special tokens)
    # embedding_size = model.get_input_embeddings().weight.shape[0]
    # if len(tokenizer) > embedding_size:
    #     logger.info("Resizing token embeddings...")
    #     model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer

# --- HPO Setup (Optuna) ---
def hp_space(trial: optuna.Trial):
    """Defines the hyperparameter search space for Optuna."""
    # Define constrained search spaces suitable for quick HPO
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True), # Focused range
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5), # Fewer epochs
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        "label_smoothing_factor": trial.suggest_float("label_smoothing_factor", 0.0, 0.15), # Slightly narrower
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.1), # Smaller warmup typical
        # "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16]), # If tuning batch size
    }

def compute_objective(metrics: dict):
    """Extracts the objective metric (SARI) from evaluation results."""
    # Optuna expects a single float value to maximize/minimize
    sari = metrics.get("eval_sari")
    if sari is None:
        logger.warning("Objective metric 'eval_sari' not found in metrics dict. Returning 0.0.")
        return 0.0
    return sari

# --- Main Function ---
def main():
    # Hugging Face Hub: set HF_TOKEN or HUGGING_FACE_HUB_TOKEN in the environment if login is required
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        try:
            login(token=token)
            logger.info("Successfully logged into Hugging Face Hub.")
        except Exception as e:
            logger.error(f"Failed to log into Hugging Face Hub: {e}")
    else:
        logger.info("HF_TOKEN not set; continuing without Hub login.")

    args = parse_args()

    # Setup logging levels for libraries
    transformers.utils.logging.set_verbosity_info()
    hf_datasets.utils.logging.set_verbosity_info() # Use set_verbosity_info or warning
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info("Starting simplification fine-tuning script...")
    logger.info(f"Script arguments: {args}")

    # Set seed before initializing model.
    set_seed(args.seed)

    # Load model and tokenizer
    logger.info(f"Loading model and tokenizer for '{args.model_checkpoint}'...")
    model, tokenizer = load_model_and_tokenizer(args)

    # Load and prepare datasets
    logger.info("Loading and preparing datasets...")
    processed_datasets = load_and_prepare_datasets(args, tokenizer)

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        pad_to_multiple_of=8 if args.mixed_precision != "no" else None, # Recommended for fp16/bf16
    )

    # Metrics function (get both potential functions)
    compute_metrics_seq2seq_fn, compute_metrics_causal_fn = get_compute_metrics_fn(tokenizer)

    # Determine if model is encoder-decoder
    is_encoder_decoder = model.config.is_encoder_decoder

    # --- Training Arguments ---
    common_args = {
        "output_dir": args.output_dir if not args.run_hpo else args.hpo_output_dir,
        "overwrite_output_dir": args.overwrite_output_dir,
        "do_train": args.do_train,
        "do_eval": args.do_eval,
        "do_predict": args.do_predict,
        # "evaluation_strategy": args.evaluation_strategy, # Handled below
        # "eval_steps": args.eval_steps if args.evaluation_strategy == "steps" else None, # Handled below
        # "save_strategy": args.save_strategy, # Handled below
        # "save_steps": args.save_steps if args.save_strategy == "steps" else None, # Handled below
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": args.load_best_model_at_end and not args.run_hpo, # Disable for HPO runs
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": args.gradient_checkpointing and not (args.use_lora or args.use_qlora),
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "logging_strategy": args.logging_strategy,
        "logging_steps": args.logging_steps,
        "report_to": args.report_to if not args.run_hpo else "none",
        "seed": args.seed,
        "fp16": args.mixed_precision == "fp16",
        "bf16": args.mixed_precision == "bf16",
        "optim": "adamw_bnb_8bit" if args.use_8bit_optimizer else "adamw_torch",
        # include_inputs_for_metrics will be added specifically for Seq2Seq below
    }

    if is_encoder_decoder:
        logger.info("Using Seq2SeqTrainingArguments and Seq2SeqTrainer.")
        training_args = Seq2SeqTrainingArguments(
            **common_args,
            evaluation_strategy=args.evaluation_strategy, # Pass strategy args here
            eval_steps=args.eval_steps if args.evaluation_strategy == "steps" else None,
            save_strategy=args.save_strategy,
            save_steps=args.save_steps if args.save_strategy == "steps" else None,
            label_smoothing_factor=args.label_smoothing_factor,
            predict_with_generate=args.predict_with_generate,
            generation_max_length=args.generation_max_length,
            generation_num_beams=args.generation_num_beams,
            metric_for_best_model=args.metric_for_best_model, # Use SARI by default for Seq2Seq
            greater_is_better=args.greater_is_better,
            include_inputs_for_metrics=True, # Needed for SARI calculation
        )
        compute_metrics_fn = compute_metrics_seq2seq_fn
        TrainerClass = Seq2SeqTrainer
    else:
        logger.info("Using TrainingArguments and Trainer (Causal LM). Evaluation metric set to 'loss'.")
        training_args = TrainingArguments(
            **common_args, # Pass common args directly
            # Ensure strategies match for load_best_model_at_end
            evaluation_strategy=args.evaluation_strategy,
            save_strategy=args.save_strategy,
            eval_steps=args.eval_steps if args.evaluation_strategy == "steps" else None, # Need eval_steps if strategy is steps
            save_steps=args.save_steps if args.save_strategy == "steps" else None, # Need save_steps if strategy is steps
            metric_for_best_model="loss", # Override metric for Causal LM
            greater_is_better=False, # Minimize loss
            # label_smoothing_factor not applicable
            # predict_with_generate not applicable
            # generation args not applicable here
        )
        compute_metrics_fn = compute_metrics_causal_fn # Use basic loss/perplexity
        TrainerClass = Trainer


    # --- Initialize Trainer ---
    # Need a model_init function for HPO (reloads model for each trial)
    def model_init_hpo(trial=None): # Add default None for non-HPO case
        # Reload the base model to reset weights for each trial or final run
        reloaded_model, _ = load_model_and_tokenizer(args)
        return reloaded_model

    trainer = TrainerClass( # Use the determined Trainer class
        model=model if not args.run_hpo else None,
        model_init=model_init_hpo if args.run_hpo else None,
        args=training_args,
        train_dataset=processed_datasets["train"] if args.do_train else None,
        eval_dataset=processed_datasets["validation"] if args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn, # Use the determined metrics function
    )

     # --- Run Training or HPO ---
    if args.run_hpo:
        logger.info(f"Starting hyperparameter optimization with {args.num_hpo_trials} trials...")
        # Note: Trainer's hyperparameter_search manages study creation/loading if needed
        best_run = trainer.hyperparameter_search(
            direction="maximize",
            backend="optuna",
            hp_space=hp_space,
            n_trials=args.num_hpo_trials,
            compute_objective=compute_objective,
            # study_name="simplification-hpo", # Optional name
            # storage=f"sqlite:///{os.path.join(args.hpo_output_dir, 'hpo_study.db')}", # Optional persistent storage
        )
        logger.info("Hyperparameter search finished.")
        logger.info(f"Best run found: Run ID {best_run.run_id}")
        logger.info(f"Best SARI score: {best_run.objective}")
        logger.info(f"Best hyperparameters: {best_run.hyperparameters}")

        # --- Train final model with best HPO params ---
        logger.info("Training final model with best hyperparameters found by HPO...")
        best_hyperparameters = best_run.hyperparameters

        # Update training args with best HPO params
        # Create a *new* TrainingArguments object (correct type)
        final_output_dir = os.path.join(args.output_dir, "final_model_hpo")
        ArgsClass = Seq2SeqTrainingArguments if is_encoder_decoder else TrainingArguments
        final_training_args = ArgsClass(
            output_dir=final_output_dir,
            **common_args, # Start with common args
        )
        # Update HPO'd args from best_run.hyperparameters
        # Need to handle potential differences in hp_space vs TrainingArguments names if any
        for key, value in best_hyperparameters.items():
             if hasattr(final_training_args, key):
                 setattr(final_training_args, key, value)
             else:
                 logger.warning(f"Hyperparameter '{key}' from HPO not found in TrainingArguments. Skipping.")

        # Add back Seq2Seq specific args if needed
        if is_encoder_decoder:
            final_training_args.label_smoothing_factor = best_hyperparameters.get("label_smoothing_factor", args.label_smoothing_factor) # Use HPO value or default
            final_training_args.predict_with_generate = args.predict_with_generate
            final_training_args.generation_max_length = args.generation_max_length
            final_training_args.generation_num_beams = args.generation_num_beams
            final_training_args.metric_for_best_model = args.metric_for_best_model
            final_training_args.greater_is_better = args.greater_is_better
        else:
            # Ensure strategies match for Causal LM HPO final run
            final_training_args.evaluation_strategy = args.evaluation_strategy
            final_training_args.save_strategy = args.save_strategy
            final_training_args.metric_for_best_model = "loss"
            final_training_args.greater_is_better = False


        # Ensure reporting is enabled for the final run
        final_training_args.report_to = args.report_to
        # Ensure best model loading is enabled
        final_training_args.load_best_model_at_end = True


        # Instantiate final trainer with a fresh model and best args
        final_model = model_init_hpo() # Get a fresh model instance
        final_trainer = TrainerClass( # Use correct Trainer class
            model=final_model,
            args=final_training_args,
            train_dataset=processed_datasets["train"] if args.do_train else None,
            eval_dataset=processed_datasets["validation"] if args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_fn, # Use correct metrics fn
        )

         # Train final model
        if args.do_train:
            logger.info("Starting final training run...")
            train_result = final_trainer.train()
            final_trainer.save_model() # Saves the best model
            final_trainer.log_metrics("train", train_result.metrics)
            final_trainer.save_metrics("train", train_result.metrics)
            final_trainer.save_state()
            logger.info(f"Final best model from HPO saved to {final_output_dir}")

        # Evaluate final model
        if args.do_eval:
            logger.info("Evaluating final model from HPO on validation set...")
            metrics = final_trainer.evaluate(metric_key_prefix="final_eval")
            final_trainer.log_metrics("final_eval", metrics)
            final_trainer.save_metrics("final_eval", metrics)

        # Predict with final model
        if args.do_predict:
             if "test" not in processed_datasets:
                 logger.warning("Test dataset not found or loaded. Skipping prediction.")
             else:
                logger.info("Running predictions with final model from HPO on test set...")
                predict_results = final_trainer.predict(
                    processed_datasets["test"], metric_key_prefix="final_predict"
                )
                metrics = predict_results.metrics
                final_trainer.log_metrics("final_predict", metrics)
                final_trainer.save_metrics("final_predict", metrics)

                # Save predictions
                if final_trainer.is_world_process_zero():
                    predictions = predict_results.predictions
                    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
                    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
                    output_predict_file = os.path.join(final_output_dir, "generated_predictions.txt")
                    with open(output_predict_file, "w", encoding="utf-8") as writer:
                        writer.write("\n".join(decoded_preds))
                    logger.info(f"Predictions saved to {output_predict_file}")

    else:
        # --- Standard Training Run ---
        if args.do_train:
            logger.info("Starting standard training run...")
            train_result = trainer.train()
            trainer.save_model() # Saves the best model if load_best_model_at_end=True
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
            trainer.save_state()
            logger.info(f"Training complete. Best model saved to {args.output_dir} (if load_best_model_at_end=True)")

        # Evaluate
        if args.do_eval:
            logger.info("Evaluating model on validation set...")
            metrics = trainer.evaluate(metric_key_prefix="eval")
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

        # Predict
        if args.do_predict:
            if "test" not in processed_datasets:
                 logger.warning("Test dataset not found or loaded. Skipping prediction.")
            else:
                logger.info("Running predictions on test set...")
                predict_results = trainer.predict(
                    processed_datasets["test"], metric_key_prefix="predict"
                )
                metrics = predict_results.metrics
                trainer.log_metrics("predict", metrics)
                trainer.save_metrics("predict", metrics)

                # Save predictions
                if trainer.is_world_process_zero():
                    predictions = predict_results.predictions
                    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
                    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
                    output_predict_file = os.path.join(args.output_dir, "generated_predictions.txt")
                    with open(output_predict_file, "w", encoding="utf-8") as writer:
                        writer.write("\n".join(decoded_preds))
                    logger.info(f"Predictions saved to {output_predict_file}")

    logger.info("Script finished.")


if __name__ == "__main__":
    main()
