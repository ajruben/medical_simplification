"""Training pipelines for each supported mode (WikiLarge and MultiCochrane).

Exposes four runners that configure Hugging Face Seq2SeqTrainer with Optuna where applicable,
plus helpers for cache directories and TrainingArguments compatibility. The MODES dictionary
maps CLI mode names to these callables.
"""
from __future__ import annotations

import inspect
import multiprocessing
import os
from typing import Any

import numpy as np
import optuna
import torch
from datasets import DatasetDict
from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    T5Tokenizer,
    set_seed,
)
import evaluate

from paths import MULTICOHRANE_EN_UNFILTERED, OUTPUT_ROOT, PROJECT_ROOT

from t5_train import data as D
from t5_train import metrics as M
from t5_train import presets as P


def _metric_include_kwargs() -> dict[str, Any]:
    """Return kwargs so eval passes inputs to metrics on both older and newer Transformers.

    Older versions use include_inputs_for_metrics; newer ones use include_for_metrics with a list.
    """
    sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    if "include_for_metrics" in sig.parameters:
        return {"include_for_metrics": ["inputs"]}
    if "include_inputs_for_metrics" in sig.parameters:
        return {"include_inputs_for_metrics": True}
    return {}


def _env_cache() -> None:
    """Ensure OUTPUT_ROOT exists; set Hugging Face cache env vars under project cache."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = str(PROJECT_ROOT / "cache/huggingface/datasets")
    os.environ["TRANSFORMERS_CACHE"] = str(PROJECT_ROOT / "cache/huggingface/transformers")


def run_wiki_small_sari() -> None:
    """Train t5-small on WikiLarge with Optuna over SARI then a long final run with best hyperparameters.

    Writes HPO checkpoints under presets WIKI_SMALL_OUT_HPO and the final model under
    WIKI_SMALL_OUT_FINAL. Wiki labels use zero masking for ignored positions.
    """
    _env_cache()
    wiki = D.load_wikilarge(P.WIKI_SMALL_PREFIX)
    tokenizer = T5Tokenizer.from_pretrained(P.WIKI_SMALL_MODEL)
    enc_train, enc_val, enc_test, _ = D.encode_wikilarge(
        wiki, tokenizer, 128, mask_pad_with="zero", num_proc=1
    )

    model_setup = {"model_checkpoint": P.WIKI_SMALL_MODEL, "learning_rate": 1e-5, "seed": 42}
    set_seed(model_setup["seed"])
    model = T5ForConditionalGeneration.from_pretrained(model_setup["model_checkpoint"])
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    compute_metrics_sari = M.make_sari_metrics(tokenizer)

    hyperparameters = {
        "train_batch_size": 64,
        "eval_batch_size": 64,
        "seed": 42,
        "mixed_precision": "fp16",
        "gradient_accumulation_steps": 2,
    }

    def model_init(trial):
        return T5ForConditionalGeneration.from_pretrained(model_setup["model_checkpoint"])

    def hp_space(trial: optuna.Trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-4, log=True),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 8),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
            "label_smoothing_factor": trial.suggest_float("label_smoothing_factor", 0.0, 0.2),
            "warmup_steps": trial.suggest_int("warmup_steps", 0, 1000),
            "lr_scheduler_type": trial.suggest_categorical(
                "lr_scheduler_type", ["linear", "cosine", "constant_with_warmup"]
            ),
        }

    def compute_objective(metrics: dict):
        return metrics.get("eval_sari", 0.0)

    training_args = Seq2SeqTrainingArguments(
        output_dir=P.WIKI_SMALL_OUT_HPO,
        overwrite_output_dir=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=hyperparameters["train_batch_size"],
        per_device_eval_batch_size=hyperparameters["eval_batch_size"],
        predict_with_generate=True,
        **_metric_include_kwargs(),
        gradient_accumulation_steps=hyperparameters["gradient_accumulation_steps"],
        generation_max_length=128,
        load_best_model_at_end=True,
        optim="adamw_torch",
        fp16=True,
        metric_for_best_model="sari",
        greater_is_better=True,
        report_to="none",
        seed=hyperparameters["seed"],
        save_total_limit=1,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model_init=model_init,
        train_dataset=enc_train,
        eval_dataset=enc_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_sari,
    )

    best_run = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=hp_space,
        n_trials=P.WIKI_SMALL_OPTUNA_TRIALS,
        compute_objective=compute_objective,
        study_name="t5-small-sari-sweep_general_simplification",
        load_if_exists=True,
    )
    print("Best SARI:", best_run.objective, best_run.hyperparameters)

    bh = best_run.hyperparameters
    if bh is None:
        raise RuntimeError("HPO returned no hyperparameters.")

    final_training_args = Seq2SeqTrainingArguments(
        output_dir=P.WIKI_SMALL_OUT_FINAL,
        overwrite_output_dir=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=hyperparameters["train_batch_size"],
        per_device_eval_batch_size=hyperparameters["eval_batch_size"],
        predict_with_generate=True,
        **_metric_include_kwargs(),
        gradient_accumulation_steps=hyperparameters["gradient_accumulation_steps"],
        generation_max_length=128,
        load_best_model_at_end=True,
        optim="adamw_torch",
        fp16=(hyperparameters["mixed_precision"] == "fp16"),
        metric_for_best_model="sari",
        greater_is_better=True,
        report_to="tensorboard",
        seed=hyperparameters["seed"],
        save_total_limit=2,
        logging_strategy="steps",
        logging_steps=100,
        num_train_epochs=25,
        learning_rate=bh["learning_rate"],
        lr_scheduler_type=bh["lr_scheduler_type"],
        warmup_steps=bh["warmup_steps"],
        weight_decay=bh["weight_decay"],
        label_smoothing_factor=bh["label_smoothing_factor"],
    )

    final_trainer = Seq2SeqTrainer(
        model=model_init(None),
        args=final_training_args,
        train_dataset=enc_train,
        eval_dataset=enc_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_sari,
    )
    final_trainer.train()
    final_trainer.save_model(P.WIKI_SMALL_OUT_FINAL)
    tokenizer.save_pretrained(P.WIKI_SMALL_OUT_FINAL)
    print("Saved to", P.WIKI_SMALL_OUT_FINAL)


def run_wiki_antiparrot() -> None:
    """Train t5-small on WikiLarge with an instructional prefix and penalized SARI objective.

    Uses a capped validation subset for speed, short Optuna search, then full-epoch training
    with dropout on the model config. Outputs go to WIKI_ANTIPARROT out directories in presets.
    """
    _env_cache()
    wiki = D.load_wikilarge(P.WIKI_ANTIPARROT_PREFIX)
    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
    enc_train, enc_val, enc_test, wiki = D.encode_wikilarge(
        wiki, tokenizer, 128, mask_pad_with="pad_id", num_proc=1
    )
    if len(enc_val) > 2000:
        enc_val = enc_val.select(range(2000))

    model_setup = {"model_checkpoint": "google-t5/t5-small", "learning_rate": 5e-5, "seed": 42}
    set_seed(model_setup["seed"])
    compute_fn = M.make_penalized_sari_metrics(tokenizer, P.WIKI_ANTIPARROT_PREFIX)

    def model_init(trial):
        m = T5ForConditionalGeneration.from_pretrained(model_setup["model_checkpoint"])
        m.config.dropout_rate = 0.2
        return m

    def hp_space(trial: optuna.Trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 2e-3, log=True),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 5),
            "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.1),
            "label_smoothing_factor": trial.suggest_float("label_smoothing_factor", 0.1, 0.2),
            "warmup_ratio": 0.1,
            "lr_scheduler_type": "linear",
        }

    def compute_objective(metrics: dict):
        return metrics.get("eval_penalized_sari", 0.0)

    training_args = Seq2SeqTrainingArguments(
        output_dir=P.WIKI_ANTIPARROT_OUT_HPO,
        overwrite_output_dir=True,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        predict_with_generate=True,
        **_metric_include_kwargs(),
        gradient_accumulation_steps=2,
        generation_max_length=128,
        load_best_model_at_end=True,
        optim="adamw_torch",
        fp16=True,
        metric_for_best_model="penalized_sari",
        greater_is_better=True,
        report_to="none",
        seed=model_setup["seed"],
        save_total_limit=1,
        max_grad_norm=1.0,
        max_steps=2000,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model_init=model_init,
        train_dataset=enc_train,
        eval_dataset=enc_val,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=T5ForConditionalGeneration.from_pretrained(model_setup["model_checkpoint"]),
        ),
        tokenizer=tokenizer,
        compute_metrics=compute_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    best_run = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=hp_space,
        n_trials=5,
        compute_objective=compute_objective,
        study_name="t5-small-antiparrot-simplification-fast",
    )
    bh = best_run.hyperparameters
    final_training_args = Seq2SeqTrainingArguments(
        output_dir=P.WIKI_ANTIPARROT_OUT_FINAL,
        overwrite_output_dir=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        predict_with_generate=True,
        **_metric_include_kwargs(),
        gradient_accumulation_steps=2,
        generation_max_length=128,
        load_best_model_at_end=True,
        optim="adamw_torch",
        fp16=True,
        metric_for_best_model="penalized_sari",
        greater_is_better=True,
        report_to="tensorboard",
        seed=model_setup["seed"],
        save_total_limit=2,
        logging_strategy="steps",
        logging_steps=100,
        num_train_epochs=bh["num_train_epochs"] + 1,
        learning_rate=bh["learning_rate"],
        lr_scheduler_type=bh.get("lr_scheduler_type", "linear"),
        warmup_ratio=bh.get("warmup_ratio", 0.1),
        weight_decay=bh["weight_decay"],
        label_smoothing_factor=bh["label_smoothing_factor"],
    )

    final_model = T5ForConditionalGeneration.from_pretrained(model_setup["model_checkpoint"])
    final_model.config.dropout_rate = 0.2
    final_trainer = Seq2SeqTrainer(
        model=final_model,
        args=final_training_args,
        train_dataset=enc_train,
        eval_dataset=enc_val,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=final_model),
        tokenizer=tokenizer,
        compute_metrics=compute_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    final_trainer.train()
    final_trainer.save_model(P.WIKI_ANTIPARROT_OUT_FINAL)
    tokenizer.save_pretrained(P.WIKI_ANTIPARROT_OUT_FINAL)
    print("Saved to", P.WIKI_ANTIPARROT_OUT_FINAL)


def run_large_medical_sari() -> None:
    """Fine-tune t5-large on unfiltered English MultiCochrane CSVs with SARI-driven Optuna and final train.

    Loads data via paths MULTICOHRANE_EN_UNFILTERED, adds LARGE_PREFIX, enables gradient
    checkpointing on the model during search, and saves to LARGE_OUT_HPO and LARGE_OUT_FINAL.
    """
    _env_cache()
    SEED = 42
    set_seed(SEED)
    base_path = str(MULTICOHRANE_EN_UNFILTERED)
    mc = D.load_multicochrane_csv(base_path)
    mc = D.add_prefix_column(mc, P.LARGE_PREFIX)
    tokenizer = T5Tokenizer.from_pretrained(P.LARGE_MODEL)
    max_length = 128

    encoded: DatasetDict = DatasetDict(
        {
            s: mc[s].map(
                lambda b: D.preprocess_medical_examples(b, tokenizer, max_length),
                batched=True,
                remove_columns=mc[s].column_names,
            )
            for s in mc
        }
    )

    sari_metric = evaluate.load("sari")

    def compute_metrics_sari(eval_preds):
        predictions, labels, inputs = eval_preds.predictions, eval_preds.label_ids, eval_preds.inputs
        predictions = np.clip(predictions, 0, tokenizer.vocab_size - 1)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        inputs = np.where(inputs != -100, inputs, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)
        all_refs = [[lbl] for lbl in decoded_labels]
        return {"sari": sari_metric.compute(sources=decoded_inputs, predictions=decoded_preds, references=all_refs)["sari"]}

    hyperparameters = {
        "model_checkpoint": P.LARGE_MODEL,
        "train_batch_size": 8,
        "eval_batch_size": 16,
        "gradient_accumulation_steps": 2,
        "mixed_precision": "fp16",
        "seed": SEED,
    }

    def model_init(trial):
        m = T5ForConditionalGeneration.from_pretrained(hyperparameters["model_checkpoint"])
        m.gradient_checkpointing_enable()
        return m

    def hp_space(trial: optuna.Trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True),
            "num_train_epochs": 2,
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
            "label_smoothing_factor": trial.suggest_float("label_smoothing_factor", 0.0, 0.2),
            "warmup_steps": trial.suggest_int("warmup_steps", 0, 1000),
            "lr_scheduler_type": trial.suggest_categorical(
                "lr_scheduler_type", ["linear", "cosine", "constant_with_warmup"]
            ),
        }

    training_args = Seq2SeqTrainingArguments(
        output_dir=P.LARGE_OUT_HPO,
        overwrite_output_dir=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=hyperparameters["train_batch_size"],
        per_device_eval_batch_size=hyperparameters["eval_batch_size"],
        gradient_accumulation_steps=hyperparameters["gradient_accumulation_steps"],
        predict_with_generate=True,
        **_metric_include_kwargs(),
        generation_max_length=max_length,
        generation_num_beams=4,
        load_best_model_at_end=True,
        optim="adamw_torch",
        fp16=False,
        metric_for_best_model="sari",
        greater_is_better=True,
        report_to="none",
        seed=SEED,
        save_total_limit=1,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model_init=model_init,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_init(None)),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_sari,
    )

    best_run = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=hp_space,
        n_trials=20,
        compute_objective=lambda m: m.get("eval_sari", 0.0),
    )
    bh = best_run.hyperparameters

    final_training_args = Seq2SeqTrainingArguments(
        output_dir=P.LARGE_OUT_FINAL,
        overwrite_output_dir=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=25,
        learning_rate=bh["learning_rate"],
        lr_scheduler_type=bh["lr_scheduler_type"],
        warmup_steps=bh["warmup_steps"],
        weight_decay=bh["weight_decay"],
        label_smoothing_factor=bh["label_smoothing_factor"],
        per_device_train_batch_size=hyperparameters["train_batch_size"],
        per_device_eval_batch_size=hyperparameters["eval_batch_size"],
        gradient_accumulation_steps=hyperparameters["gradient_accumulation_steps"],
        predict_with_generate=True,
        **_metric_include_kwargs(),
        generation_max_length=max_length,
        generation_num_beams=4,
        load_best_model_at_end=True,
        optim="adamw_torch",
        fp16=True,
        metric_for_best_model="sari",
        greater_is_better=True,
        report_to="tensorboard",
        seed=SEED,
        save_total_limit=2,
        logging_strategy="steps",
        logging_steps=100,
    )

    ft = Seq2SeqTrainer(
        model=model_init(None),
        args=final_training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_init(None)),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_sari,
    )
    ft.train()
    ft.save_model(P.LARGE_OUT_FINAL)
    tokenizer.save_pretrained(P.LARGE_OUT_FINAL)
    print("Saved to", P.LARGE_OUT_FINAL)


def run_medical_combined() -> None:
    """Continue from MEDICAL_BASE_CHECKPOINT on MultiCochrane with CombinedScoreMetrics and Optuna.

    Uses a subset of validation during search for speed, then full validation for the final run.
    Objective is eval_combined_score. Writes to MEDICAL_HPO_OUT and MEDICAL_FINAL_OUT.
    """
    _env_cache()
    global_tok_holder = M.CombinedScoreMetrics(P.MEDICAL_PREFIX)

    raw = D.load_multicochrane_csv(str(MULTICOHRANE_EN_UNFILTERED))
    mc = D.add_prefix_column(raw, P.MEDICAL_PREFIX)
    model_checkpoint = P.MEDICAL_BASE_CHECKPOINT
    tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
    global_tok_holder.set_tokenizer(tokenizer)

    max_length = 128
    encoded = D.encode_multicochrane(mc, tokenizer, max_length, num_proc=1)
    eval_sub = min(1000, len(encoded["validation"]))
    enc_val_sub = encoded["validation"].select(range(eval_sub))

    model_setup = {"model_checkpoint": model_checkpoint, "seed": 42}
    set_seed(model_setup["seed"])

    def model_init(trial):
        return T5ForConditionalGeneration.from_pretrained(model_setup["model_checkpoint"])

    def hp_space(trial: optuna.Trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 6),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
            "label_smoothing_factor": trial.suggest_float("label_smoothing_factor", 0.0, 0.15),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
            "lr_scheduler_type": trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine"]),
            "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [2, 4, 8]),
        }

    def compute_objective(metrics: dict):
        return metrics.get("eval_combined_score", 0.0)

    hpo_args = Seq2SeqTrainingArguments(
        output_dir=P.MEDICAL_HPO_OUT,
        overwrite_output_dir=True,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=100,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        predict_with_generate=True,
        **_metric_include_kwargs(),
        generation_max_length=max_length,
        load_best_model_at_end=False,
        optim="adamw_torch",
        fp16=True,
        metric_for_best_model="combined_score",
        greater_is_better=True,
        report_to="none",
        seed=model_setup["seed"],
        max_steps=1500,
    )

    hpo_trainer = Seq2SeqTrainer(
        args=hpo_args,
        model_init=model_init,
        train_dataset=encoded["train"],
        eval_dataset=enc_val_sub,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        tokenizer=tokenizer,
        compute_metrics=global_tok_holder,
    )

    best_run = hpo_trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=hp_space,
        n_trials=10,
        compute_objective=compute_objective,
        study_name="medical-finetune-t5-small-hpo",
        gc_after_trial=True,
    )
    bh = best_run.hyperparameters

    final_args = Seq2SeqTrainingArguments(
        output_dir=P.MEDICAL_FINAL_OUT,
        overwrite_output_dir=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=bh["gradient_accumulation_steps"],
        predict_with_generate=True,
        **_metric_include_kwargs(),
        generation_max_length=max_length,
        load_best_model_at_end=True,
        optim="adamw_torch",
        fp16=True,
        metric_for_best_model="combined_score",
        greater_is_better=True,
        report_to="tensorboard",
        seed=model_setup["seed"],
        save_total_limit=2,
        logging_strategy="steps",
        logging_steps=100,
        num_train_epochs=bh["num_train_epochs"],
        learning_rate=bh["learning_rate"],
        lr_scheduler_type=bh["lr_scheduler_type"],
        warmup_ratio=bh["warmup_ratio"],
        weight_decay=bh["weight_decay"],
        label_smoothing_factor=bh["label_smoothing_factor"],
    )

    final_model = T5ForConditionalGeneration.from_pretrained(model_setup["model_checkpoint"])
    final_trainer = Seq2SeqTrainer(
        model=final_model,
        args=final_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=final_model),
        tokenizer=tokenizer,
        compute_metrics=global_tok_holder,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)],
    )
    final_trainer.train()
    final_trainer.save_model(P.MEDICAL_FINAL_OUT)
    tokenizer.save_pretrained(P.MEDICAL_FINAL_OUT)
    print("Saved to", P.MEDICAL_FINAL_OUT)


# Keys match train_t5.py --mode and t5_train.cli choices.
MODES: dict[str, Any] = {
    "wiki_small_sari": run_wiki_small_sari,
    "wiki_antiparrot": run_wiki_antiparrot,
    "large_medical_sari": run_large_medical_sari,
    "medical_combined": run_medical_combined,
}
