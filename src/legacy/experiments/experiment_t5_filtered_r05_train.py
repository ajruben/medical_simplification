"""Legacy training script for filtered r0.5 MultiCochrane with manual optimization loop."""
import os
import torch
import random
import datasets
import transformers
import math

from datasets import load_dataset
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration, 
    set_seed, 
    get_scheduler
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.notebook import tqdm
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import bootstrap_paths  # noqa: F401
from paths import MULTICOHRANE_EN_FILTERED_R05, OUTPUT_ROOT

###############################################################################
# 1. Load your dataset
###############################################################################
base_path = str(MULTICOHRANE_EN_FILTERED_R05)
data_files = {
    "train": os.path.join(base_path, "train0.5_en.csv"),
    "validation": os.path.join(base_path, "val0.5_en.csv"),
    "test": os.path.join(base_path, "test0.5_en.csv")
}

try:
    multi_cochrane_dataset = load_dataset("csv", data_files=data_files)
    print("Dataset loaded successfully:")
    print(multi_cochrane_dataset)
except Exception as e:
    print(f"\nAn error occurred during dataset loading: {e}")
    raise

# Optional: Inspect a few samples to verify input vs. target
print("\nSample from training data:")
print("input_text:", multi_cochrane_dataset["train"][0]["input_text"])
print("target_text:", multi_cochrane_dataset["train"][0]["target_text"])

###############################################################################
# 2. Set up tokenizer & define preprocessing
###############################################################################
model_checkpoint = "google-t5/t5-small"  
tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)

# We'll prepend a "Simplify: " prefix so the model learns the task clearly
prefix = "Simplify: "
max_input_length = 512
max_target_length = 512

def preprocess_examples(examples):
    """
    Tokenize input_text as encoder input, 
    and target_text as decoder output (labels).
    """
    input_txt = examples["input_text"]
    target_txt = examples["target_text"]
    
    # Add prefix to each input
    inputs = [prefix + inp for inp in input_txt]
    
    # Tokenize
    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        padding="max_length",
        truncation=True
    )
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            target_txt,
            max_length=max_target_length,
            padding="max_length",
            truncation=True
        )["input_ids"]
    
    # Replace pad tokens with -100 so they are ignored by the loss
    labels_with_ignore_index = []
    for label_example in labels:
        label_example = [l if l != tokenizer.pad_token_id else -100 for l in label_example]
        labels_with_ignore_index.append(label_example)

    model_inputs["labels"] = labels_with_ignore_index
    return model_inputs

###############################################################################
# 3. Preprocess and encode the dataset
###############################################################################
train_ds = multi_cochrane_dataset["train"]
val_ds = multi_cochrane_dataset["validation"]
test_ds = multi_cochrane_dataset["test"]

encoded_train_ds = train_ds.map(
    preprocess_examples, 
    batched=True, 
    remove_columns=train_ds.column_names
)
encoded_val_ds = val_ds.map(
    preprocess_examples, 
    batched=True, 
    remove_columns=val_ds.column_names
)
encoded_test_ds = test_ds.map(
    preprocess_examples, 
    batched=True, 
    remove_columns=test_ds.column_names
)

# Set format to PyTorch Tensors
encoded_train_ds.set_format(type="torch")
encoded_val_ds.set_format(type="torch")
encoded_test_ds.set_format(type="torch")

###############################################################################
# 4. Create dataloaders
###############################################################################
def create_dataloaders(train_batch_size=32, eval_batch_size=32):
    train_dataloader = DataLoader(encoded_train_ds, shuffle=True, batch_size=train_batch_size)
    val_dataloader   = DataLoader(encoded_val_ds, shuffle=False, batch_size=eval_batch_size)
    return train_dataloader, val_dataloader

###############################################################################
# 5. Hyperparameters
###############################################################################
hyperparameters = {
    "model_checkpoint": model_checkpoint,  # Must match the tokenizer
    "learning_rate": 5e-5,
    "num_epochs": 5,
    "train_batch_size": 32,
    "gradient_accumulation_steps": 4,
    "eval_batch_size": 32,
    "seed": 42,
    "patience": 3, 
    "output_dir": str(OUTPUT_ROOT / "content_simplifier_t5_small_simple"),
    "mixed_precision": "fp16"  # or "no" if you're not on GPU w/ AMP
}

###############################################################################
# 6. Training function
###############################################################################
def training_function():
    accelerator = Accelerator(mixed_precision=hyperparameters["mixed_precision"])
    
    if accelerator.is_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    
    set_seed(hyperparameters["seed"])
    
    # Load model
    model = T5ForConditionalGeneration.from_pretrained(hyperparameters["model_checkpoint"])
    
    # Instantiate optimizer
    optimizer = AdamW(model.parameters(), lr=hyperparameters["learning_rate"])
    
    # Prepare dataloaders
    train_dataloader, val_dataloader = create_dataloaders(
        train_batch_size=hyperparameters["train_batch_size"],
        eval_batch_size=hyperparameters["eval_batch_size"]
    )
    
    # Prepare for multi-GPU / multi-TPU if applicable
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )
    
    epochs_no_improve = 0
    min_val_loss = float("inf")
    
    for epoch in range(hyperparameters["num_epochs"]):
        progress_bar = tqdm(range(len(train_dataloader)), disable=not accelerator.is_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        
        # Training
        model.train()
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            optimizer.zero_grad()
            progress_bar.set_postfix({'loss': loss.item()})
            progress_bar.update(1)
        
        # Evaluation
        model.eval()
        validation_losses = []
        for batch in val_dataloader:
            with torch.no_grad():
                outputs = model(**batch)
                val_loss = outputs.loss
            # Gather losses if using multiple processes
            validation_losses.append(accelerator.gather(val_loss[None]))
        
        # Compute average validation loss
        val_loss = torch.stack(validation_losses).sum().item() / len(validation_losses)
        accelerator.print(f"Epoch {epoch}: Validation Loss = {val_loss:.4f}")
        
        # Early stopping check
        if val_loss < min_val_loss:
            epochs_no_improve = 0
            min_val_loss = val_loss
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= hyperparameters["patience"]:
                accelerator.print("Early stopping triggered.")
                break
    
    # Save model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        hyperparameters["output_dir"], 
        save_function=accelerator.save
    )

# Call the training function once
training_function()

###############################################################################
# 7. Inference with the newly fine-tuned model
###############################################################################
# Load the trained model from disk
trained_model_path = hyperparameters["output_dir"]
trained_model = T5ForConditionalGeneration.from_pretrained(trained_model_path)
trained_tokenizer = T5Tokenizer.from_pretrained(trained_model_path)

sample_text = "The patient presented with refractory ventricular tachycardia."
prompt = "Simplify: " + sample_text

input_ids = trained_tokenizer(prompt, return_tensors="pt").input_ids

# Generate a simplified text; try beam search for stable output
generated_ids = trained_model.generate(
    input_ids=input_ids, 
    max_length=50,
    num_beams=4,
    early_stopping=True
)

summary = trained_tokenizer.decode(generated_ids.squeeze(), skip_special_tokens=True)
print("Original Text:", sample_text)
print("Simplified:", summary)
