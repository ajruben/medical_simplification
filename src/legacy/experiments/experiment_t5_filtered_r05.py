"""Legacy experiment: T5 training loop on MultiCochrane filtered at r equals 0.5."""
import torch
from transformers import T5ForConditionalGeneration, AdamW, set_seed
from accelerate import Accelerator
from tqdm.notebook import tqdm
import datasets
import transformers
import requests
import os
import datetime
import random
import zipfile
import shutil
import math
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW # Changed from transformers import AdamW
import datasets as hf_datasets # Alias to avoid conflict with local variables
import transformers as hf_transformers # Alias
from datasets import Dataset, DatasetDict, load_dataset
from transformers import T5Tokenizer, T5TokenizerFast, T5ForConditionalGeneration, set_seed, get_scheduler
from accelerate import Accelerator, notebook_launcher
from tqdm.notebook import tqdm # Or from tqdm import tqdm
import importlib.metadata # For version checking
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import bootstrap_paths  # noqa: F401
from paths import MULTICOHRANE_EN_FILTERED_R05, OUTPUT_ROOT

base_path = str(MULTICOHRANE_EN_FILTERED_R05)
data_files = {
    "train": os.path.join(base_path, "train0.5_en.csv"),
    "test": os.path.join(base_path, "test0.5_en.csv"),
    "validation": os.path.join(base_path, "val0.5_en.csv") 
}

try:
    multi_cochrane_dataset = load_dataset("csv", data_files=data_files)
    print(multi_cochrane_dataset)
except Exception as e:
    print(f"\nAn error occurred during dataset loading: {e}")

#based on https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/T5/Fine_tuning_Dutch_T5_base_on_CNN_Daily_Mail_for_summarization_(on_TPU_using_HuggingFace_Accelerate).ipynb#scrollTo=tiLdcTmkg-_o
from transformers import T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")

prefix = "Simplify: "
max_input_length = 512
max_target_length = 512

def preprocess_examples(examples):
  # encode the documents
  input_txt = examples['input_text']
  target_txt = examples['target_text']
  
  inputs = [prefix + inp for inp in input_txt]
  model_inputs = tokenizer(inputs, max_length=max_input_length, padding="max_length", truncation=True)

  # encode the simplifications
  labels = tokenizer(target_txt, max_length=max_target_length, padding="max_length", truncation=True).input_ids

  # important: we need to replace the index of the padding tokens by -100
  # such that they are not taken into account by the CrossEntropyLoss
  labels_with_ignore_index = []
  for labels_example in labels:
    labels_example = [label if label != 0 else -100 for label in labels_example]
    labels_with_ignore_index.append(labels_example)
  
  model_inputs["labels"] = labels_with_ignore_index

  return model_inputs

train_ds = multi_cochrane_dataset['train']
val_ds = multi_cochrane_dataset['validation']
test_ds = multi_cochrane_dataset['test']
encoded_train_ds = train_ds.map(preprocess_examples, batched=True, remove_columns=train_ds.column_names)
encoded_val_ds = val_ds.map(preprocess_examples, batched=True, remove_columns=val_ds.column_names)
encoded_test_ds = test_ds.map(preprocess_examples, batched=True, remove_columns=test_ds.column_names)

#set format to PyTorch
encoded_train_ds.set_format(type="torch")
encoded_val_ds.set_format(type="torch")
encoded_test_ds.set_format(type="torch")

from torch.utils.data import DataLoader

def create_dataloaders(train_batch_size=8, eval_batch_size=32):
    train_dataloader = DataLoader(encoded_train_ds, shuffle=True, batch_size=train_batch_size)
    val_dataloader = DataLoader(encoded_val_ds, shuffle=False, batch_size=eval_batch_size)
    return train_dataloader, val_dataloader

hyperparameters = {
    "model_checkpoint": "google-t5/t5-small",
    "learning_rate": 5e-5, 
    "num_epochs": 3, 
    "train_batch_size": 8, 
    "gradient_accumulation_steps": 4,
    "eval_batch_size": 16, 
    "seed": 42,
    "patience": 3,
    "output_dir": str(OUTPUT_ROOT / "content_simplifier_t5_small_new_data"),
    "mixed_precision": "fp16",
}



def training_function():
    # Initialize accelerator
    accelerator = Accelerator()

    # To have only one message (and not 8) per logs of Transformers or Datasets, we set the logging verbosity
    # to INFO for the main process only.
    if accelerator.is_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # The seed need to be set before we instantiate the model, as it will determine the random head.
    set_seed(hyperparameters["seed"])

    # Instantiate the model, let Accelerate handle the device placement.
    model = T5ForConditionalGeneration.from_pretrained(hyperparameters["model_checkpoint"])

    # Instantiate optimizer
    optimizer = AdamW(model.parameters(), lr=hyperparameters["learning_rate"])

    # Prepare everything
    train_dataloader, val_dataloader = create_dataloaders(
        train_batch_size=hyperparameters["train_batch_size"], eval_batch_size=hyperparameters["eval_batch_size"]
    )
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(model, optimizer, 
                                                                             train_dataloader, val_dataloader)
    
    # Now we train the model
    epochs_no_improve = 0
    min_val_loss = 1000000
    for epoch in range(hyperparameters["num_epochs"]):
        # We only enable the progress bar on the main process to avoid having 8 progress bars.
        progress_bar = tqdm(range(len(train_dataloader)), disable=not accelerator.is_main_process)
        progress_bar.set_description(f"Epoch: {epoch}")
        model.train()
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.set_postfix({'loss': loss.item()})
            progress_bar.update(1)

        # Evaluate at the end of the epoch (distributed evaluation as we have 8 TPU cores)
        model.eval()
        validation_losses = []
        for batch in val_dataloader:
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss

            # We gather the loss from the 8 TPU cores to have them all.
            validation_losses.append(accelerator.gather(loss[None]))

        # Compute average validation loss
        val_loss = torch.stack(validation_losses).sum().item() / len(validation_losses)
        # Use accelerator.print to print only on the main process.
        accelerator.print(f"epoch {epoch}: validation loss:", val_loss)
        if val_loss < min_val_loss:
          epochs_no_improve = 0
          min_val_loss = val_loss
          continue
        else:
          epochs_no_improve += 1
          # Check early stopping condition
          if epochs_no_improve == hyperparameters["patience"]:
            accelerator.print("Early stopping!")
            break

    # save trained model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    # Use accelerator.save to save
    unwrapped_model.save_pretrained(hyperparameters["output_dir"], save_function=accelerator.save)

training_function()

text = """Simplify: The patient presented with refractory ventricular tachycardia""" 

trained_model = T5ForConditionalGeneration.from_pretrained(str(OUTPUT_ROOT / "content_simplifier_t5_base_simple"))

input_ids = tokenizer(text, return_tensors="pt").input_ids
 
generated_ids = trained_model.generate(input_ids, do_sample=True, 
    max_length=50, 
    top_k=0, 
    temperature=0.7
)

summary = tokenizer.decode(generated_ids.squeeze(), skip_special_tokens=True)
print(summary)