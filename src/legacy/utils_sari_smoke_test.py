#!/usr/bin/env python
# coding: utf-8

"""Smoke test: load MultiCochrane English CSVs and compute SARI on a tiny sample."""

import os
import numpy as np
from datasets import load_dataset
import evaluate
import bootstrap_paths  # noqa: F401
from paths import MULTICOHRANE_EN_UNFILTERED

base_path = str(MULTICOHRANE_EN_UNFILTERED)
data_files = {
    "train": os.path.join(base_path, "train0_en.csv"),
    "test": os.path.join(base_path, "test0_en.csv"),
    "validation": os.path.join(base_path, "val0_en.csv")
}
multi_cochrane_dataset = load_dataset("csv", data_files=data_files)

# Log dataset sizes
print("Dataset sizes:")
for split, ds in multi_cochrane_dataset.items():
    print(f"{split}: {len(ds)} examples")

# --- Step 2: Calculate SARI Scores ---
sari_metric = evaluate.load("sari")

def calculate_sari_scores(dataset):
    sari_scores = []
    for example in dataset:
        source = example['input_text']
        prediction = example['target_text']  # Use target as "simplified" reference
        references = [[example['target_text']]]  # Reference is self for simplicity
        sari_score = sari_metric.compute(sources=[source], predictions=[prediction], references=references)["sari"]
        sari_scores.append(sari_score)
    return sari_scores

# Calculate SARI scores for each split
train_sari_scores = calculate_sari_scores(multi_cochrane_dataset["train"])
test_sari_scores = calculate_sari_scores(multi_cochrane_dataset["test"])
validation_sari_scores = calculate_sari_scores(multi_cochrane_dataset["validation"])

# Print average SARI scores for each split
print(f"Average SARI Score - Train: {np.mean(train_sari_scores)}")
print(f"Average SARI Score - Test: {np.mean(test_sari_scores)}")
print(f"Average SARI Score - Validation: {np.mean(validation_sari_scores)}")

# Optional: Print SARI scores for each sentence
# for i, score in enumerate(train_sari_scores):
#     print(f"Train sentence {i}: SARI = {score}")

#Optional: Saving the SARI scores to a file.
#with open ("sari_scores.txt","w") as f:
#   f.write(f"Average SARI Score - Train: {np.mean(train_sari_scores)}\n")
#   f.write(f"Average SARI Score - Test: {np.mean(test_sari_scores)}\n")
#   f.write(f"Average SARI Score - Validation: {np.mean(validation_sari_scores)}\n")
#   for i, score in enumerate(train_sari_scores):
#        f.write(f"Train sentence {i}: SARI = {score}\n")
#   for i, score in enumerate(test_sari_scores):
#        f.write(f"Test sentence {i}: SARI = {score}\n")
#   for i, score in enumerate(validation_sari_scores):
#        f.write(f"Validation sentence {i}: SARI = {score}\n")