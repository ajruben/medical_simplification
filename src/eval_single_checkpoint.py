"""Evaluate one seq2seq checkpoint on combined MultiCochrane splits: BLEU, ROUGE, SARI, Jaccard.

Loads configuration from module-level paths, generates batched predictions, prints scores, and
appends a row to the checkpoint evaluation CSV. Intended as a script; adjust MODEL_PATH and paths
for your run.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed
from datasets import load_dataset
import evaluate
import nltk
import os
from tqdm import tqdm
import gc
import pandas as pd # Import pandas
from pathlib import Path

from paths import CHECKPOINT_EVAL_CSV, MULTICOHRANE_EN_UNFILTERED, OUTPUT_ROOT

# --- Configuration ---
OUTPUT_CSV_FILE = str(CHECKPOINT_EVAL_CSV)
Path(OUTPUT_CSV_FILE).parent.mkdir(parents=True, exist_ok=True)

MODEL_PATH = str(
    OUTPUT_ROOT / "final_t5_small_opt_sweep_pretrained_wiki_run4_check960_RUN2" / "checkpoint-57360"
)
DATA_BASE_PATH = str(MULTICOHRANE_EN_UNFILTERED)
TRAIN_FILE = os.path.join(DATA_BASE_PATH, "train0_en.csv")
VALIDATION_FILE = os.path.join(DATA_BASE_PATH, "val0_en.csv")
TEST_FILE = os.path.join(DATA_BASE_PATH, "test0_en.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIMPLIFICATION_PREFIX = "simplify medical text without copying: "
MAX_LENGTH = 128
BATCH_SIZE = 8 # Keep batch size reasonable
NUM_SAMPLES_VALIDATION = None # Use the full validation set

# Ensure necessary NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)
    print("Download complete.")

set_seed(42)

def preprocess_text_for_metric(texts):
    """Normalize strings or lists for BLEU, ROUGE, and SARI: sentence tokenization and newlines."""
    if isinstance(texts, str):
        return "\n".join(nltk.sent_tokenize(texts.strip()))
    elif isinstance(texts, list):
        # Handle list of lists for references
        if texts and isinstance(texts[0], list):
             return [["\n".join(nltk.sent_tokenize(ref.strip())) for ref in ref_list] for ref_list in texts]
        else:
             return ["\n".join(nltk.sent_tokenize(text.strip())) for text in texts]
    return texts # Return as is if format is unexpected

def calculate_jaccard(text1, text2):
    """Jaccard index over lowercased word sets. Returns None if NLTK tokenization is unavailable."""
    if not text1 or not text2:
        return 0.0
    # Ensure NLTK is available or handle error
    try:
        set1 = set(nltk.word_tokenize(text1.lower()))
        set2 = set(nltk.word_tokenize(text2.lower()))
    except LookupError:
        print("NLTK punkt tokenizer not found. Please ensure it's downloaded.")
        return None # Indicate error
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

print(f"--- Evaluating Model: {MODEL_PATH} ---") # Changed Checkpoint to Model
print(f"Using device: {DEVICE}")

# --- Load Data ---
print("Loading train, validation, and test datasets...")
from datasets import concatenate_datasets, DatasetDict

try:
    data_files = {}
    if os.path.exists(TRAIN_FILE): data_files["train"] = TRAIN_FILE
    if os.path.exists(VALIDATION_FILE): data_files["validation"] = VALIDATION_FILE
    if os.path.exists(TEST_FILE): data_files["test"] = TEST_FILE

    if not data_files:
        raise FileNotFoundError("No data files found in the specified path.")

    raw_datasets = load_dataset("csv", data_files=data_files)
    print(f"Loaded splits: {list(raw_datasets.keys())}")

    combined_splits = []
    for split_name, ds in raw_datasets.items():
        print(f"Processing split: {split_name}")
        # Rename columns if necessary
        current_cols = ds.column_names
        # Correct indentation: This block should be inside the loop
        if "Expert" in current_cols and "Simple" in current_cols:
            ds = ds.rename_column("Expert", "input_text")
            ds = ds.rename_column("Simple", "target_text")
            print(f"Renamed columns in {split_name} split.")
        elif "input_text" not in current_cols or "target_text" not in current_cols:
             raise ValueError(f"Expected columns ('Expert', 'Simple' or 'input_text', 'target_text') not found in {split_name}. Found: {current_cols}")
        # Correct indentation: Append ds after processing within the loop
        combined_splits.append(ds)

    # Combine all loaded splits
    if not combined_splits:
        raise ValueError("No data splits could be loaded and processed.")

    evaluation_dataset = concatenate_datasets(combined_splits)
    print(f"Combined dataset for evaluation has {len(evaluation_dataset)} samples.")

    # Override NUM_SAMPLES_VALIDATION as we are using all data
    NUM_SAMPLES_VALIDATION = None
    eval_validation_dataset = evaluation_dataset # Use the combined dataset

except Exception as e:
    print(f"Error loading or processing dataset: {e}")
    exit()

# --- Load Metrics ---
print("Loading evaluation metrics (BLEU, ROUGE, SARI)...")
try:
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    sari_metric = evaluate.load("sari") # Added SARI
    print("Metrics loaded successfully.")
except Exception as e:
    print(f"Error loading metrics: {e}")
    exit()

# --- Load Model and Tokenizer ---
model = None
tokenizer = None
try:
    print(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print(f"Loading model from {MODEL_PATH}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    print("Model and tokenizer loaded.")
except Exception as e:
    print(f"Error loading model or tokenizer from {MODEL_PATH}: {e}")
    exit()

# --- Generate Predictions ---
print("Generating predictions on validation set...")
val_inputs = [SIMPLIFICATION_PREFIX + text for text in eval_validation_dataset['input_text']]
val_references_raw = eval_validation_dataset['target_text']
val_sources_raw = eval_validation_dataset['input_text'] # Needed for SARI
val_predictions_raw = []

try:
    with torch.no_grad():
        for i in tqdm(range(0, len(val_inputs), BATCH_SIZE), desc=f"Validation Gen ({os.path.basename(MODEL_PATH)})"):
            batch_inputs = val_inputs[i:i+BATCH_SIZE]
            inputs_tokenized = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(DEVICE)
            outputs_tokenized = model.generate(
                **inputs_tokenized,
                max_length=MAX_LENGTH,
                num_beams=4,
                early_stopping=True
            )
            batch_preds = tokenizer.batch_decode(outputs_tokenized, skip_special_tokens=True)
            val_predictions_raw.extend(batch_preds)
except Exception as e:
    print(f"Error during prediction generation: {e}")
    # Clean up before exiting
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    exit()

# --- Calculate Metrics ---
print("Calculating metrics...")
val_predictions_pp = preprocess_text_for_metric(val_predictions_raw)
val_references_pp = preprocess_text_for_metric([[r] for r in val_references_raw]) # List of lists for metrics
val_sources_pp = preprocess_text_for_metric(val_sources_raw) # Preprocess sources for SARI

bleu_score = None
rouge_scores = None
sari_score = None # Initialize SARI score
jaccard_input_output_score = None # Initialize Jaccard score

try:
    bleu_results = bleu_metric.compute(predictions=val_predictions_pp, references=val_references_pp)
    bleu_score = bleu_results.get('bleu', 0.0)
    print(f"\n--- Results for {MODEL_PATH} ---")
    print(f"  BLEU: {bleu_score:.4f}")
except Exception as e:
    print(f"  Error calculating BLEU: {e}")

try:
    rouge_results = rouge_metric.compute(predictions=val_predictions_pp, references=val_references_pp)
    rouge_scores = {
        'rouge1': rouge_results.get('rouge1', 0.0),
        'rouge2': rouge_results.get('rouge2', 0.0),
        'rougeL': rouge_results.get('rougeL', 0.0)
    }
    print(f"  ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"  ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"  ROUGE-L: {rouge_scores['rougeL']:.4f}")
except Exception as e:
    print(f"  Error calculating ROUGE: {e}")

try:
    sari_results = sari_metric.compute(sources=val_sources_pp, predictions=val_predictions_pp, references=val_references_pp)
    sari_score = sari_results.get('sari', 0.0)
    print(f"  SARI: {sari_score:.4f}")
except Exception as e:
    print(f"  Error calculating SARI: {e}")

# Calculate Jaccard Similarity (Input vs Output)
try:
    jaccard_scores_list = []
    # Remove prefix from original inputs for comparison
    original_inputs_no_prefix = [inp.replace(SIMPLIFICATION_PREFIX, '', 1) for inp in val_inputs]
    for original_input, pred in zip(original_inputs_no_prefix, val_predictions_raw):
         score = calculate_jaccard(original_input, pred)
         if score is not None:
             jaccard_scores_list.append(score)

    if jaccard_scores_list:
        jaccard_input_output_score = sum(jaccard_scores_list) / len(jaccard_scores_list)
        print(f"  Jaccard (Input vs Output): {jaccard_input_output_score:.4f}")
    else:
        print("  Could not calculate Jaccard similarity (Input vs Output).")
except Exception as e:
    print(f"  Error calculating Jaccard similarity (Input vs Output): {e}")


print("------------------------------------")

# --- Save Results to CSV ---
print(f"Saving results to {OUTPUT_CSV_FILE}...")
results_data = {
    'model_path': [MODEL_PATH],
    'bleu': [bleu_score],
    'rouge1': [rouge_scores['rouge1'] if rouge_scores else None],
    'rouge2': [rouge_scores['rouge2'] if rouge_scores else None],
    'rougeL': [rouge_scores['rougeL'] if rouge_scores else None],
    'sari': [sari_score],
    'jaccard_similarity_input_output': [jaccard_input_output_score], # Add Jaccard score
    'all_data': [1] # Add the new flag column
}
new_results_df = pd.DataFrame(results_data)

try:
    if os.path.exists(OUTPUT_CSV_FILE):
        print("Appending results to existing CSV...")
        existing_df = pd.read_csv(OUTPUT_CSV_FILE)
        # Add 'all_data' column to existing_df if it doesn't exist, fill with 0 or NaN
        if 'all_data' not in existing_df.columns:
            existing_df['all_data'] = 0 # Or np.nan if preferred

        # Ensure columns match before concatenating (reindex based on union of columns)
        all_cols = list(existing_df.columns.union(new_results_df.columns))
        existing_df = existing_df.reindex(columns=all_cols)
        new_results_df = new_results_df.reindex(columns=all_cols)

        combined_df = pd.concat([existing_df, new_results_df], ignore_index=True)
        # Remove duplicates based on model_path AND all_data flag, keeping the latest run
        combined_df.drop_duplicates(subset=['model_path', 'all_data'], keep='last', inplace=True)
        combined_df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8')
    else:
        print("Creating new CSV file for results...")
        new_results_df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8')
    print("Results saved successfully.")
except Exception as e:
    print(f"Error saving results to CSV: {e}")


# --- Clean up ---
del model
del tokenizer
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
print("Evaluation complete. Memory cleaned.")
