"""Load a GEM WikiLarge-style dataset id and print split sizes (helper for data inspection)."""
from datasets import load_dataset
import os
import bootstrap_paths  # noqa: F401
from paths import PROJECT_ROOT

os.environ["HF_DATASETS_CACHE"] = str(PROJECT_ROOT / "cache/huggingface/datasets")

DATASET_NAME = "gem/wiki_large" # Trying standard GEM benchmark identifier

try:
    print(f"Loading dataset info for {DATASET_NAME}...")
    # Load the dataset - this might download metadata first
    dataset = load_dataset(DATASET_NAME)

    print("\nDataset sizes:")
    for split_name in dataset.keys():
        print(f"- {split_name}: {len(dataset[split_name])} samples")

except Exception as e:
    print(f"Error loading dataset {DATASET_NAME}: {e}")
