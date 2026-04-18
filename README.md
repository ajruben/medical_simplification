# Medical text simplification (T5)

T5-based sentence simplification on WikiLarge (general domain) and MultiCochrane English CSVs (medical). Training code lives under the src directory; datasets under data at the repo root.

## Requirements

Python 3.10+ with PyTorch, transformers, datasets, evaluate, optuna, accelerate, nltk, and usual scientific dependencies. Install versions compatible with your CUDA setup.

## Training entrypoint

From the repository root:

    python src/train_t5.py --mode MODE

Or from the src directory:

    cd src
    python train_t5.py --mode MODE

Equivalent: python -m t5_train.cli --mode MODE with src on PYTHONPATH (the scripts above add src automatically).

### Modes

| Mode | Description |
|------|-------------|
| wiki_small_sari | WikiLarge, google-t5/t5-small, SARI plus Optuna, then final run |
| wiki_antiparrot | WikiLarge, anti-parrot-style prefix, penalized SARI plus Optuna |
| large_medical_sari | MultiCochrane EN unfiltered, google-t5/t5-large, SARI plus Optuna |
| medical_combined | MultiCochrane, fine-tune from anti-parrot checkpoint, combined medical objective |

Implementation: src/t5_train/pipeline.py (MODES dict). Shared pieces: src/t5_train/data.py, metrics.py, presets.py.

Legacy filenames (train_t5_*.py in src) are thin wrappers that call the same pipelines.

## Data layout

Local training and eval CSV inputs resolve through src/paths.py:

- DATA_ROOT — project data directory; MultiCochrane English splits are MULTICOHRANE_EN_UNFILTERED and MULTICOHRANE_EN_FILTERED_R05 under data/multiCochrane_all/...

WikiLarge is streamed from Hugging Face (bogdancazan/wikilarge-text-simplification) unless you change t5_train.data; it is not read from DATA_ROOT.

## Checkpoints and outputs

All training checkpoints, Optuna sweeps, eval CSV append targets, and legacy script defaults write under OUTPUT_ROOT (the outputs directory at the project root). Constants such as MEDICAL_BASE_CHECKPOINT in src/t5_train/presets.py are absolute paths derived from OUTPUT_ROOT, so results do not depend on the shell current working directory.

CHECKPOINT_EVAL_CSV is outputs/checkpoint_evaluation_results.csv. If you had an older CSV at the repository root, move it into outputs/ to keep a single log.

The medical combined mode loads the anti-parrot base from outputs/final_t5_small_simplification_antiparrot_fast (see MEDICAL_BASE_CHECKPOINT in presets.py) after you have trained that stage.

## Other contents

- src/notebooks — Experiment notebooks (MultiCochrane, WikiLarge, checkpoint eval).
- src/eval_single_checkpoint.py — Single-checkpoint evaluation helper.
- src/legacy — Older or out-of-scope scripts (BART, Mistral, simpletransformers, extra notebooks). Import legacy/bootstrap_paths before paths when running from legacy.
- Report final assignment Transformers — HTML report for the assignment (figures under images/ as referenced by that report).

## Caches

Each training run calls _env_cache() in t5_train.pipeline, which sets:

- HF_DATASETS_CACHE to project/cache/huggingface/datasets
- TRANSFORMERS_CACHE to project/cache/huggingface/transformers

Override by editing that helper or the environment before launch if you need a different location.

rubenswarts.nl
