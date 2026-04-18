"""Named defaults for each training mode (aligned with former standalone scripts).

Constants group into four areas: WikiLarge small model with plain SARI, WikiLarge anti-parrot
penalized SARI, MultiCochrane large model with SARI, and medical combined fine-tuning from an
anti-parrot checkpoint. All checkpoint and sweep directories are under paths.OUTPUT_ROOT; local
MultiCochrane CSV inputs use paths.DATA_ROOT via pipeline (see paths.MULTICOHRANE_*).
"""
from __future__ import annotations

from paths import OUTPUT_ROOT

# WikiLarge plus t5-small plus plain SARI
WIKI_SMALL_PREFIX = "simplify: "
WIKI_SMALL_MODEL = "google-t5/t5-small"
WIKI_SMALL_OPTUNA_TRIALS = 20
WIKI_SMALL_OUT_HPO = str(OUTPUT_ROOT / "optuna_sweep_checkpoints_t5_small_simplification_general_v1")
WIKI_SMALL_OUT_FINAL = str(OUTPUT_ROOT / "final_t5-small-sari_general_simplification_v1")

# WikiLarge plus anti-parrot instruction prefix
WIKI_ANTIPARROT_PREFIX = "Convert this complex text to simple language: "
WIKI_ANTIPARROT_OUT_HPO = str(OUTPUT_ROOT / "optuna_sweep_t5_small_simplification_antiparrot_fast")
WIKI_ANTIPARROT_OUT_FINAL = str(OUTPUT_ROOT / "final_t5_small_simplification_antiparrot_fast")

# MultiCochrane English plus t5-large plus SARI
LARGE_MODEL = "google-t5/t5-large"
LARGE_OUT_HPO = str(OUTPUT_ROOT / "optuna_sweep_checkpoints_t5_large")
LARGE_OUT_FINAL = str(OUTPUT_ROOT / "final_t5_sari_best_model_epoch_25")
LARGE_PREFIX = "simplify: "

# MultiCochrane continuation from anti-parrot checkpoint with combined score objective
MEDICAL_PREFIX = "simplify medical text without copying: "
MEDICAL_BASE_CHECKPOINT = str(OUTPUT_ROOT / "final_t5_small_simplification_antiparrot_fast")
MEDICAL_HPO_OUT = str(OUTPUT_ROOT / "hpo_medical_finetune_t5_small")
MEDICAL_FINAL_OUT = str(OUTPUT_ROOT / "final_medical_finetune_t5_small")
