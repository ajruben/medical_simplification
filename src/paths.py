"""Project root and shared filesystem paths (works regardless of current working directory).

PROJECT_ROOT is the repository root (parent of the src directory). DATA_ROOT is the canonical
input tree for local corpora (MultiCochrane CSVs and similar). OUTPUT_ROOT is the canonical tree
for checkpoints, training runs, and evaluation logs. MULTICOHRANE paths point at English CSV
splits under DATA_ROOT. CHECKPOINT_EVAL_CSV lives under OUTPUT_ROOT.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data"

OUTPUT_ROOT = PROJECT_ROOT / "outputs"

MULTICOHRANE_EN_UNFILTERED = (
    DATA_ROOT / "multiCochrane_all" / "unfiltered (r=0)" / "en"
)

MULTICOHRANE_EN_FILTERED_R05 = (
    DATA_ROOT / "multiCochrane_all" / "filtered (r=0.5)" / "en"
)

CHECKPOINT_EVAL_CSV = OUTPUT_ROOT / "checkpoint_evaluation_results.csv"
