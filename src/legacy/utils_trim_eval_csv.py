"""Remove the last one or two rows from the checkpoint evaluation CSV if it has enough rows."""
import os
from pathlib import Path

import pandas as pd
import bootstrap_paths  # noqa: F401
from paths import CHECKPOINT_EVAL_CSV

CSV_FILE = str(CHECKPOINT_EVAL_CSV)
Path(CSV_FILE).parent.mkdir(parents=True, exist_ok=True)

if os.path.exists(CSV_FILE):
    try:
        df = pd.read_csv(CSV_FILE)
        if len(df) >= 2:
            print(f"Removing last 2 rows from {CSV_FILE}...")
            df_updated = df[:-2] # Select all rows except the last two
            df_updated.to_csv(CSV_FILE, index=False, encoding='utf-8')
            print("Rows removed successfully.")
        elif len(df) == 1:
             print(f"Removing last 1 row from {CSV_FILE}...")
             df_updated = df[:-1] # Select all rows except the last one
             df_updated.to_csv(CSV_FILE, index=False, encoding='utf-8')
             print("Row removed successfully.")
        else:
            print(f"{CSV_FILE} has less than 2 rows. No rows removed.")
    except Exception as e:
        print(f"Error processing {CSV_FILE}: {e}")
else:
    print(f"{CSV_FILE} not found. No rows removed.")
