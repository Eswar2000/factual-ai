import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.utils.text_cleaning import preprocess_text

# Columns per LIAR dataset spec
COLUMNS = [
    "id", "label", "statement", "subject", "speaker", "speaker_job_title",
    "state_info", "party_affiliation", "barely_true_counts", "false_counts",
    "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", "context"
]

LABEL_MAP = {
    "true": "true",
    "mostly-true": "true",
    "half-true": "true",
    "false": "false",
    "barely-true": "false",
    "pants-fire": "false"
}

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = COLUMNS
    df = df[["label", "statement", "subject", "speaker", "party_affiliation", "context"]]

    # Normalize labels
    df["label"] = df["label"].map(LABEL_MAP)

    # Drop rows with missing data
    df = df.dropna()

    # Optional: remove very short statements
    df = df[df["statement"].str.len() > 10]

    # Clean statement text
    df["statement"] = df["statement"].apply(preprocess_text)

    return df