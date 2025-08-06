import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.utils.text_cleaning import preprocess_text_aggressive, preprocess_text_light

# Columns per LIAR dataset spec
COLUMNS = [
    "id", "label", "statement", "subject", "speaker", "speaker_job_title",
    "state_info", "party_affiliation", "barely_true_counts", "false_counts",
    "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", "context"
]

NUM_COLUMNS = [
    "barely_true_counts", "false_counts", "half_true_counts",
    "mostly_true_counts", "pants_on_fire_counts"
]

CAT_COLUMNS = ["subject", "speaker", "speaker_job_title", "state_info", "party_affiliation", "context"]

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
    df = df[["label", "statement", "subject", "speaker", "speaker_job_title",
    "state_info", "party_affiliation", "barely_true_counts", "false_counts",
    "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", "context"]]

    # Normalize labels
    df["label"] = df["label"].map(LABEL_MAP)

    # Drop rows where statement or label are missing
    df = df.dropna(subset=["label", "statement"])

    # Numerical column imputation with 0
    df[NUM_COLUMNS] = df[NUM_COLUMNS].fillna(0).astype(int)

    # Categorical column imputation with "unknown"
    df[CAT_COLUMNS] = df[CAT_COLUMNS].fillna("unknown")

    return df

def run_cleaning(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    df = preprocess_df(df)

    # Apply aggressive text preprocessing for Classic Models
    df_classic = df.copy()
    df_classic["statement"] = df_classic["statement"].apply(preprocess_text_aggressive)

    # Apply light text preprocessing for Transformers
    df_transformer = df.copy()
    df_transformer["statement"] = df_transformer["statement"].apply(preprocess_text_light)

    return df_classic, df_transformer