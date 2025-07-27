import pandas as pd
from pathlib import Path

def load_liar_dataset(raw_dir: Path) -> dict:
    """
    Load LIAR dataset from a given raw directory.

    Returns:
        dict: A dictionary with keys 'train', 'val', 'test'
    """
    return {
        "train": pd.read_csv(raw_dir / "train.tsv", sep="\t", header=None),
        "val": pd.read_csv(raw_dir / "valid.tsv", sep="\t", header=None),
        "test": pd.read_csv(raw_dir / "test.tsv", sep="\t", header=None),
    }