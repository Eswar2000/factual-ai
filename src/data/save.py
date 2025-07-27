from pathlib import Path

def save_processed_data(df_dict: dict, out_dir: Path):
    """
    Save processed datasets to the output directory.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, df in df_dict.items():
        df.to_csv(out_dir / f"{split}.csv", index=False)