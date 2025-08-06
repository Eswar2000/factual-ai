import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data import load_liar_dataset, save_processed_data
from src.data.clean import run_cleaning

def main():
    raw_dir = Path("data/raw")
    classic_out_dir = Path("data/processed/classic")
    transformer_out_dir = Path("data/processed/transformer")

    print("[INFO] Loading raw data...")
    raw_data = load_liar_dataset(raw_dir)

    for split, df in raw_data.items():
        print(f"[INFO] Loaded {split} split: {len(df)} records")

    print("[INFO] Preprocessing...")

    # Prepare dictionaries to store processed outputs
    classic_data = {}
    transformer_data = {}
    for split, df in raw_data.items():
        df_classic, df_transformer = run_cleaning(df)
        classic_data[split] = df_classic
        transformer_data[split] = df_transformer

    # Save processed outputs
    print("[INFO] Saving classic-cleaned data...")
    save_processed_data(classic_data, classic_out_dir)

    print("[INFO] Saving transformer-cleaned data...")
    save_processed_data(transformer_data, transformer_out_dir)

    print("[✅] Preprocessing complete. Output saved to:")
    print(f" - {classic_out_dir}/")
    print(f" - {transformer_out_dir}/")

if __name__ == "__main__":
    main()