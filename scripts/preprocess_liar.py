import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data import load_liar_dataset, preprocess_df, save_processed_data

def main():
    raw_dir = Path("data/raw")
    out_dir = Path("data/processed")

    print("[INFO] Loading raw data...")
    raw_data = load_liar_dataset(raw_dir)

    for split, df in raw_data.items():
        print(f"[INFO] Loaded {split} split: {len(df)} records")

    print("[INFO] Preprocessing...")
    processed = {}
    for split, df in raw_data.items():
        processed_df = preprocess_df(df)
        print(f"[INFO] Processed {split} split: {len(processed_df)} records after cleaning")
        processed[split] = processed_df

    print("[INFO] Saving processed data...")
    save_processed_data(processed, out_dir)

    print("[✅] Preprocessing complete. Files saved to data/processed/")

if __name__ == "__main__":
    main()