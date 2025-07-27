import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.features.final_features import save_labels, verify_alignment

if __name__ == "__main__":
    os.makedirs("data/features", exist_ok=True)

    for dataset in ["train", "val", "test"]:
        print(f"Saving labels for {dataset} set...")
        save_labels(
            input_csv_path=f"data/processed/{dataset}.csv",
            output_dir="data/features/labels_output/",
            dataset=dataset
        )
        print(f"Completed {dataset} set label saving...\n")
        print(f"Verifying alignment for {dataset} set...")
        verify_alignment(
            input_csv_path=f"data/processed/{dataset}.csv",
            output_dir="data/features/",
            dataset=dataset
        )
        print(f"Completed {dataset} set verification...\n")