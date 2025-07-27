import sys
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.features.basic_features import extract_basic_features

if __name__ == "__main__":
    datasets = ["train", "test", "val"]
    for dataset in datasets:
        print(f"Extracting basic features for {dataset} set...")
        extract_basic_features(
            input_csv_path=f"data/processed/{dataset}.csv",
            output_dir="data/features/basic_features_output/",
            dataset=dataset
        )
        print(f"Completed {dataset} set processing...\n")