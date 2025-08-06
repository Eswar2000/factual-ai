import sys
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.features.basic_features import extract_basic_features

if __name__ == "__main__":
    modes = ["classic", "transformer"]
    datasets = ["train", "test", "val"]
    for mode in modes:
        for dataset in datasets:
            print(f"Extracting basic features for {dataset} dataset - {mode} mode...")
            extract_basic_features(
                input_csv_path=f"data/processed/{mode}/{dataset}.csv",
                output_dir="data/features/basic_features_output/",
                encoding_dir="data/features/encoders",
                dataset=dataset,
                mode=mode
            )
            print(f"Completed {dataset} dataset - {mode} mode processing...\n")