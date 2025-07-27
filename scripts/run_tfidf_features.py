import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.features.vectorizers import extract_tfidf_features

if __name__ == "__main__":
    datasets = ["train", "test", "val"]

    for dataset in datasets:
        print(f"Extracting TF-IDF features for {dataset} set...")
        extract_tfidf_features(
            input_csv_path=f"data/processed/{dataset}.csv",
            output_dir="data/features/vectorizers_output/",
            dataset=dataset,
            max_features=5000
        )
        print(f"Completed {dataset} set processing...\n")