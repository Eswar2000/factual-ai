import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.features.embeddings import extract_embeddings

if __name__ == "__main__":
    datasets = ["train", "test", "val"]

    for dataset in datasets:
        print(f"Extracting embeddings for {dataset} set...")
        extract_embeddings(
            input_csv_path=f"data/processed/{dataset}.csv",
            output_dir="data/features/embeddings_output/",
            dataset=dataset
        )
        print(f"Completed {dataset} set processing...\n")