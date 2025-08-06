import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.features.vectorizers import extract_tfidf_features

if __name__ == "__main__":
    extract_tfidf_features(
    train_path="data/processed/classic/train.csv",
    val_path="data/processed/classic/val.csv",
    test_path="data/processed/classic/test.csv",
    output_dir="data/features/vectorizers_output/",
    max_features=1000000
    )