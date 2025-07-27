import os
import pandas as pd
import numpy as np
from scipy import sparse
import joblib

def save_labels(input_csv_path: str, output_dir: str, dataset: str):
    df = pd.read_csv(input_csv_path)
    labels = df["label"].values

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save to file
    label_path = os.path.join(output_dir, f"{dataset}_labels.npy")
    np.save(label_path, labels)

    print(f"Labels saved to: {label_path}")
    print(f"Labels shape: {labels.shape}")

def verify_alignment(input_csv_path: str, output_dir: str, dataset: str):
    df = pd.read_csv(input_csv_path)
    basic = pd.read_csv(f"{output_dir}basic_features_output/{dataset}_basic_features.csv")
    tfidf = sparse.load_npz(f"{output_dir}vectorizers_output/{dataset}_tfidf_features.npz")
    embeddings = np.load(f"{output_dir}embeddings_output/{dataset}_embeddings.npy")
    labels = np.load(f"{output_dir}labels_output/{dataset}_labels.npy")

    print(f"\n[VERIFY - {dataset.upper()}]")
    print(f"Processed:     {len(df)}")
    print(f"Basic:        {len(basic)}")
    print(f"TF-IDF:        {tfidf.shape[0]}")
    print(f"Embeddings:    {embeddings.shape[0]}")
    print(f"Labels:        {len(labels)}")