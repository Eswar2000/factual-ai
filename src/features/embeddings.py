import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def extract_embeddings(input_csv_path: str, output_dir: str, dataset: str, model_name: str = "all-MiniLM-L6-v2"):
    # Load the processed dataset
    df = pd.read_csv(input_csv_path)

    if "statement" not in df.columns:
        raise ValueError("Missing 'statement' column in the dataset.")
    
    texts = df["statement"].fillna("").astype(str).tolist()

    # Load model
    model = SentenceTransformer(model_name)

    # Encode statements
    embeddings = model.encode(texts, show_progress_bar=True)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save to file
    embedding_path = os.path.join(output_dir, f"{dataset}_embeddings.npy")
    np.save(embedding_path, embeddings)

    print(f"Embeddings saved to: {embedding_path}")
    print(f"Embeddings shape: {embeddings.shape}")