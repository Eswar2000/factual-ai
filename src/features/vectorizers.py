import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

def extract_tfidf_features(input_csv_path: str, output_dir: str, dataset: str, max_features: int = 5000):
    # Load the processed dataset
    df = pd.read_csv(input_csv_path)
    
    if "statement" not in df.columns:
        raise ValueError("Missing 'statement' column in the dataset.")

    texts = df["statement"].fillna("").astype(str).tolist()

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),  # Unigrams and bigrams
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(texts)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save TF-IDF features (sparse matrix)
    tfidf_path = os.path.join(output_dir, f"{dataset}_tfidf_features.npz")
    save_npz(tfidf_path, tfidf_matrix)

    # Save vectorizer
    vectorizer_path = os.path.join(output_dir, f"{dataset}_tfidf_vectorizer.joblib")
    joblib.dump(vectorizer, vectorizer_path)

    print(f"TF-IDF features saved to: {tfidf_path}")
    print(f"Vectorizer saved to: {vectorizer_path}")
    print(f"TF-IDF shape: {tfidf_matrix.shape}")
