import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

def extract_tfidf_features(train_path, val_path, test_path, output_dir, max_features=5000):
    os.makedirs(output_dir, exist_ok=True)

    # Load datasets
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    train_texts = train_df["statement"].fillna("").astype(str).tolist()
    val_texts = val_df["statement"].fillna("").astype(str).tolist()
    test_texts = test_df["statement"].fillna("").astype(str).tolist()

    # Fit vectorizer on training only
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), stop_words="english")
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)

    # Save vectorizer
    joblib.dump(vectorizer, os.path.join(output_dir, "tfidf_vectorizer.joblib"))

    # Save features
    save_npz(os.path.join(output_dir, "tfidf_train.npz"), X_train)
    save_npz(os.path.join(output_dir, "tfidf_val.npz"), X_val)
    save_npz(os.path.join(output_dir, "tfidf_test.npz"), X_test)

    print(f"✅ TF-IDF features saved in: {output_dir}")
    print(f"Shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
