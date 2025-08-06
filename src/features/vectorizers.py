import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz, hstack, csr_matrix
import numpy as np

def extract_tfidf_features(train_path, val_path, test_path, output_dir, max_features=5000):
    os.makedirs(output_dir, exist_ok=True)
    encoder_dir = "data/features/encoders/classic"

    # Load datasets
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Helper: Combine relevant text fields
    def combine_text(row):
        return f"{row['statement']} [CTX] {row['context']} [SUBJ] {row['subject']}"

    train_df["combined"] = train_df.apply(combine_text, axis=1)
    val_df["combined"] = val_df.apply(combine_text, axis=1)
    test_df["combined"] = test_df.apply(combine_text, axis=1)

    # Fit vectorizer on training only
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(train_df["combined"])
    X_val_tfidf = vectorizer.transform(val_df["combined"])
    X_test_tfidf = vectorizer.transform(test_df["combined"])

    categorical_fields = ["party_affiliation", "speaker", "subject"]
    encoder_map = {
        "party_affiliation": "party",
        "speaker": "speaker",
        "subject": "subject"
    }
    X_train_cat, X_val_cat, X_test_cat = [], [], []

    for field in categorical_fields:
        le_train = joblib.load(encoder_dir + "/train_" + encoder_map[field] + "_encoder.joblib")
        le_val = joblib.load(encoder_dir + "/val_" + encoder_map[field] + "_encoder.joblib")
        le_test = joblib.load(encoder_dir + "/test_" + encoder_map[field] + "_encoder.joblib")

        # Ensure 'Unknown' is in the encoder classes
        if "Unknown" not in le_train.classes_:
            le_train.classes_ = np.append(le_train.classes_, "Unknown")

        # Ensure 'Unknown' is in the encoder classes
        if "Unknown" not in le_val.classes_:
            le_val.classes_ = np.append(le_val.classes_, "Unknown")

        # Ensure 'Unknown' is in the encoder classes
        if "Unknown" not in le_test.classes_:
            le_test.classes_ = np.append(le_test.classes_, "Unknown")

        def safe_encode(values, encoder):
            return [v if v in encoder.classes_ else "Unknown" for v in values]

        if field == "speaker":
            # Group rare speakers as 'Other'
            top_speakers = train_df["speaker"].value_counts().nlargest(20).index
            train_df["speaker_group"] = train_df["speaker"].apply(lambda x: x if x in top_speakers else "Other")
            val_df["speaker_group"] = val_df["speaker"].apply(lambda x: x if x in top_speakers else "Other")
            test_df["speaker_group"] = test_df["speaker"].apply(lambda x: x if x in top_speakers else "Other")

            train_vals = train_df["speaker_group"].fillna("Other")
            val_vals = val_df["speaker_group"].fillna("Other")
            test_vals = test_df["speaker_group"].fillna("Other")
        else:
            train_vals = train_df[field].fillna("unknown")
            val_vals = val_df[field].fillna("unknown")
            test_vals = test_df[field].fillna("unknown")

        train_enc = le_train.transform(safe_encode(train_vals, le_train))
        val_enc = le_train.transform(safe_encode(val_vals, le_train))
        test_enc = le_train.transform(safe_encode(test_vals, le_train))

        # train_enc = le_train.transform(train_df[field])
        # val_enc = le_val.transform(val_df[field])
        # test_enc = le_test.transform(test_df[field])

        X_train_cat.append(train_enc.reshape(-1, 1))
        X_val_cat.append(val_enc.reshape(-1, 1))
        X_test_cat.append(test_enc.reshape(-1, 1))

    # Stack all features
    def sparse_stack(tfidf_matrix, cat_arrays):
        cat_sparse = csr_matrix(np.hstack(cat_arrays))
        return hstack([tfidf_matrix, cat_sparse])

    X_train = sparse_stack(X_train_tfidf, X_train_cat)
    X_val = sparse_stack(X_val_tfidf, X_val_cat)
    X_test = sparse_stack(X_test_tfidf, X_test_cat)

    # Save vectorizer
    joblib.dump(vectorizer, os.path.join(output_dir, "tfidf_vectorizer.joblib"))

    # Save features
    save_npz(os.path.join(output_dir, "tfidf_train.npz"), X_train)
    save_npz(os.path.join(output_dir, "tfidf_val.npz"), X_val)
    save_npz(os.path.join(output_dir, "tfidf_test.npz"), X_test)

    print(f"✅ TF-IDF features saved in: {output_dir}")
    print(f"Shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
