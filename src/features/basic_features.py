import pandas as pd
import numpy as np
import string
import re
import os
from nltk.corpus import stopwords
from textblob import TextBlob

STOPWORDS = set(stopwords.words('english'))

def extract_basic_features(input_csv_path: str, output_dir: str, dataset: str):
    # Load the processed dataset
    df = pd.read_csv(input_csv_path)

    if "statement" not in df.columns:
        raise ValueError("Missing 'statement' column in the dataset.")

    df["statement"].fillna("", inplace=True)

    def avg_word_length(text):
        words = text.split()
        return np.mean([len(word) for word in words]) if words else 0

    def stopword_ratio(text):
        words = text.lower().split()
        if not words:
            return 0
        stopword_count = sum(1 for word in words if word in STOPWORDS)
        return stopword_count / len(words)

    def punctuation_count(text):
        return sum(1 for c in text if c in string.punctuation)

    def digit_count(text):
        return sum(c.isdigit() for c in text)

    def sentiment_polarity(text):
        return TextBlob(text).sentiment.polarity

    def sentiment_subjectivity(text):
        return TextBlob(text).sentiment.subjectivity

    features = pd.DataFrame()
    features["char_count"] = df["statement"].apply(len)
    features["word_count"] = df["statement"].apply(lambda x: len(x.split()))
    features["avg_word_length"] = df["statement"].apply(avg_word_length)
    features["stopword_ratio"] = df["statement"].apply(stopword_ratio)
    features["punctuation_count"] = df["statement"].apply(punctuation_count)
    features["digit_count"] = df["statement"].apply(digit_count)
    features["sentiment_polarity"] = df["statement"].apply(sentiment_polarity)
    features["sentiment_subjectivity"] = df["statement"].apply(sentiment_subjectivity)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save basic features
    basic_path = os.path.join(output_dir, f"{dataset}_basic_features.csv")
    features.to_csv(basic_path, index=False)

    print(f"Basic features saved to: {basic_path}")
    print(f"Basic features shape: {features.shape}")