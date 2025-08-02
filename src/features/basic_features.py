import pandas as pd
import numpy as np
import string
import os
from nltk.corpus import stopwords
from textblob import TextBlob
import spacy
import textstat
nlp = spacy.load("en_core_web_sm")

STOPWORDS = set(stopwords.words('english'))

def extract_basic_features(input_csv_path: str, output_dir: str, dataset: str):
    # Load the processed dataset
    df = pd.read_csv(input_csv_path)

    if "statement" not in df.columns:
        raise ValueError("Missing 'statement' column in the dataset.")

    df["statement"].fillna("", inplace=True)

    features = []
    
    for text in df["statement"].fillna("").astype(str):
        doc = nlp(text)
        
        char_count = len(text)
        word_count = len(text.split())
        avg_word_len = sum(len(word) for word in text.split()) / word_count if word_count else 0
        stopword_ratio = sum(1 for word in doc if word.is_stop) / len(doc) if len(doc) > 0 else 0
        punctuation_count = sum(1 for c in text if c in string.punctuation)
        digit_count = sum(1 for c in text if c.isdigit())
        uppercase_ratio = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
        exclamation_count = text.count("!")
        noun_ratio = sum(1 for token in doc if token.pos_ == "NOUN") / len(doc) if len(doc) > 0 else 0
        verb_ratio = sum(1 for token in doc if token.pos_ == "VERB") / len(doc) if len(doc) > 0 else 0
        named_entity_count = len(list(doc.ents))
        readability_score = textstat.flesch_reading_ease(text)
        sentiment = TextBlob(text).sentiment
        
        features.append({
            "char_count": char_count,
            "word_count": word_count,
            "avg_word_length": avg_word_len,
            "stopword_ratio": stopword_ratio,
            "punctuation_count": punctuation_count,
            "digit_count": digit_count,
            "uppercase_ratio": uppercase_ratio,
            "exclamation_count": exclamation_count,
            "noun_ratio": noun_ratio,
            "verb_ratio": verb_ratio,
            "named_entity_count": named_entity_count,
            "readability_score": readability_score,
            "sentiment_polarity": sentiment.polarity,
            "sentiment_subjectivity": sentiment.subjectivity
        })

    features_df = pd.DataFrame(features)
    df_out = pd.concat([df, features_df], axis=1)
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save basic features
    basic_path = os.path.join(output_dir, f"{dataset}_basic_features.csv")
    df_out.to_csv(basic_path, index=False)

    print(f"Basic features saved to: {basic_path}")
    print(f"Basic features shape: {df_out.shape}")