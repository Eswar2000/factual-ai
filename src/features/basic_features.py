import pandas as pd
import numpy as np
import string
import os
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob
import spacy
import textstat
from pathlib import Path
import joblib
from tqdm import tqdm
nlp = spacy.load("en_core_web_sm")

STOPWORDS = set(stopwords.words('english'))

def extract_basic_features(input_csv_path: str, output_dir: str, encoding_dir: str, dataset: str, mode: str):
    # Load the processed dataset
    df = pd.read_csv(input_csv_path)

    if "statement" not in df.columns:
        raise ValueError("Missing 'statement' column in the dataset.")

    df["statement"].fillna("", inplace=True)

    # Handle encoders
    le_subject = LabelEncoder()
    le_party = LabelEncoder()

    # Limit speaker cardinality
    top_speakers = df["speaker"].value_counts().nlargest(20).index
    df["speaker_group"] = df["speaker"].apply(lambda x: x if x in top_speakers else "Other")
    le_speaker = LabelEncoder()

    df["subject_encoded"] = le_subject.fit_transform(df["subject"].fillna("Unknown"))
    df["party_encoded"] = le_party.fit_transform(df["party_affiliation"].fillna("Unknown"))
    df["speaker_encoded"] = le_speaker.fit_transform(df["speaker_group"].fillna("Other"))

    # Save encoders
    Path(encoding_dir+"/"+mode).mkdir(parents=True, exist_ok=True)
    joblib.dump(le_subject, f"{encoding_dir}/{mode}/{dataset}_subject_encoder.joblib")
    joblib.dump(le_party, f"{encoding_dir}/{mode}/{dataset}_party_encoder.joblib")
    joblib.dump(le_speaker, f"{encoding_dir}/{mode}/{dataset}_speaker_encoder.joblib")

    features = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        statement = str(row["statement"])
        context = str(row["context"])

        doc_stmt = nlp(statement)
        doc_ctx = nlp(context)

        sentiment = TextBlob(statement).sentiment
        ctx_sentiment = TextBlob(context).sentiment

        features.append({
            # Statement-based features
            "char_count": len(statement),
            "word_count": len(statement.split()),
            "avg_word_len": np.mean([len(w) for w in statement.split()]) if statement.split() else 0,
            "stopword_ratio": len([t for t in doc_stmt if t.is_stop]) / (len(doc_stmt) + 1e-5),
            "punctuation_ratio": len([c for c in statement if c in string.punctuation]) / (len(statement) + 1e-5),
            "uppercase_ratio": sum(1 for c in statement if c.isupper()) / (len(statement) + 1e-5),
            "digit_ratio": sum(1 for c in statement if c.isdigit()) / (len(statement) + 1e-5),
            "exclamation_ratio": statement.count("!") / (len(statement) + 1e-5),
            "noun_ratio": len([t for t in doc_stmt if t.pos_ == "NOUN"]) / (len(doc_stmt) + 1e-5),
            "verb_ratio": len([t for t in doc_stmt if t.pos_ == "VERB"]) / (len(doc_stmt) + 1e-5),
            "num_named_entities": len(doc_stmt.ents),
            "readability": textstat.flesch_reading_ease(statement),
            "sentiment_polarity": sentiment.polarity,
            "sentiment_subjectivity": sentiment.subjectivity,

            # Context-based features
            "ctx_char_count": len(context),
            "ctx_word_count": len(context.split()),
            "ctx_avg_word_len": np.mean([len(w) for w in context.split()]) if context.split() else 0,
            "ctx_stopword_ratio": len([t for t in doc_ctx if t.is_stop]) / (len(doc_ctx) + 1e-5),
            "ctx_punctuation_ratio": len([c for c in context if c in string.punctuation]) / (len(context) + 1e-5),
            "ctx_readability": textstat.flesch_reading_ease(context),
            "ctx_sentiment_polarity": ctx_sentiment.polarity,
            "ctx_sentiment_subjectivity": ctx_sentiment.subjectivity,

            # Encoded categorical fields
            "subject_encoded": row["subject_encoded"],
            "party_encoded": row["party_encoded"],
            "speaker_encoded": row["speaker_encoded"]
        })

    features_df = pd.DataFrame(features)
    df_out = pd.concat([df, features_df], axis=1)
    # Ensure output directory exists
    os.makedirs(output_dir+"/"+mode, exist_ok=True)

    # Save basic features
    basic_path = os.path.join(output_dir, mode, f"{dataset}_basic_features.csv")
    df_out.to_csv(basic_path, index=False)

    print(f"Basic features saved to: {basic_path}")
    print(f"Basic features shape: {df_out.shape}")