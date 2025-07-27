# analysis/eda_liar.py

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import umap
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Output folder
EDA_OUTPUT_DIR = "analysis/eda_outputs"
os.makedirs(EDA_OUTPUT_DIR, exist_ok=True)


def load_data(path="data/processed/train.csv"):
    return pd.read_csv(path)


def plot_label_distribution(df):
    sns.countplot(data=df, x="label", order=df["label"].value_counts().index)
    plt.title("Label Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUTPUT_DIR, "label_distribution.png"))
    plt.close()


def plot_sentence_length_distribution(df):
    df["sentence_length"] = df["statement"].str.split().apply(len)
    sns.histplot(df["sentence_length"], kde=True, bins=30)
    plt.title("Distribution of Sentence Lengths")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUTPUT_DIR, "sentence_length_distribution.png"))
    plt.close()


def plot_sentiment_distribution(df):
    df["sentiment"] = df["statement"].apply(lambda x: TextBlob(x).sentiment.polarity)
    sns.histplot(df["sentiment"], kde=True, bins=30)
    plt.title("Sentiment Polarity Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUTPUT_DIR, "sentiment_distribution.png"))
    plt.close()


def generate_wordclouds(df):
    labels = df["label"].unique()
    for label in labels:
        text = " ".join(df[df["label"] == label]["statement"])
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        wc.to_file(os.path.join(EDA_OUTPUT_DIR, f"wordcloud_{label}.png"))


def plot_top_tfidf_words(df, top_n=15):
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
    X = vectorizer.fit_transform(df["statement"])
    labels = df["label"].unique()
    
    for label in labels:
        class_indices = df[df["label"] == label].index
        class_tfidf = X[class_indices].mean(axis=0).A1
        top_indices = np.argsort(class_tfidf)[-top_n:][::-1]
        top_words = [vectorizer.get_feature_names_out()[i] for i in top_indices]
        top_scores = class_tfidf[top_indices]

        plt.figure(figsize=(10, 4))
        sns.barplot(x=top_scores, y=top_words)
        plt.title(f"Top TF-IDF Words - {label}")
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_OUTPUT_DIR, f"top_tfidf_{label}.png"))
        plt.close()


def plot_umap_projection(df):
    import torch
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    def embed(texts, tokenizer, model):
        tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            model_output = model(**tokens)
        embeddings = model_output.last_hidden_state.mean(dim=1)
        return embeddings.numpy()

    embeddings = embed(df["statement"].tolist()[:1000], tokenizer, model)  # Subset for speed
    reducer = umap.UMAP()
    embedding_2d = reducer.fit_transform(embeddings)
    
    label_encoder = LabelEncoder()
    label_ids = label_encoder.fit_transform(df["label"].tolist()[:1000])

    plt.figure(figsize=(8, 6))
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=label_ids, cmap="tab10", s=10)
    plt.title("UMAP Projection of Sentence Embeddings")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUTPUT_DIR, "umap_projection.png"))
    plt.close()


def run_eda():
    df = load_data()
    print(f"Loaded {len(df)} records")

    plot_label_distribution(df)
    plot_sentence_length_distribution(df)
    plot_sentiment_distribution(df)
    generate_wordclouds(df)
    plot_top_tfidf_words(df)
    plot_umap_projection(df)


if __name__ == "__main__":
    run_eda()