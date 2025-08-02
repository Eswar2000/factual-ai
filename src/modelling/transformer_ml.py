import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

# --------------------------- Config ---------------------------
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
NUM_LABELS = 6
SEED = 42
set_seed(SEED)

# --------------------------- Load Data ---------------------------
def load_liar_dataset():
    def read_and_format(path):
        df = pd.read_csv(path)
        df = df[["label", "statement"]].dropna()
        return df

    train_df = read_and_format("data/processed/train.csv")
    val_df = read_and_format("data/processed/val.csv")
    test_df = read_and_format("data/processed/test.csv")

    return train_df, val_df, test_df

train_df, val_df, test_df = load_liar_dataset()

# --------------------------- Encode Labels ---------------------------
label_encoder = LabelEncoder()
train_df["label"] = label_encoder.fit_transform(train_df["label"])
val_df["label"] = label_encoder.transform(val_df["label"])
test_df["label"] = label_encoder.transform(test_df["label"])

# Save label classes for reference
os.makedirs("data/models", exist_ok=True)
np.save("data/models/label_classes.npy", label_encoder.classes_)

# --------------------------- Tokenization ---------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(example):
    return tokenizer(
        example["statement"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

# Convert to HuggingFace datasets
dataset_dict = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "validation": Dataset.from_pandas(val_df),
    "test": Dataset.from_pandas(test_df)
})

# Tokenize
tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)

# --------------------------- Model ---------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)

# --------------------------- Metrics ---------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    acc = accuracy_score(labels, predictions)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# --------------------------- Training Arguments ---------------------------
training_args = TrainingArguments(
    output_dir="data/models/transformer",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="data/models/transformer/logs",
    logging_steps=50,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

# --------------------------- Trainer ---------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)

# --------------------------- Train ---------------------------
trainer.train()

# --------------------------- Evaluate & Save ---------------------------
eval_result = trainer.evaluate(tokenized_datasets["test"])
print("📊 Test Metrics:", eval_result)

# Save model and tokenizer
model.save_pretrained("data/models/transformer/final_model")
tokenizer.save_pretrained("data/models/transformer/final_model")
