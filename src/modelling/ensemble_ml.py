import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from scipy.sparse import load_npz, hstack, csr_matrix

# -------------------- Paths --------------------
FEATURE_DIR = "data/features"
OUTPUT_DIR = "data/models/ensemble"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_split(split):
    tfidf = load_npz(f"{FEATURE_DIR}/vectorizers_output/{split}_tfidf_features.npz")
    embed = np.load(f"{FEATURE_DIR}/embeddings_output/{split}_embeddings.npy")
    labels = np.load(f"{FEATURE_DIR}/labels_output/{split}_labels.npy")
    return hstack([tfidf, csr_matrix(embed)]), labels

print("🔹 Loading features...")
X_train, y_train = load_split("train")
X_val, y_val = load_split("val")
X_test, y_test = load_split("test")

# -------------------- Logistic Regression --------------------
print("🔹 Training Logistic Regression...")
lr_clf = LogisticRegression(max_iter=1000)
lr_clf.fit(X_train, y_train)
y_pred_lr = lr_clf.predict(X_test)
joblib.dump(lr_clf, os.path.join(OUTPUT_DIR, "meta_logreg_model.joblib"))

# -------------------- XGBoost Classifier --------------------
print("🔹 Training XGBoost...")
xgb_clf = XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric="mlogloss")
xgb_clf.fit(X_train.toarray(), y_train)
y_pred_xgb = xgb_clf.predict(X_test.toarray())
joblib.dump(xgb_clf, os.path.join(OUTPUT_DIR, "meta_xgboost_model.joblib"))

# -------------------- Evaluation & Visualization --------------------
def evaluate_and_save(y_true, y_pred, model_name):
    report = classification_report(y_true, y_pred, digits=3)
    conf_mat = confusion_matrix(y_true, y_pred)

    # Save report
    with open(os.path.join(OUTPUT_DIR, f"{model_name}_report.txt"), "w") as f:
        f.write(report)

    # Save matrix data
    np.save(os.path.join(OUTPUT_DIR, f"{model_name}_confusion_matrix.npy"), conf_mat)

    # Plot matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_confusion_matrix.png"))
    plt.close()

    print(f"✅ Saved evaluation for: {model_name}")

# Evaluate both models
evaluate_and_save(y_test, y_pred_lr, "logreg")
evaluate_and_save(y_test, y_pred_xgb, "xgboost")