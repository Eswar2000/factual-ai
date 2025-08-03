import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib
from scipy.sparse import load_npz


# Paths
FEATURE_DIR = "data/features"
MODEL_DIR = "data/models"

# Load TF-IDF features
X_train = load_npz(os.path.join(FEATURE_DIR, "vectorizers_output", "train_tfidf_features.npz"))
X_val = load_npz(os.path.join(FEATURE_DIR, "vectorizers_output", "val_tfidf_features.npz"))
X_test = load_npz(os.path.join(FEATURE_DIR, "vectorizers_output", "test_tfidf_features.npz"))

# Load labels
y_train = np.load(os.path.join(FEATURE_DIR, "labels_output", "train_labels.npy"))
y_val = np.load(os.path.join(FEATURE_DIR, "labels_output", "val_labels.npy"))
y_test = np.load(os.path.join(FEATURE_DIR, "labels_output", "test_labels.npy"))


def evaluate(model, X, y, name="Set"):
    y_pred = model.predict(X)
    print(f"\n📊 Evaluation on {name}:")
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))
    print("\nClassification Report:")
    print(classification_report(y, y_pred))


def train_logistic_regression():
    print("\n🔧 Training: Logistic Regression")
    param_grid = {
        "C": [0.01, 0.1, 1, 10]
    }
    grid = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), param_grid, cv=3)
    grid.fit(X_train, y_train)

    model = grid.best_estimator_

    # Evaluate
    evaluate(model, X_val, y_val, "Validation")
    evaluate(model, X_test, y_test, "Test")

    # Save model
    joblib.dump(model, os.path.join(MODEL_DIR, "vectorizer", "logreg_tfidf.joblib"))


def train_random_forest():
    print("\n🔧 Training: Random Forest")
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    }

    rf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    model = grid.best_estimator_

    # Evaluate
    evaluate(model, X_val, y_val, "Validation")
    evaluate(model, X_test, y_test, "Test")

    # Save model
    joblib.dump(model, os.path.join(MODEL_DIR, "vectorizer", "rf_tfidf.joblib"))


def train_svm():
    print("\n🔧 Training: Support Vector Machine")
    param_grid = {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"]
    }

    svm = SVC(probability=True)
    grid = GridSearchCV(svm, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    model = grid.best_estimator_
    # Evaluate
    evaluate(model, X_val, y_val, "Validation")
    evaluate(model, X_test, y_test, "Test")

    # Save model
    joblib.dump(model, os.path.join(MODEL_DIR, "vectorizer", "svm_tfidf.joblib"))


if __name__ == "__main__":
    train_logistic_regression()
    train_random_forest()
    train_svm()