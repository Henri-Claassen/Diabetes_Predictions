"""
train_models.py
---------------
Trains and evaluates all four models required for the Diabetes Risk
Decision Support System:

    Supervised (predict diabetes_stage)
        1. Decision Tree
        2. Random Forest
        3. XGBoost

    Unsupervised (lifestyle segmentation)
        4. K-Means (k = 3)

Pipeline
    * Load the already-processed data produced by preprocess_data.py
      - X_train.csv / X_test.csv / y_train.csv / y_test.csv   (tree models)
      - X_train_scaled.csv / X_test_scaled.csv                (K-Means)
    * Tune each classifier with 5-fold stratified CV on the TRAINING set
      (GridSearchCV, scoring = weighted F1 to handle class imbalance).
    * Evaluate the final tuned model ONCE on the held-out test set.
    * Fit K-Means on the scaled training data and assign test clusters.
    * Save every trained model + a metrics report to artifacts/.

Run:
    python src/train_models.py
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths  (mirrors the convention used in preprocess_data.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
CV_FOLDS = 5
SCORING = "f1_weighted"   # handles class imbalance better than accuracy


# ---------------------------------------------------------------------------
# 1. Load processed data
# ---------------------------------------------------------------------------
def load_processed_data():
    """Load the CSVs produced by preprocess_data.py."""
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")

    X_train_scaled = pd.read_csv(DATA_DIR / "X_train_scaled.csv")
    X_test_scaled = pd.read_csv(DATA_DIR / "X_test_scaled.csv")

    # Load label encoder so we can print human-readable class names
    with open(ARTIFACTS_DIR / "label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"Classes: {dict(enumerate(label_encoder.classes_))}")
    print("Training class distribution:")
    print(y_train.value_counts().sort_index().to_string())
    print()

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, label_encoder


# ---------------------------------------------------------------------------
# 2. Evaluation helper
# ---------------------------------------------------------------------------
def evaluate_classifier(name, model, X_test, y_test, label_encoder):
    """Compute metrics on the test set and return a dict."""
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n===== {name} — Test set results =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )
    print("Confusion matrix (rows = true, cols = predicted):")
    print(pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_))

    return {
        "accuracy": float(acc),
        "precision_weighted": float(prec),
        "recall_weighted": float(rec),
        "f1_weighted": float(f1),
        "confusion_matrix": cm.tolist(),
    }


def tune_and_fit(name, estimator, param_grid, X_train, y_train):
    """Run GridSearchCV with stratified CV on the training set."""
    print(f"\n>>> Tuning {name} with {CV_FOLDS}-fold CV...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=SCORING,
        cv=cv,
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    print(f"Best params : {grid.best_params_}")
    print(f"Best CV F1  : {grid.best_score_:.4f}")
    return grid.best_estimator_, grid.best_params_, float(grid.best_score_)


# ---------------------------------------------------------------------------
# 3. Decision Tree
# ---------------------------------------------------------------------------
def train_decision_tree(X_train, y_train):
    param_grid = {
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 10, 20],
        "min_samples_leaf": [1, 5, 10],
        "criterion": ["gini", "entropy"],
    }
    base = DecisionTreeClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    return tune_and_fit("Decision Tree", base, param_grid, X_train, y_train)


# ---------------------------------------------------------------------------
# 4. Random Forest
# ---------------------------------------------------------------------------
def train_random_forest(X_train, y_train):
    param_grid = {
        "n_estimators": [100, 150],
        "max_depth": [10, 15],
        "min_samples_split": [10, 20],
        "max_features": ["sqrt", "log2"],
    }
    base = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return tune_and_fit("Random Forest", base, param_grid, X_train, y_train)


# ---------------------------------------------------------------------------
# 5. XGBoost
# ---------------------------------------------------------------------------
def train_xgboost(X_train, y_train):
    # XGBoost has no class_weight param; compute sample weights manually
    class_counts = y_train.value_counts().sort_index()
    total = len(y_train)
    n_classes = len(class_counts)
    class_weights = {cls: total / (n_classes * cnt) for cls, cnt in class_counts.items()}
    sample_weights = y_train.map(class_weights).values

    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
    }
    base = XGBClassifier(
        objective="multi:softprob",
        num_class=len(class_counts),
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print(f"\n>>> Tuning XGBoost with {CV_FOLDS}-fold CV (with class-balanced sample weights)...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        scoring=SCORING,
        cv=cv,
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train, sample_weight=sample_weights)
    print(f"Best params : {grid.best_params_}")
    print(f"Best CV F1  : {grid.best_score_:.4f}")
    return grid.best_estimator_, grid.best_params_, float(grid.best_score_)


# ---------------------------------------------------------------------------
# 6. K-Means segmentation
# ---------------------------------------------------------------------------
def train_kmeans(X_train_scaled, X_test_scaled, k=3):
    """Fit K-Means on scaled training data, score on scaled test data."""
    print(f"\n>>> Fitting K-Means with k={k} on scaled training data...")
    kmeans = KMeans(
        n_clusters=k,
        n_init=20,
        random_state=RANDOM_STATE,
    )
    train_labels = kmeans.fit_predict(X_train_scaled)
    test_labels = kmeans.predict(X_test_scaled)

    train_sil = silhouette_score(X_train_scaled, train_labels)
    test_sil = silhouette_score(X_test_scaled, test_labels)

    train_counts = pd.Series(train_labels).value_counts().sort_index()
    print(f"Silhouette (train): {train_sil:.4f}")
    print(f"Silhouette (test) : {test_sil:.4f}")
    print("Training cluster sizes:")
    print(train_counts.to_string())

    return kmeans, {
        "k": k,
        "silhouette_train": float(train_sil),
        "silhouette_test": float(test_sil),
        "train_cluster_sizes": train_counts.to_dict(),
        "inertia": float(kmeans.inertia_),
    }


# ---------------------------------------------------------------------------
# 7. Save artifacts
# ---------------------------------------------------------------------------
def save_model(model, filename):
    path = ARTIFACTS_DIR / filename
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved -> {path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------
def main():
    (
        X_train, X_test,
        y_train, y_test,
        X_train_scaled, X_test_scaled,
        label_encoder,
    ) = load_processed_data()

    results = {}

    # --- Decision Tree -----------------------------------------------------
    dt_model, dt_params, dt_cv = train_decision_tree(X_train, y_train)
    dt_metrics = evaluate_classifier("Decision Tree", dt_model, X_test, y_test, label_encoder)
    save_model(dt_model, "decision_tree.pkl")
    results["decision_tree"] = {"best_params": dt_params, "cv_f1_weighted": dt_cv, **dt_metrics}

    # --- Random Forest -----------------------------------------------------
    rf_model, rf_params, rf_cv = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_classifier("Random Forest", rf_model, X_test, y_test, label_encoder)
    save_model(rf_model, "random_forest.pkl")
    results["random_forest"] = {"best_params": rf_params, "cv_f1_weighted": rf_cv, **rf_metrics}

    # --- XGBoost -----------------------------------------------------------
    xgb_model, xgb_params, xgb_cv = train_xgboost(X_train, y_train)
    xgb_metrics = evaluate_classifier("XGBoost", xgb_model, X_test, y_test, label_encoder)
    save_model(xgb_model, "xgboost.pkl")
    results["xgboost"] = {"best_params": xgb_params, "cv_f1_weighted": xgb_cv, **xgb_metrics}

    # --- K-Means -----------------------------------------------------------
    kmeans_model, kmeans_metrics = train_kmeans(X_train_scaled, X_test_scaled, k=3)
    save_model(kmeans_model, "kmeans.pkl")
    results["kmeans"] = kmeans_metrics

    # --- Best classifier summary ------------------------------------------
    classifier_names = ["decision_tree", "random_forest", "xgboost"]
    best_name = max(classifier_names, key=lambda n: results[n]["f1_weighted"])
    results["best_classifier"] = best_name
    print(f"\n>>> Best classifier on test set: {best_name.upper()} "
          f"(F1 = {results[best_name]['f1_weighted']:.4f})")

    # --- Save metrics JSON -------------------------------------------------
    metrics_path = ARTIFACTS_DIR / "model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved metrics -> {metrics_path.relative_to(PROJECT_ROOT)}")
    print("\nAll models trained and saved successfully.")


if __name__ == "__main__":
    main()