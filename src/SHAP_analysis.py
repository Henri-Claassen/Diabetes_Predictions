"""
SHAP_analysis.py
----------------
Reusable SHAP functions for the Diabetes Risk Decision Support System.
Called by notebooks/shap_analysis.ipynb and the Dash web app.
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


# ---------------------------------------------------------------------------
# 1. Load model
# ---------------------------------------------------------------------------
def load_model(filename):
    """Load any pickled model from artifacts/."""
    with open(ARTIFACTS / filename, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# 2. Create SHAP explainer for tree models (XGBoost, RF, DT)
# ---------------------------------------------------------------------------
def get_tree_explainer(model):
    """Return a TreeExplainer for the given model."""
    return shap.TreeExplainer(model)


# ---------------------------------------------------------------------------
# 3. Compute SHAP values
# ---------------------------------------------------------------------------
def compute_shap_values(explainer, X_sample):
    """
    Compute SHAP values for a sample.
    Always returns a list of 2D arrays — one per class,
    regardless of SHAP version output format.
    """
    raw = explainer.shap_values(X_sample)

    # Newer SHAP versions return 3D array (samples, features, classes)
    if isinstance(raw, np.ndarray) and raw.ndim == 3:
        return [raw[:, :, i] for i in range(raw.shape[2])]

    # Older versions return a list of (samples, features) arrays
    return raw


# ---------------------------------------------------------------------------
# 4. Global bar chart — top features across all classes
# ---------------------------------------------------------------------------
def plot_global_importance(shap_values, X_sample, class_names, save_path=None, max_display=15):
    """
    Bar chart showing mean |SHAP| per feature across all classes.
    """
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        class_names=class_names,
        show=False,
        max_display=max_display
    )
    plt.title("SHAP Feature Importance — All Classes", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# 5. Beeswarm plot — direction of impact for one class
# ---------------------------------------------------------------------------
def plot_beeswarm(shap_values_class, X_sample, class_name, save_path=None, max_display=15):
    """
    Beeswarm plot for a single class.
    shap_values_class = shap_values[i] for class i.
    """
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values_class,
        X_sample,
        show=False,
        max_display=max_display
    )
    plt.title(f"SHAP Beeswarm — {class_name}", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# 6. Waterfall plot — single patient explanation
# ---------------------------------------------------------------------------
def plot_waterfall(explainer, shap_values, X_sample, patient_index,
                   predicted_class_idx, class_names, save_path=None):
    """
    Waterfall plot explaining one patient's prediction.
    Returns the predicted class name.
    """
    predicted_class_name = class_names[predicted_class_idx]

    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[predicted_class_idx][patient_index],
            base_values=explainer.expected_value[predicted_class_idx],
            data=X_sample.iloc[patient_index],
            feature_names=X_sample.columns.tolist()
        )
    )
    plt.title(f"Prediction explanation — {predicted_class_name}", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
    return predicted_class_name


# ---------------------------------------------------------------------------
# 7. SHAP for KMeans — proxy RF approach
# ---------------------------------------------------------------------------
def get_cluster_explainer(kmeans_model, X_train_scaled, X_test_scaled):
    """
    KMeans has no native SHAP support.
    Train a proxy Random Forest to predict cluster labels,
    then return its explainer and SHAP values on the test set.
    """
    cluster_labels_train = kmeans_model.predict(X_train_scaled)

    rf_proxy = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_proxy.fit(X_train_scaled, cluster_labels_train)

    explainer = shap.TreeExplainer(rf_proxy)
    shap_values = explainer.shap_values(X_test_scaled)

    return explainer, shap_values, rf_proxy


# ---------------------------------------------------------------------------
# 8. Cluster importance bar chart
# ---------------------------------------------------------------------------
def plot_cluster_importance(shap_values_cluster, X_test_scaled,
                             cluster_names=None, save_path=None, max_display=15):
    """
    Bar chart showing which lifestyle features define each cluster.
    """
    if cluster_names is None:
        cluster_names = ['Cluster 0', 'Cluster 1', 'Cluster 2']

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values_cluster,
        X_test_scaled,
        plot_type="bar",
        class_names=cluster_names,
        show=False,
        max_display=max_display
    )
    plt.title("SHAP Feature Importance — Lifestyle Cluster Drivers", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# 9. Cluster profile table
# ---------------------------------------------------------------------------
def build_cluster_profiles(kmeans_model, X_train_scaled, save_path=None):
    """
    Compute mean feature values per cluster.
    Returns a DataFrame — rows = features, columns = clusters.
    """
    labels = kmeans_model.predict(X_train_scaled)
    X_copy = X_train_scaled.copy()
    X_copy['cluster'] = labels

    profile = X_copy.groupby('cluster').mean().T
    profile.columns = [f'Cluster {i}' for i in profile.columns]

    if save_path:
        profile.to_csv(save_path)
        print(f"Saved: {save_path}")

    return profile


# ---------------------------------------------------------------------------
# 10. Single patient SHAP — for Dash app (returns values, no plot)
# ---------------------------------------------------------------------------
def explain_single_patient(explainer, model, patient_df, class_names):
    """
    Used by the Dash app to explain a single patient prediction.
    Returns a dict with predicted class, SHAP values, and feature names.
    No matplotlib — returns raw data for the app to render.
    """
    shap_values = explainer.shap_values(patient_df)
    predicted_class_idx = int(model.predict(patient_df)[0])
    predicted_class_name = class_names[predicted_class_idx]

    feature_shap = pd.Series(
        shap_values[predicted_class_idx][0],
        index=patient_df.columns
    ).sort_values(key=abs, ascending=False)

    return {
        "predicted_class": predicted_class_name,
        "predicted_class_idx": predicted_class_idx,
        "shap_series": feature_shap,
        "base_value": explainer.expected_value[predicted_class_idx]
    }
