# Diabetes Risk Decision Support System

A machine learning web application that predicts diabetes risk and segments patients by lifestyle profile. Built with Python, Dash, and scikit-learn as part of the MLG382 Guided Project.

---

## What It Does

This system takes patient health and lifestyle data and:

- **Predicts diabetes stage** — classifies patients into one of five categories: No Diabetes, Pre-Diabetes, Type 1, Type 2, or Gestational Diabetes
- **Segments patients by lifestyle** — uses K-Means clustering (k=3) to group patients by shared lifestyle patterns
- **Explains predictions** — uses SHAP values to show which features drive each prediction, both globally and per-class
- **Compares models** — presents side-by-side metrics for all trained classifiers so you can see how each performed

The interactive Dash dashboard lets you enter patient data, get a risk prediction, view feature importance, and explore cluster profiles — all in one place.

---

## Model Performance

Three classifiers were trained and evaluated on a held-out test set. **Random Forest is the best-performing model**, achieving the highest accuracy and F1 score across all classes:

| Model | Accuracy | Weighted F1 |
|---|---|---|
| **Random Forest** | **91.6%** | **0.915** |
| Decision Tree | 88.9% | 0.897 |
| XGBoost | 80.1% | 0.850 |

Random Forest was selected as the production model for the dashboard's live predictions.

---

## Project Structure

```
MLG382-Guided-Project/
│
├── app/
│   └── app.py                  # Dash web application
│
├── src/
│   ├── prepare_data.py         # Raw data loading and initial preparation
│   ├── preprocess_data.py      # Feature encoding, scaling, train/test split
│   ├── train_models.py         # Model training, tuning, and evaluation
│   └── SHAP_analysis.py        # SHAP explainability analysis
│
├── notebooks/
│   ├── EDA.ipynb               # Exploratory data analysis
│   ├── modeling.ipynb          # Model development walkthrough
│   └── shap_analysis.ipynb     # SHAP analysis notebook
│
├── data/
│   ├── Diabetes_and_LifeStyle_Dataset.csv  # Raw dataset
│   ├── train.csv / test.csv                # Split raw data
│   ├── X_train.csv / X_test.csv           # Processed features
│   ├── X_train_scaled.csv / X_test_scaled.csv  # Scaled features (K-Means)
│   └── y_train.csv / y_test.csv           # Labels
│
├── artifacts/
│   ├── random_forest.pkl       # Best model (91.6% accuracy)
│   ├── decision_tree.pkl       # Decision Tree model
│   ├── xgboost.pkl             # XGBoost model
│   ├── kmeans.pkl              # K-Means clustering model
│   ├── scaler.pkl              # Feature scaler
│   ├── label_encoder.pkl       # Target label encoder
│   ├── feature_encoders.pkl    # Categorical feature encoders
│   ├── model_metrics.json      # All model evaluation metrics
│   └── *.png                   # Plots (confusion matrices, SHAP, clusters)
│
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip

### Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/Henri-Claassen/MLG382-Guided-Project.git
   cd MLG382-Guided-Project
   ```

2. **Create and activate a virtual environment** (recommended)

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the web application**

   ```bash
   python app/app.py
   ```

   Then open your browser and go to `http://127.0.0.1:8050`

---

## Retraining the Models

If you want to retrain from scratch, run the pipeline scripts in order:

```bash
python src/prepare_data.py
python src/preprocess_data.py
python src/train_models.py
python src/SHAP_analysis.py
```

All outputs (models, metrics, plots) will be saved to the `artifacts/` directory.

---

## Dependencies

| Package | Version |
|---|---|
| dash | 4.1.0 |
| dash-bootstrap-components | 1.6.0 |
| plotly | 5.24.1 |
| pandas | 2.2.3 |
| numpy | 2.4.4 |
| scikit-learn | 1.8.0 |
| xgboost | 3.2.0 |
| shap | 0.51.0 |
| joblib | 1.5.3 |
| scipy | 1.17.1 |
| Flask | 3.0.3 |
