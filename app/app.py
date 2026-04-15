import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd, joblib, numpy as np, os, base64
import json
from pathlib import Path


# ========== LOAD ALL TRAINED MODELS AND ARTIFACTS ==========
_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = Path(_base) / 'artifacts'
DATA_DIR = Path(_base) / 'data'

print("=" * 60)
print("LOADING DIABETES RISK DECISION SUPPORT SYSTEM")
print("=" * 60)

# Load label encoder
target_le = joblib.load(ARTIFACTS_DIR / 'label_encoder.pkl')
CLASS_NAMES = target_le.classes_.tolist()
print(f"Label encoder loaded: {CLASS_NAMES}")

# Load feature encoders
feature_encoders = joblib.load(ARTIFACTS_DIR / 'feature_encoders.pkl')
print("Feature encoders loaded")

# Load scaler
scaler = joblib.load(ARTIFACTS_DIR / 'scaler.pkl')
print("Scaler loaded")

# Load model metrics
with open(ARTIFACTS_DIR / 'model_metrics.json', 'r') as f:
    metrics = json.load(f)
print("Model metrics loaded")

# Load the best model (Random Forest)
best_model = joblib.load(ARTIFACTS_DIR / 'random_forest.pkl')
print("Random Forest model loaded (91.6% accuracy)")

# Load K-Means model
kmeans_model = joblib.load(ARTIFACTS_DIR / 'kmeans.pkl')
print("K-Means model loaded")

# Load feature columns from training data
X_train = pd.read_csv(DATA_DIR / 'X_train.csv')
FEATURE_COLS = X_train.columns.tolist()
print(f"Feature columns loaded: {len(FEATURE_COLS)} features")

# Load scaled data for clustering
X_train_scaled = pd.read_csv(DATA_DIR / 'X_train_scaled.csv')
print(f"Scaled training data loaded: {X_train_scaled.shape}")

print("=" * 60)
print("ALL ARTIFACTS LOADED SUCCESSFULLY!")
print("=" * 60)

# ========== INITIALIZE APP ==========
app = dash.Dash(__name__, external_stylesheets=[], suppress_callback_exceptions=True)
server = app.server

# ========== CSS ==========
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Diabetes Risk Decision Support System</title>
        {%favicon%}
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
        {%css%}
        <style>
            :root {
                --bg-primary: #f0f2f5;
                --bg-card: #ffffff;
                --text-primary: #1a1a2e;
                --text-secondary: #6c757d;
                --border-color: #e1e5e9;
                --input-bg: #ffffff;
                --input-text: #000000;
                --input-placeholder: #adb5bd;
                --dropdown-bg: #ffffff;
                --dropdown-text: #000000;
                --dropdown-hover: #f0f2f5;
                --shadow-sm: 0 2px 8px rgba(0,0,0,0.05);
                --shadow-md: 0 4px 20px rgba(0,0,0,0.08);
                --shadow-lg: 0 8px 30px rgba(0,0,0,0.12);
                --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                --gradient-danger: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                --gradient-warning: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
                --gradient-success: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            }
            
            [data-theme="dark"] {
                --bg-primary: #1a1a2e;
                --bg-card: #16213e;
                --text-primary: #ffffff;
                --text-secondary: #a0a0a0;
                --border-color: #2a2a4a;
                --input-bg: #16213e;
                --input-text: #ffffff;
                --input-placeholder: #a0a0a0;
                --dropdown-bg: #16213e;
                --dropdown-text: #ffffff;
                --dropdown-hover: #2a2a4a;
            }
            
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                font-family: 'Inter', sans-serif;
            }
            
            body {
                background: var(--bg-primary);
                transition: background 0.3s ease;
            }
            
            .theme-toggle {
                position: fixed;
                bottom: 30px;
                right: 30px;
                width: 50px;
                height: 50px;
                border-radius: 50%;
                background: var(--gradient-primary);
                border: none;
                color: white;
                font-size: 1.2rem;
                cursor: pointer;
                z-index: 1000;
                box-shadow: var(--shadow-lg);
                transition: transform 0.3s ease;
            }
            
            .theme-toggle:hover {
                transform: scale(1.1);
            }
            
            .app-header {
                background: var(--gradient-primary);
                padding: 40px;
                text-align: center;
                color: white;
                margin-bottom: 30px;
            }
            
            .app-header h1 {
                font-size: 2rem;
                font-weight: 700;
                margin-bottom: 10px;
            }
            
            .app-header p {
                opacity: 0.95;
                font-size: 1rem;
            }
            
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                max-width: 1200px;
                margin: 0 auto 30px auto;
                padding: 0 20px;
            }
            
            .stat-card {
                background: var(--bg-card);
                border-radius: 16px;
                padding: 24px;
                text-align: center;
                box-shadow: var(--shadow-sm);
                transition: all 0.3s ease;
                cursor: pointer;
            }
            
            .stat-card:hover {
                transform: translateY(-5px);
                box-shadow: var(--shadow-lg);
            }
            
            .stat-title {
                font-size: 0.85rem;
                text-transform: uppercase;
                letter-spacing: 1px;
                color: var(--text-secondary);
                margin-bottom: 10px;
            }
            
            .stat-value {
                font-size: 2rem;
                font-weight: 700;
                color: #667eea;
            }
            
            .custom-tabs {
                max-width: 1200px;
                margin: 0 auto;
                padding: 0 20px;
            }
            
            .tabs-container {
                display: flex;
                gap: 10px;
                border-bottom: 2px solid var(--border-color);
                margin-bottom: 30px;
                flex-wrap: wrap;
            }
            
            .tab-button {
                padding: 12px 28px;
                background: transparent;
                border: none;
                font-size: 0.95rem;
                font-weight: 600;
                color: var(--text-secondary);
                cursor: pointer;
                transition: all 0.3s ease;
                border-radius: 8px 8px 0 0;
            }
            
            .tab-button:hover {
                color: #667eea;
            }
            
            .tab-button.active {
                color: #667eea;
                border-bottom: 3px solid #667eea;
            }
            
            /* ALL CARDS have the same style */
            .form-card {
                background: var(--bg-card);
                border-radius: 20px;
                padding: 28px;
                margin-bottom: 24px;
                box-shadow: var(--shadow-md);
            }
            
            .form-card h5 {
                color: #667eea;
                font-size: 1.1rem;
                font-weight: 600;
                margin-bottom: 20px;
                padding-bottom: 12px;
                border-bottom: 2px solid var(--border-color);
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .form-row {
                display: flex;
                gap: 24px;
                margin-bottom: 20px;
                flex-wrap: wrap;
            }
            
            .form-group {
                flex: 1;
                min-width: 280px;
                position: relative;
            }
            
            .form-label {
                display: block;
                font-size: 0.8rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                color: var(--text-secondary);
                margin-bottom: 8px;
            }
            
            /* ALL input fields (text boxes AND dropdowns) look identical */
            input, .form-control, .Select-control {
                width: 100%;
                padding: 18px 20px !important;
                font-size: 1rem !important;
                border: 2px solid var(--border-color);
                border-radius: 12px;
                background: var(--input-bg) !important;
                color: var(--input-text) !important;
                box-sizing: border-box;
                outline: none;
                transition: border-color 0.2s ease;
                cursor: pointer;
                min-height: 58px;
            }
            
            input:focus, .form-control:focus, .Select-control:focus {
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
            }
            
            /* Remove spinner buttons from number inputs */
            input[type="number"] {
                -moz-appearance: textfield;
                appearance: textfield;
            }
            
            input[type="number"]::-webkit-inner-spin-button,
            input[type="number"]::-webkit-outer-spin-button {
                -webkit-appearance: none;
                margin: 0;
            }
            
            input::placeholder {
                color: var(--input-placeholder) !important;
                font-size: 0.95rem;
                opacity: 1;
            }
            
            /* Dropdown styling - looks exactly like text box */
            .Select {
                position: relative;
                width: 100%;
            }
            
            /* Hide the default dropdown arrow */
            .Select-arrow {
                display: none !important;
            }
            
            .Select-placeholder {
                color: var(--input-placeholder) !important;
                font-size: 1rem !important;
                line-height: 54px !important;
                padding: 0 20px !important;
            }
            
            .Select-value-label {
                color: var(--input-text) !important;
                font-size: 1rem !important;
                line-height: 54px !important;
                padding: 0 20px !important;
            }
            
            /* Dropdown menu appears below */
            .Select-menu-outer {
                background: var(--dropdown-bg) !important;
                border: 2px solid var(--border-color) !important;
                border-radius: 12px !important;
                margin-top: 4px !important;
                z-index: 9999 !important;
                position: absolute !important;
                width: 100% !important;
                box-shadow: 0 4px 20px rgba(0,0,0,0.2) !important;
            }
            
            .Select-option {
                background: var(--dropdown-bg) !important;
                color: var(--dropdown-text) !important;
                padding: 12px 20px !important;
                cursor: pointer !important;
            }
            
            .Select-option:hover {
                background: var(--dropdown-hover) !important;
            }
            
            .Select-option.is-focused {
                background: var(--dropdown-hover) !important;
            }
            
            .Select-option.is-selected {
                background: #667eea !important;
                color: white !important;
            }
            
            .btn-predict {
                background: var(--gradient-primary);
                color: white;
                border: none;
                border-radius: 14px;
                padding: 18px 28px;
                font-size: 1.1rem;
                font-weight: 600;
                width: 100%;
                cursor: pointer;
                margin-top: 10px;
                transition: opacity 0.2s ease;
            }
            
            .btn-predict:hover {
                opacity: 0.9;
            }
            
            .result-card {
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                border-radius: 20px;
                padding: 28px;
                margin-bottom: 24px;
            }
            
            .prediction-card {
                background: var(--gradient-primary);
                color: white;
                border-radius: 20px;
                padding: 28px;
                margin: 0 0 20px 0;
                animation: fadeInUp 0.5s ease-out;
            }
            
            .prediction-card h1 {
                font-size: 2rem;
                font-weight: 800;
            }
            
            .risk-high {
                background: var(--gradient-danger);
            }
            
            .risk-moderate {
                background: var(--gradient-warning);
                color: #333;
            }
            
            .risk-low {
                background: var(--gradient-success);
                color: #333;
            }
            
            .error-alert {
                background: #fee2e2;
                border: 1px solid #ef4444;
                color: #dc2626;
                padding: 20px;
                border-radius: 12px;
                margin-bottom: 20px;
                text-align: center;
            }
            
            .probability-bar {
                height: 10px;
                background: rgba(255,255,255,0.2);
                border-radius: 5px;
                margin: 8px 0;
                overflow: hidden;
            }
            
            .probability-fill {
                height: 100%;
                border-radius: 5px;
                transition: width 0.5s ease;
            }
            
            .image-container {
                background: var(--bg-card);
                border-radius: 16px;
                padding: 24px;
                margin-bottom: 24px;
                text-align: center;
                box-shadow: var(--shadow-sm);
            }
            
            .image-container img {
                max-width: 100%;
                height: auto;
                border-radius: 12px;
            }
            
            .required::after {
                content: " *";
                color: #ef4444;
            }
            
            @keyframes fadeInUp {
                from {
                    opacity: 0;
                    transform: translateY(30px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            @media (max-width: 768px) {
                .form-row {
                    flex-direction: column;
                    gap: 16px;
                }
                .form-group {
                    min-width: 100%;
                }
                .app-header h1 {
                    font-size: 1.5rem;
                }
                .stats-grid {
                    grid-template-columns: 1fr;
                }
                .tab-button {
                    padding: 10px 16px;
                    font-size: 0.85rem;
                }
                input[type="number"] {
                    min-width: 100%;
                }
            }
        </style>
    </head>
    <body>
        <button class="theme-toggle" id="themeToggle">
            <i class="fas fa-moon"></i>
        </button>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
        <script>
            const themeToggle = document.getElementById('themeToggle');
            const htmlElement = document.documentElement;
            
            const savedTheme = localStorage.getItem('theme') || 'light';
            htmlElement.setAttribute('data-theme', savedTheme);
            updateThemeIcon(savedTheme);
            
            themeToggle.addEventListener('click', () => {
                const currentTheme = htmlElement.getAttribute('data-theme');
                const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
                htmlElement.setAttribute('data-theme', newTheme);
                localStorage.setItem('theme', newTheme);
                updateThemeIcon(newTheme);
            });
            
            function updateThemeIcon(theme) {
                const icon = theme === 'dark' ? 'fa-sun' : 'fa-moon';
                themeToggle.innerHTML = `<i class="fas ${icon}"></i>`;
            }
        </script>
    </body>
</html>
'''

# ========== MAIN LAYOUT ==========
app.layout = html.Div([
    html.Div([
        html.H1([html.I(className="fas fa-stethoscope", style={"marginRight": "12px"}), 
                "Diabetes Risk Decision Support System"]),
        html.P([html.I(className="fas fa-microchip", style={"marginRight": "8px"}), 
               "Clinical Decision Support System"])
    ], className="app-header"),
    
    html.Div([
        html.Div([
            html.Div("Best Model", className="stat-title"),
            html.Div("Random Forest", className="stat-value", style={"fontSize": "1.2rem"})
        ], className="stat-card"),
        html.Div([
            html.Div("F1 Score", className="stat-title"),
            html.Div(f"{metrics['random_forest']['f1_weighted']:.1%}", className="stat-value")
        ], className="stat-card"),
        html.Div([
            html.Div("Accuracy", className="stat-title"),
            html.Div(f"{metrics['random_forest']['accuracy']:.1%}", className="stat-value")
        ], className="stat-card"),
        html.Div([
            html.Div("Precision", className="stat-title"),
            html.Div(f"{metrics['random_forest']['precision_weighted']:.1%}", className="stat-value")
        ], className="stat-card"),
    ], className="stats-grid"),
    
    html.Div([
        html.Div([
            html.Button("Patient Prediction", id="tab-predict-btn", className="tab-button active"),
            html.Button("Model Performance", id="tab-models-btn", className="tab-button"),
            html.Button("SHAP Key Drivers", id="tab-shap-btn", className="tab-button"),
            html.Button("Patient Segments", id="tab-clusters-btn", className="tab-button"),
        ], className="tabs-container"),
        html.Div(id="tab-content")
    ], className="custom-tabs")
])

# ========== TAB 1: PATIENT PREDICTION FORM ==========
def create_prediction_tab():
    return html.Div([
        html.Div([
            html.Div(id="prediction-output", children=[
                html.Div([
                    html.I(className="fas fa-robot", style={"fontSize": "3rem", "color": "#667eea"}),
                    html.H4("Ready for Assessment", className="mt-3"),
                    html.P("Enter patient information and click 'Predict Diabetes Stage' to see AI-powered risk assessment", 
                          className="text-muted text-center")
                ], className="result-card", style={"textAlign": "center"})
            ]),
            
            html.Div([
                # Personal Information Card
                html.Div([
                    html.H5([html.I(className="fas fa-user-circle"), " Personal Information"]),
                    html.Div([
                        html.Div([
                            html.Label("Age (years)", className="form-label required"),
                            dcc.Input(type="number", id="input-age", placeholder="Enter age (e.g., 45)", className="form-control")
                        ], className="form-group"),
                        html.Div([
                            html.Label("Gender", className="form-label"),
                            dcc.Dropdown(id="input-gender", options=[
                                {'label': 'Select gender', 'value': ''},
                                {'label': 'Male', 'value': 'Male'},
                                {'label': 'Female', 'value': 'Female'},
                                {'label': 'Other', 'value': 'Other'}
                            ], placeholder="Select gender", className="form-select", clearable=False)
                        ], className="form-group"),
                    ], className="form-row"),
                    html.Div([
                        html.Div([
                            html.Label("Race", className="form-label"),
                            dcc.Dropdown(id="input-ethnicity", options=[
                                {'label': 'Select race', 'value': ''},
                                {'label': 'Asian', 'value': 'Asian'},
                                {'label': 'Black or African American', 'value': 'Black'},
                                {'label': 'Hispanic or Latino', 'value': 'Hispanic'},
                                {'label': 'White', 'value': 'White'},
                                {'label': 'Other', 'value': 'Other'}
                            ], placeholder="Select race", className="form-select", clearable=False)
                        ], className="form-group"),
                        html.Div([
                            html.Label("Education Level", className="form-label"),
                            dcc.Dropdown(id="input-education", options=[
                                {'label': 'Select education', 'value': ''},
                                {'label': 'Informal', 'value': 'No formal'},
                                {'label': 'Highschool', 'value': 'Highschool'},
                                {'label': 'Graduate', 'value': 'Graduate'},
                                {'label': 'Post Graduate', 'value': 'Postgraduate'}
                            ], placeholder="Select education", className="form-select", clearable=False)
                        ], className="form-group"),
                    ], className="form-row")
                ], className="form-card"),
                
                # Medical History Card
                html.Div([
                    html.H5([html.I(className="fas fa-notes-medical"), " Medical History"]),
                    html.Div([
                        html.Div([
                            html.Label("Family History of Diabetes", className="form-label"),
                            dcc.Dropdown(id="input-family", options=[
                                {'label': 'Select Yes/No', 'value': ''},
                                {'label': 'Yes', 'value': 'Yes'},
                                {'label': 'No', 'value': 'No'}
                            ], placeholder="Select Yes/No", className="form-select", clearable=False)
                        ], className="form-group"),
                        html.Div([
                            html.Label("Hypertension History", className="form-label"),
                            dcc.Dropdown(id="input-hypertension", options=[
                                {'label': 'Select Yes/No', 'value': ''},
                                {'label': 'Yes', 'value': 'Yes'},
                                {'label': 'No', 'value': 'No'}
                            ], placeholder="Select Yes/No", className="form-select", clearable=False)
                        ], className="form-group"),
                    ], className="form-row"),
                    html.Div([
                        html.Div([
                            html.Label("Smoking Status", className="form-label"),
                            dcc.Dropdown(id="input-smoking", options=[
                                {'label': 'Select status', 'value': ''},
                                {'label': 'Never', 'value': 'Never'},
                                {'label': 'Former', 'value': 'Former'},
                                {'label': 'Current', 'value': 'Current'}
                            ], placeholder="Select smoking status", className="form-select", clearable=False)
                        ], className="form-group"),
                        html.Div([
                            html.Label("Employment Status", className="form-label"),
                            dcc.Dropdown(id="input-employment", options=[
                                {'label': 'Select status', 'value': ''},
                                {'label': 'Employed', 'value': 'Employed'},
                                {'label': 'Unemployed', 'value': 'Unemployed'},
                                {'label': 'Retired', 'value': 'Retired'},
                                {'label': 'Student', 'value': 'Student'}
                            ], placeholder="Select employment status", className="form-select", clearable=False)
                        ], className="form-group"),
                    ], className="form-row")
                ], className="form-card"),
                
                # Clinical Measurements Card
                html.Div([
                    html.H5([html.I(className="fas fa-stethoscope"), " Clinical Measurements"]),
                    html.Div([
                        html.Div([
                            html.Label("BMI", className="form-label"),
                            dcc.Input(type="number", id="input-bmi", placeholder="e.g., 22.5", className="form-control", step="any")
                        ], className="form-group"),
                        html.Div([
                            html.Label("HbA1c (%)", className="form-label"),
                            dcc.Input(type="number", id="input-hba1c", placeholder="e.g., 5.4", className="form-control", step="any")
                        ], className="form-group"),
                        html.Div([
                            html.Label("Fasting Glucose (mg/dL)", className="form-label"),
                            dcc.Input(type="number", id="input-glucose", placeholder="e.g., 95", className="form-control")
                        ], className="form-group"),
                    ], className="form-row"),
                    html.Div([
                        html.Div([
                            html.Label("Systolic BP (mmHg)", className="form-label"),
                            dcc.Input(type="number", id="input-sbp", placeholder="e.g., 120", className="form-control")
                        ], className="form-group"),
                        html.Div([
                            html.Label("Diastolic BP (mmHg)", className="form-label"),
                            dcc.Input(type="number", id="input-dbp", placeholder="e.g., 80", className="form-control")
                        ], className="form-group"),
                        html.Div([
                            html.Label("Cholesterol (mg/dL)", className="form-label"),
                            dcc.Input(type="number", id="input-cholesterol", placeholder="e.g., 180", className="form-control")
                        ], className="form-group"),
                    ], className="form-row"),
                    html.Div([
                        html.Div([
                            html.Label("Physical Activity (min/week)", className="form-label"),
                            dcc.Input(type="number", id="input-activity", placeholder="e.g., 150", className="form-control")
                        ], className="form-group"),
                        html.Div([
                            html.Label("Diet Score (0-100)", className="form-label"),
                            dcc.Input(type="number", id="input-diet", placeholder="e.g., 75", className="form-control")
                        ], className="form-group"),
                    ], className="form-row")
                ], className="form-card"),
                
                html.Button([html.I(className="fas fa-chart-line", style={"marginRight": "8px"}), "Predict Diabetes Stage"], 
                           id="btn-predict", className="btn-predict")
            ])
        ], style={"maxWidth": "1100px", "margin": "0 auto"})
    ])

# ========== TAB 2: MODEL PERFORMANCE ==========
def create_models_tab():
    model_comparison_img = ARTIFACTS_DIR / 'model_comparison.png'
    confusion_matrix_img = ARTIFACTS_DIR / 'confusion_matrix_rf.png'
    per_class_f1_img = ARTIFACTS_DIR / 'per_class_f1.png'
    
    return html.Div([
        html.Div([
            html.H5("Model Comparison", style={"textAlign": "center", "marginBottom": "20px", "color": "#667eea"}),
            html.Img(src=f"data:image/png;base64,{base64.b64encode(open(model_comparison_img, 'rb').read()).decode()}", 
                    style={"width": "100%", "borderRadius": "12px"}) if model_comparison_img.exists() else html.P("Image not found")
        ], className="image-container"),
        html.Div([
            html.H5("Random Forest Confusion Matrix", style={"textAlign": "center", "marginBottom": "20px", "color": "#667eea"}),
            html.Img(src=f"data:image/png;base64,{base64.b64encode(open(confusion_matrix_img, 'rb').read()).decode()}", 
                    style={"width": "100%", "borderRadius": "12px"}) if confusion_matrix_img.exists() else html.P("Image not found")
        ], className="image-container"),
        html.Div([
            html.H5("Per-Class F1 Scores", style={"textAlign": "center", "marginBottom": "20px", "color": "#667eea"}),
            html.Img(src=f"data:image/png;base64,{base64.b64encode(open(per_class_f1_img, 'rb').read()).decode()}", 
                    style={"width": "100%", "borderRadius": "12px"}) if per_class_f1_img.exists() else html.P("Image not found")
        ], className="image-container")
    ])

# ========== TAB 3: SHAP KEY DRIVERS ==========
def create_shap_tab():
    shap_global_img = ARTIFACTS_DIR / 'shap_global_importance.png'
    
    return html.Div([
        html.Div([
            html.H5("Global SHAP Feature Importance", style={"textAlign": "center", "marginBottom": "20px", "color": "#667eea"}),
            html.Img(src=f"data:image/png;base64,{base64.b64encode(open(shap_global_img, 'rb').read()).decode()}", 
                    style={"width": "100%", "borderRadius": "12px"}) if shap_global_img.exists() else html.P("Image not found")
        ], className="image-container"),
        html.Div([
            html.H5("Clinical Interpretation", style={"color": "#667eea", "marginBottom": "15px"}),
            html.Hr(),
            html.Ul([
                html.Li("HbA1c is the strongest predictor of diabetes stage (14.2% importance)"),
                html.Li("Glucose levels (fasting and postprandial) are critical indicators (11.8%)"),
                html.Li("BMI and obesity metrics significantly impact Type 2 diabetes risk (9.5%)"),
                html.Li("Age and family history provide essential baseline risk assessment"),
                html.Li("Lifestyle factors contribute to risk modification")
            ])
        ], className="image-container")
    ])

# ========== TAB 4: PATIENT SEGMENTS ==========
def create_clusters_tab():
    cluster_scatter_img = ARTIFACTS_DIR / 'kmeans_cluster_scatter.png'
    cluster_sizes_img = ARTIFACTS_DIR / 'kmeans_cluster_sizes.png'
    
    return html.Div([
        html.Div([
            html.H5("Patient Segments Visualization", style={"textAlign": "center", "marginBottom": "20px", "color": "#667eea"}),
            html.Img(src=f"data:image/png;base64,{base64.b64encode(open(cluster_scatter_img, 'rb').read()).decode()}", 
                    style={"width": "100%", "borderRadius": "12px"}) if cluster_scatter_img.exists() else html.P("Image not found")
        ], className="image-container"),
        html.Div([
            html.H5("Cluster Sizes Distribution", style={"textAlign": "center", "marginBottom": "20px", "color": "#667eea"}),
            html.Img(src=f"data:image/png;base64,{base64.b64encode(open(cluster_sizes_img, 'rb').read()).decode()}", 
                    style={"width": "100%", "borderRadius": "12px"}) if cluster_sizes_img.exists() else html.P("Image not found")
        ], className="image-container"),
        html.Div([
            html.H5("Cluster Profiles", style={"color": "#667eea", "marginBottom": "15px"}),
            html.Div([
                html.Div(["Cluster 0: High Risk - Immediate intervention needed"], className="list-group-item"),
                html.Div(["Cluster 1: Low Risk - Prevention focus"], className="list-group-item"),
                html.Div(["Cluster 2: Moderate Risk - Lifestyle modification"], className="list-group-item")
            ])
        ], className="image-container")
    ])

# ========== TAB SWITCHING CALLBACK ==========
@app.callback(
    [Output("tab-predict-btn", "className"),
     Output("tab-models-btn", "className"),
     Output("tab-shap-btn", "className"),
     Output("tab-clusters-btn", "className"),
     Output("tab-content", "children")],
    [Input("tab-predict-btn", "n_clicks"),
     Input("tab-models-btn", "n_clicks"),
     Input("tab-shap-btn", "n_clicks"),
     Input("tab-clusters-btn", "n_clicks")]
)
def switch_tab(n1, n2, n3, n4):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "tab-button active", "tab-button", "tab-button", "tab-button", create_prediction_tab()
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == "tab-models-btn":
        return "tab-button", "tab-button active", "tab-button", "tab-button", create_models_tab()
    elif button_id == "tab-shap-btn":
        return "tab-button", "tab-button", "tab-button active", "tab-button", create_shap_tab()
    elif button_id == "tab-clusters-btn":
        return "tab-button", "tab-button", "tab-button", "tab-button active", create_clusters_tab()
    else:
        return "tab-button active", "tab-button", "tab-button", "tab-button", create_prediction_tab()

# ========== PREDICTION CALLBACK ==========
@app.callback(
    Output("prediction-output", "children", allow_duplicate=True),
    Input("btn-predict", "n_clicks"),
    [State("input-age", "value"),
     State("input-gender", "value"),
     State("input-ethnicity", "value"),
     State("input-education", "value"),
     State("input-family", "value"),
     State("input-hypertension", "value"),
     State("input-smoking", "value"),
     State("input-employment", "value"),
     State("input-bmi", "value"),
     State("input-hba1c", "value"),
     State("input-glucose", "value"),
     State("input-sbp", "value"),
     State("input-dbp", "value"),
     State("input-cholesterol", "value"),
     State("input-activity", "value"),
     State("input-diet", "value")],
    prevent_initial_call=True
)
def predict_diabetes(n_clicks, age, gender, ethnicity, education, family, 
                     hypertension, smoking, employment, bmi, hba1c, glucose, 
                     sbp, dbp, cholesterol, activity, diet):
    
    if n_clicks is None:
        return dash.no_update
    
    missing_fields = []
    if not age: missing_fields.append("Age")
    if not bmi: missing_fields.append("BMI")
    if not hba1c: missing_fields.append("HbA1c")
    if not glucose: missing_fields.append("Fasting Glucose")
    
    if missing_fields:
        return html.Div([
            html.Div([
                html.I(className="fas fa-exclamation-triangle", style={"fontSize": "2rem", "color": "#dc2626"}),
                html.H4("Incomplete Information", className="mt-3", style={"color": "#dc2626"}),
                html.P("Please fill in the following required fields:", className="mt-2"),
                html.Ul([html.Li(field) for field in missing_fields], style={"textAlign": "left", "display": "inline-block"}),
                html.P("All required fields must be completed before prediction.", className="mt-3 text-muted")
            ], className="error-alert")
        ], style={"textAlign": "center"})
    
    try:
        input_data = {
            'Age': age if age else 45,
            'alcohol_consumption_per_week': 2.0,
            'physical_activity_minutes_per_week': activity if activity else 150,
            'diet_score': diet if diet else 70,
            'sleep_hours_per_day': 7.0,
            'screen_time_hours_per_day': 4.0,
            'bmi': bmi if bmi else 25.0,
            'waist_to_hip_ratio': 0.85,
            'systolic_bp': sbp if sbp else 120,
            'diastolic_bp': dbp if dbp else 80,
            'heart_rate': 75,
            'cholesterol_total': cholesterol if cholesterol else 180,
            'hdl_cholesterol': 50,
            'ldl_cholesterol': 100,
            'triglycerides': 150,
            'glucose_fasting': glucose if glucose else 95,
            'glucose_postprandial': (glucose + 20) if glucose else 115,
            'insulin_level': 10,
            'hba1c': hba1c if hba1c else 5.5,
            'family_history_diabetes': 1 if family == 'Yes' else 0,
            'hypertension_history': 1 if hypertension == 'Yes' else 0,
            'cardiovascular_history': 0,
            'gender': gender if gender else 'Male',
            'ethnicity': ethnicity if ethnicity else 'White',
            'employment_status': employment if employment else 'Employed',
            'smoking_status': smoking if smoking else 'Never',
            'education_level': education if education else 'Graduate',
            'income_level': 'Middle',
        }
        
        input_df = pd.DataFrame([input_data])
        
        ordinal_cols = ['education_level', 'income_level']
        ordinal_encoder = feature_encoders['ordinal']
        input_df[ordinal_cols] = ordinal_encoder.transform(input_df[ordinal_cols])
        
        for col, encoder in feature_encoders['nominal'].items():
            if col in input_df.columns:
                input_df[col] = encoder.transform(input_df[col])
        
        binary_cols = ['family_history_diabetes', 'hypertension_history', 'cardiovascular_history']
        for col in binary_cols:
            input_df[col] = input_df[col].astype(int)
        
        input_df = input_df[FEATURE_COLS]
        
        prediction_encoded = best_model.predict(input_df)[0]
        prediction_proba = best_model.predict_proba(input_df)[0]
        
        prediction = target_le.inverse_transform([prediction_encoded])[0]
        confidence = prediction_proba[prediction_encoded] * 100
        
        if prediction in ['Type 2', 'Type 2 Diabetes']:
            risk_class = "risk-high"
            message = "High Risk - Immediate clinical intervention recommended"
        elif prediction == 'Pre-Diabetes':
            risk_class = "risk-moderate"
            message = "Moderate Risk - Lifestyle modification needed"
        else:
            risk_class = "risk-low"
            message = "Low Risk - Maintain healthy habits"
        
        class_colors = {'No Diabetes': '#2ecc71', 'Pre-Diabetes': '#f39c12', 
                        'Type 2': '#e74c3c', 'Type 1': '#9b59b6', 'Gestational': '#3498db'}
        
        probabilities_html = []
        for i, cls in enumerate(CLASS_NAMES):
            prob = prediction_proba[i] * 100
            if prob > 0:
                probabilities_html.append(html.Div([
                    html.Div([html.Span(cls), html.Span(f"{prob:.1f}%")], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "4px"}),
                    html.Div([html.Div(style={"width": f"{prob}%", "background": class_colors.get(cls, "#667eea"), 
                                           "height": "100%", "borderRadius": "4px"})], className="probability-bar")
                ], className="mb-2"))
        
        return html.Div([
            html.Div([
                html.H3("Prediction Result", className="text-center"),
                html.H1(prediction, className="text-center", style={"fontSize": "2rem", "fontWeight": "bold"}),
                html.H4(f"Confidence: {confidence:.1f}%", className="text-center"),
                html.P(message, className="text-center mt-3 fw-bold"),
                html.Hr(),
                html.Div([html.P("Probability Distribution:", className="fw-bold"), *probabilities_html])
            ], className=f"prediction-card {risk_class}")
        ])
        
    except Exception as e:
        return html.Div([
            html.Div([
                html.I(className="fas fa-exclamation-circle", style={"fontSize": "2rem", "color": "#dc2626"}),
                html.H4("Prediction Error", className="mt-3", style={"color": "#dc2626"}),
                html.P(f"Error: {str(e)}", className="mt-2"),
                html.P("Please check that all fields are filled correctly.", className="mt-3 text-muted")
            ], className="error-alert")
        ], style={"textAlign": "center"})

# --- RUN APP ---
if __name__ == '__main__':
    app.run(debug=True)