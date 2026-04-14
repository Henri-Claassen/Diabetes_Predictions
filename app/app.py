import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd, joblib, numpy as np, os, base64

# --- LOAD MODELS ---
# We updated the paths to look in '../artifacts/' instead of '../models/'
_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
target_le = joblib.load(os.path.join(_base, 'artifacts', 'label_encoder.pkl'))

# TODO: You need to find these two files from your team and put them in the artifacts folder!
# feature_cols = joblib.load('../artifacts/feature_cols.pkl') 
# best_model = joblib.load('../artifacts/xgboost.pkl') 

# --- INITIALIZE APP ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server # REQUIRED for Render deployment

# --- LAYOUT ---
app.layout = dbc.Container([
    html.H1("Diabetes Risk Decision Support System", className="mt-4 mb-4"),
    dbc.Tabs([
        dbc.Tab(label="Patient Prediction", tab_id="tab-predict"),
        dbc.Tab(label="Model Performance", tab_id="tab-models"),
        dbc.Tab(label="SHAP Key Drivers", tab_id="tab-shap"),
        dbc.Tab(label="Patient Segments", tab_id="tab-clusters"),
    ], id="tabs", active_tab="tab-predict"),
    html.Div(id="tab-content", className="p-4")
])

# --- CALLBACKS ---
@app.callback(Output("tab-content", "children"),
              Input("tabs", "active_tab"))
def render_tab(tab):
    if tab == "tab-predict":
        return html.Div("Patient prediction form goes here. You will need to build the inputs and button.")
    elif tab == "tab-models":
        return html.Div("Model accuracy comparison bar chart and confusion matrix go here.")
    elif tab == "tab-shap":
        return html.Div("SHAP feature importance plots go here.")
    elif tab == "tab-clusters":
        return html.Div("K-Means cluster scatter plot and cluster profiles table go here.")
    
    return html.Div("Tab content not found.")

# --- RUN APP ---
if __name__ == '__main__':
    app.run(debug=True)