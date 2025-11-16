# ============================================================
# app.py — MAIN STREAMLIT APPLICATION (ROOT-BASED VERSION)
# All assets are loaded directly from the GitHub root.
# ============================================================

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
import os

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="Cellular Beam Inverse Design Tool",
    layout="wide"
)
st.title("🧠 Cellular Beam Inverse Design Tool")

# ============================================================
# ROOT-LEVEL PATHS (all assets stored in GitHub root)
# ============================================================
BEST_MODEL      = "best_model.joblib"
SCALER_MODEL    = "scaler.joblib"
FEAT_COLS_FILE  = "feature_cols.joblib"

FAIL_MODEL      = "best_failure_classifier.joblib"
FAIL_SCALER     = "scaler_failure.joblib"
FAIL_LABELS     = "label_encoder.joblib"

DATA_FILE       = "original_data.xlsx"

# ============================================================
# LOAD CACHED ASSETS
# ============================================================

@st.cache_resource
def load_forward_assets():
    """Load Phase 1 surrogate model and scaler."""
    model = joblib.load(BEST_MODEL)
    scaler = joblib.load(SCALER_MODEL)
    feature_cols = joblib.load(FEAT_COLS_FILE)
    return model, scaler, feature_cols


@st.cache_resource
def load_failure_mode_assets():
    """Load Phase 5 failure mode classifier."""
    clf = joblib.load(FAIL_MODEL)
    scaler_clf = joblib.load(FAIL_SCALER)
    encoder = joblib.load(FAIL_LABELS)
    return clf, scaler_clf, encoder


@st.cache_resource
def load_dataset():
    """Load main FEA dataset from root."""
    df = pd.read_excel(DATA_FILE)
    return df

# ============================================================
# INITIAL LOAD
# ============================================================

try:
    model_forward, scaler_forward, feature_cols = load_forward_assets()
    clf_failure, scaler_failure, label_encoder = load_failure_mode_assets()
    df_full = load_dataset()
    st.success("✔ All pipeline assets loaded successfully from root.")
except Exception as e:
    st.error("❌ Failed to load assets from root directory.")
    st.code(str(e))
    st.stop()

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.header("📌 Navigation")

page = st.sidebar.radio(
    "Choose a module:",
    [
        "🏗 Designer Tool (Inverse Design)",
        "📊 Interpretability & Diagnostics",
        "📁 Database Explorer",
        "📘 Methodology"
    ]
)

# ============================================================
# PAGE ROUTING
# ============================================================

if page == "🏗 Designer Tool (Inverse Design)":
    import designer_page
    designer_page.render(
        model_forward, scaler_forward, feature_cols,
        clf_failure, scaler_failure, label_encoder, df_full
    )

elif page == "📊 Interpretability & Diagnostics":
    import interpret_page
    interpret_page.render(model_forward, scaler_forward, feature_cols, df_full)

elif page == "📁 Database Explorer":
    import explorer_page
    explorer_page.render(df_full)

elif page == "📘 Methodology":
    import methodology_page
    methodology_page.render()

# ============================================================
# END OF MAIN FILE
# ============================================================
