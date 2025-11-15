import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(
    page_title="Cellular Beam Inverse Design Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 Cellular Beam Inverse Design Tool")


# -----------------------------------------------------------
# LOAD PHASE-1 ASSETS
# -----------------------------------------------------------
@st.cache_resource
def load_forward_assets():
    model = joblib.load("best_model.joblib")
    scaler = joblib.load("scaler.joblib")
    features = joblib.load("feature_cols.joblib")
    return model, scaler, features


# -----------------------------------------------------------
# LOAD DATASET
# -----------------------------------------------------------
@st.cache_resource
def load_dataset():
    # User must upload entire FEA dataset (the one used in all phases)
    df = pd.read_excel("21.xlsx")
    df.columns = [str(c).replace("\n"," ").replace('"','').replace(" ","_") for c in df.columns]

    # Enrich dataset exactly as Phase 1
    df["L/H"] = df["L"]/df["H"]
    df["H/ho"] = df["H"]/df["ho"]
    df["s/ho"] = df["s"]/df["ho"]
    df["tf/tw"] = df["tf"]/df["tw"]
    df["bf/H"] = df["bf"]/df["H"]
    df["Area"] = df["bf"]*df["tf"] + df["tw"]*(df["H"] - 2*df["tf"])
    df["fy_Area"] = df["fy"]*df["Area"]

    return df


# -----------------------------------------------------------
# FAILURE-MODE ASSETS
# (Phase 5 — classifier + encoder + rebuild scaler_clf)
# -----------------------------------------------------------
@st.cache_resource
def load_failure_mode_assets():
    clf = joblib.load("best_failure_classifier.joblib")
    encoder = joblib.load("label_encoder.joblib")

    df = load_dataset()
    df = df.dropna(subset=["Failure_mode"])
    feature_cols = joblib.load("feature_cols.joblib")

    scaler_clf = MinMaxScaler()
    scaler_clf.fit(df[feature_cols])

    return clf, encoder, scaler_clf


# -----------------------------------------------------------
# Feature engineering (same for all phases)
# -----------------------------------------------------------
def enrich(df):
    df = df.copy()
    df["L/H"]   = df["L"]/df["H"]
    df["H/ho"]  = df["H"]/df["ho"]
    df["s/ho"]  = df["s"]/df["ho"]
    df["tf/tw"] = df["tf"]/df["tw"]
    df["bf/H"]  = df["bf"]/df["H"]
    df["Area"]  = df["bf"]*df["tf"] + df["tw"]*(df["H"] - 2*df["tf"])
    df["fy_Area"] = df["fy"] * df["Area"]
    return df


# -----------------------------------------------------------
# LOAD INVERSE DESIGN RESULTS (Phase 2)
# -----------------------------------------------------------
def load_phase2():
    try:
        with open("result.json") as f:
            return json.load(f)
    except:
        return None


# -----------------------------------------------------------
# LOAD MULTI-OBJECTIVE RESULTS (Phase 3)
# -----------------------------------------------------------
def load_phase3():
    try:
        with open("best_compromise.json") as f:
            return json.load(f)
    except:
        return None


# -----------------------------------------------------------
# CODE CHECK (PHASE 6)
# -----------------------------------------------------------
def run_code_check(design, model, scaler, feature_cols, df):
    feats = pd.DataFrame([design])
    feats = enrich(feats)
    X = feats[feature_cols]
    wu_pred = float(model.predict(scaler.transform(X))[0])

    # Nearest neighbour lookup
    NN = NearestNeighbors(n_neighbors=1)
    NN.fit(df[feature_cols])
    idx = NN.kneighbors(scaler.transform(X))[1][0][0]
    row = df.iloc[idx]

    return {
        "wu_pred": wu_pred,
        "SCI":   {"R": row["wSCI"], "Safe": wu_pred <= row["wSCI"]},
        "EN_M":  {"R": row["wENM"], "Safe": wu_pred <= row["wENM"]},
        "EN_A":  {"R": row["wENA"], "Safe": wu_pred <= row["wENA"]},
        "AISC":  {"R": row["wAISC"], "Safe": wu_pred <= row["wAISC"]},
    }


# ===========================================================
# SIDEBAR NAVIGATION
# ===========================================================
menu = st.sidebar.radio(
    "📌 Navigation",
    [
        "Home",
        "Inverse Design (Phase 2)",
        "Multi-Objective Optimization (Phase 3)",
        "Failure Mode Prediction (Phase 5)",
        "Code Checks (Phase 6)",
        "Interpretability (Phase 4)",
        "Database Viewer"
    ]
)


# ===========================================================
# HOME
# ===========================================================
if menu == "Home":
    st.markdown("""
    ### Welcome!  
    Use the sidebar to explore the full pipeline:
    - 🔹 **Inverse Design** → Phase 2  
    - 🔹 **Multi-objective Optimization** → Phase 3  
    - 🔹 **Failure Mode Prediction** → Phase 5  
    - 🔹 **Code Checks** → SCI / EN / AISC  
    - 🔹 **Interpretability** → SHAP + diagnostics  
    - 🔹 **Database Viewer** → Raw dataset  
    """)
    st.success("System ready ❤️")


# ===========================================================
# PHASE 2 — DETERMINISTIC INVERSE DESIGN
# ===========================================================
if menu == "Inverse Design (Phase 2)":

    st.subheader("🎯 Deterministic Inverse Design (Phase 2)")
    result = load_phase2()

    if not result:
        st.error("Phase 2 results not found. Run Phase 2 Python script first.")
    else:
        st.json(result)


# ===========================================================
# PHASE 3 — MULTI-OBJECTIVE RESULT (Best Compromise)
# ===========================================================
if menu == "Multi-Objective Optimization (Phase 3)":

    st.subheader("🌈 NSGA-II Multi-Objective Optimization (Phase 3)")
    data = load_phase3()

    if not data:
        st.error("No Phase 3 optimization outputs found.")
    else:
        st.json(data)
        st.image("ParetoFront.png", caption="Pareto Front")


# ===========================================================
# PHASE 5 — FAILURE MODE PREDICTION
# ===========================================================
if menu == "Failure Mode Prediction (Phase 5)":

    st.subheader("⚠️ Failure-Mode Classifier (Phase 5)")
    clf, encoder, scaler_clf = load_failure_mode_assets()
    feature_cols = joblib.load("feature_cols.joblib")

    st.info("Enter your beam geometry:")

    inputs = {}
    for col in ["H","bf","tw","tf","L","ho","s","fy"]:
        inputs[col] = st.number_input(col, value=300.0)

    df_temp = pd.DataFrame([inputs])
    df_temp = enrich(df_temp)

    X = df_temp[feature_cols]
    X_scaled = scaler_clf.transform(X)

    pred = clf.predict(X_scaled)[0]
    failure_mode = encoder.inverse_transform([pred])[0]

    st.success(f"Predicted Failure Mode: **{failure_mode}**")


# ===========================================================
# PHASE 6 — CODE CHECK
# ===========================================================
if menu == "Code Checks (Phase 6)":

    st.subheader("📘 Code-Based Design Checks (Phase 6)")

    df = load_dataset()
    model, scaler, feature_cols = load_forward_assets()

    st.info("Enter design to check:")

    design = {}
    for col in ["H","bf","tw","tf","L","ho","s","fy"]:
        design[col] = st.number_input(col, value=300.0)

    summary = run_code_check(design, model, scaler, feature_cols, df)
    st.json(summary)


# ===========================================================
# PHASE 4 — INTERPRETABILITY
# (This page just loads saved SHAP figures)
# ===========================================================
if menu == "Interpretability (Phase 4)":
    st.subheader("🔍 Interpretability & Diagnostics (Phase 4)")

    files = [
        "01_ParityPlot.png",
        "02_ErrorHistogram.png",
        "03_ResidualPlot.png",
        "04_InternalFeatureImportance.png",
        "06_SHAP_GlobalBar.png",
        "07_SHAP_Beeswarm.png",
    ]

    for f in files:
        if os.path.exists(f):
            st.image(f)


# ===========================================================
# DATABASE EXPLORER
# ===========================================================
if menu == "Database Viewer":
    st.subheader("📂 FEA Database Viewer")
    df = load_dataset()
    st.dataframe(df)
