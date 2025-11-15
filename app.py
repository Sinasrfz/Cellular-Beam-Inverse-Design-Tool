# ==========================================
# STREAMLIT GUI – Cellular Beam Inverse Design Tool
# ==========================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from io import BytesIO

# ------------------------
# IMPORT OUR UTILS
# ------------------------
from utils.inverse_single import single_objective_inverse, build_features
from utils.inverse_multi import run_nsga2
from utils.shap_utils import compute_shap, plot_shap_summary, plot_shap_bar
from utils.failure_classifier import predict_failure_mode
from utils.code_check import prepare_code_database, lookup_code_resistance
from utils.plotting import (
    plot_response_comparison,
    plot_pareto,
    plot_code_bars
)

# ------------------------
# PAGE SETTINGS
# ------------------------
st.set_page_config(
    page_title="Cellular Beam Inverse Design",
    layout="wide",
)

st.title("🧠 Cellular Beam Inverse Design Tool")
st.write("A complete AI + Engineering environment for optimized cellular beam design.")


# ================================
# LOAD ALL MODELS + DATA
# ================================
@st.cache_resource
def load_all_assets():
    model = joblib.load("ForwardModel_Results/model.joblib")
    scaler = joblib.load("ForwardModel_Results/scaler.joblib")
    feature_cols = joblib.load("ForwardModel_Results/feature_cols.joblib")

    # Failure classifier (optional)
    try:
        clf = joblib.load("FailureMode_Results/failure_mode_classifier.joblib")
        encoder = joblib.load("FailureMode_Results/label_encoder.joblib")
    except:
        clf = None
        encoder = None

    # Code-based database (SCI / EN / AISC)
    df = pd.read_excel("data/21.xlsx")
    db, nbrs = prepare_code_database(df)

    return model, scaler, feature_cols, clf, encoder, db, nbrs, df


model, scaler, feature_cols, clf, encoder, db, nbrs, df_full = load_all_assets()


# ================================
# SIDEBAR SELECTOR
# ================================
task = st.sidebar.selectbox(
    "Choose a task",
    [
        "Single-Objective Inverse Design",
        "Multi-Objective Inverse Design (NSGA-II)",
        "SHAP Model Explainability",
        "Code-Based Resistance Check (SCI/EN/AISC)",
        "Failure Mode Prediction"
    ]
)

# ============================================================
# 1) SINGLE OBJECTIVE INVERSE DESIGN
# ============================================================
if task == "Single-Objective Inverse Design":

    st.header("🎯 Single-Objective Inverse Design")
    wu_target = st.number_input("Enter target ultimate load (kN/m):", min_value=5.0, max_value=80.0, value=30.0)

    if st.button("Run Inverse Design"):

        x_opt, wu_pred = single_objective_inverse(model, scaler, feature_cols, wu_target)

        st.success("Optimization Completed!")

        # Display numerical output
        df_res = pd.DataFrame([x_opt], columns=["H","bf","tw","tf","L","ho","s","fy"])
        df_res["wu_pred"] = wu_pred
        st.write(df_res)

        # Engineering plot
        fig = plt.figure(figsize=(4,3))
        plot_response_comparison(wu_target, wu_pred)
        st.pyplot(fig)


# ============================================================
# 2) MULTI-OBJECTIVE INVERSE DESIGN
# ============================================================
if task == "Multi-Objective Inverse Design (NSGA-II)":

    st.header("⚙️ NSGA-II Multi-Objective Optimization")
    wu_target = st.number_input("Target ultimate load (kN/m):", min_value=5.0, max_value=80.0, value=30.0)

    if st.button("Run NSGA-II"):

        res = run_nsga2(model, scaler, feature_cols, wu_target)

        st.success("NSGA-II Completed!")

        X = res.X
        F = res.F

        # Show best compromise point
        idx = np.argmin(F[:,0] + F[:,1])
        best = X[idx]

        df_best = pd.DataFrame([best], columns=["H","bf","tw","tf","L","ho","s","fy"])
        df_best["wu_pred"] = res.opt.get("wu_pred")[idx]

        st.subheader("Best Compromise Design")
        st.write(df_best)

        # Pareto plot
        fig = plt.figure(figsize=(5,4))
        plot_pareto(F)
        st.pyplot(fig)

        # Option to download full Pareto solutions
        df_pareto = pd.DataFrame(X, columns=["H","bf","tw","tf","L","ho","s","fy"])
        df_pareto["load_error"] = F[:,0]
        df_pareto["area"] = F[:,1]
        st.download_button("Download Pareto Excel", df_pareto.to_csv(index=False), "pareto.csv")


# ============================================================
# 3) SHAP MODEL EXPLAINABILITY
# ============================================================
if task == "SHAP Model Explainability":

    st.header("🔍 SHAP Model Explainability")

    st.write("Computing SHAP values for the test dataset…")

    X_raw = df_full[feature_cols]
    X_scaled = scaler.transform(X_raw)

    shap_values = compute_shap(model, X_scaled, X_raw)

    st.subheader("Global Importance (Bar)")
    fig = plt.figure(figsize=(7,5))
    plot_shap_bar(shap_values, X_raw)
    st.pyplot(fig)

    st.subheader("SHAP Summary Plot")
    fig = plt.figure(figsize=(7,5))
    plot_shap_summary(shap_values, X_raw)
    st.pyplot(fig)


# ============================================================
# 4) CODE CHECK (SCI/EN/AISC)
# ============================================================
if task == "Code-Based Resistance Check (SCI/EN/AISC)":

    st.header("📘 Eurocode / SCI / AISC Code-Based Check")

    st.write("Enter a beam design:")

    H = st.number_input("H", value=300.0)
    bf = st.number_input("bf", value=120.0)
    tw = st.number_input("tw", value=6.0)
    tf = st.number_input("tf", value=10.0)
    L  = st.number_input("L", value=15000.0)
    ho = st.number_input("ho", value=240.0)
    s  = st.number_input("s", value=350.0)
    fy = st.number_input("fy", value=285.0)

    if st.button("Check Code Resistances"):

        x = [H,bf,tw,tf,L,ho,s,fy]
        res = lookup_code_resistance(x, db, nbrs)

        st.success("Code Check Completed!")

        st.write(res)

        fig = plt.figure(figsize=(4,3))
        plot_code_bars(res)
        st.pyplot(fig)


# ============================================================
# 5) FAILURE MODE PREDICTION
# ============================================================
if task == "Failure Mode Prediction":

    st.header("💥 Failure Mode Classifier")

    if clf is None:
        st.error("⚠ No trained failure classifier found.")
    else:
        st.write("Enter your beam design:")

        H = st.number_input("H", value=300.0)
        bf = st.number_input("bf", value=120.0)
        tw = st.number_input("tw", value=6.0)
        tf = st.number_input("tf", value=10.0)
        L  = st.number_input("L", value=15000.0)
        ho = st.number_input("ho", value=240.0)
        s  = st.number_input("s", value=350.0)
        fy = st.number_input("fy", value=285.0)

        x = [H,bf,tw,tf,L,ho,s,fy]
        feats = build_features(x)[feature_cols]
        feats_scaled = scaler.transform(feats)

        if st.button("Predict Failure Mode"):

            mode = predict_failure_mode(clf, encoder, feats_scaled)
            st.success(f"Predicted failure mode: {mode}")
