import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Cellular Beam Inverse Design Tool",
    layout="wide"
)

st.title("🧠 Cellular Beam Inverse Design Tool")
st.write("This tool uses the trained ML models and inverse design pipeline to recommend optimal cellular beam geometry for a target ultimate load (wu).")
st.write("❤️ Version A (Practical) — Loads trained models, no training inside GUI.")

# =========================================================
# LOAD ML ASSETS
# =========================================================
@st.cache_resource
def load_forward_assets():
    model = joblib.load("best_model.joblib")
    scaler = joblib.load("scaler.joblib")
    feature_cols = joblib.load("feature_cols.joblib")
    return model, scaler, feature_cols

@st.cache_resource
def load_failure_assets():
    clf = joblib.load("best_failure_classifier.joblib")
    encoder = joblib.load("label_encoder.joblib")
    return clf, encoder

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

# ==================================================================================
# PAGE SELECTION
# ==================================================================================
pages = ["Home", "Inverse Design", "Code Check"]
choice = st.sidebar.selectbox("📌 Select Module", pages)

model, scaler, feature_cols = load_forward_assets()

# Pixel-perfect ranges (same as training)
var_names = ["H","bf","tw","tf","L","ho","s","fy"]
bounds = {
    "H":  (300,700),
    "bf": (120,270),
    "tw": (6,15),
    "tf": (10,25),
    "L":  (6000,21000),
    "ho": (200,560),
    "s":  (200,830),
    "fy": (250,460)
}

# ==================================================================================
# PAGE 1 — HOME
# ==================================================================================
if choice == "Home":
    st.subheader("Welcome ❤️")
    st.write("""
    This GUI provides:
    - Deterministic inverse design (Phase 2)
    - Code-based design check (Phase 6)
    - Failure mode prediction (Phase 5)
    - Multi-objective results if uploaded (Phase 3)

    👉 Use the sidebar to navigate.
    """)

# ==================================================================================
# PAGE 2 — INVERSE DESIGN (Phase 2)
# ==================================================================================
if choice == "Inverse Design":
    st.subheader("🎯 Deterministic Inverse Design (Phase 2)")

    wu_target = st.number_input("Enter target ultimate load (kN/m)", min_value=5.0, max_value=80.0, value=30.0)

    if st.button("Run Inverse Design"):
        # Check if Phase-2 result file exists (otherwise warn)
        if os.path.exists("UnifiedPipeline/InverseDeterministic/result.json"):
            with open("UnifiedPipeline/InverseDeterministic/result.json") as f:
                result = json.load(f)
        else:
            st.error("❌ Phase-2 results not found.\nPlease run Phase-2 on your laptop and upload result.json.")
            st.stop()

        st.success("Inverse design result loaded from saved Phase-2 output.")
        st.json(result)

# ==================================================================================
# PAGE 3 — CODE CHECK (Phase 6)
# ==================================================================================
if choice == "Code Check":
    st.subheader("📘 Code-Based Check (SCI • EN • AISC)")

    # Designer enters geometry manually or passes from Phase 2 output
    H  = st.number_input("H",  min_value=300, max_value=700, value=500)
    bf = st.number_input("bf", min_value=120, max_value=270, value=200)
    tw = st.number_input("tw", min_value=6,   max_value=15,  value=10)
    tf = st.number_input("tf", min_value=10,  max_value=25,  value=15)
    L  = st.number_input("L",  min_value=6000,max_value=21000,value=12000)
    ho = st.number_input("ho", min_value=200, max_value=560, value=350)
    s  = st.number_input("s",  min_value=200, max_value=830, value=600)
    fy = st.number_input("fy", min_value=250, max_value=460, value=355)

    if st.button("Run Code Check"):
        # Load original dataset
        if not os.path.exists("21.xlsx"):
            st.error("❌ Missing '21.xlsx' (your FEA dataset). Upload it to GitHub root.")
            st.stop()

        df = pd.read_excel("21.xlsx")
        df = enrich(df)

        from sklearn.neighbors import NearestNeighbors
        NN = NearestNeighbors(n_neighbors=1)
        NN.fit(df[feature_cols])

        # Build ML features
        df_temp = pd.DataFrame([{
            "H":H,"bf":bf,"tw":tw,"tf":tf,"L":L,"ho":ho,"s":s,"fy":fy
        }])
        df_temp = enrich(df_temp)
        feats = df_temp[feature_cols]

        scaled = scaler.transform(feats)
        wu_pred = float(model.predict(scaled)[0])

        dist, idx = NN.kneighbors(feats[feature_cols])
        match = df.iloc[idx[0][0]]

        SCI  = match.get("wSCI", None)
        ENM  = match.get("wENM", None)
        ENA  = match.get("wENA", None)
        AISC = match.get("wAISC", None)

        result = {
            "ML_predicted_wu": wu_pred,
            "SCI":  SCI,
            "ENM":  ENM,
            "ENA":  ENA,
            "AISC": AISC
        }

        st.write("### 🔍 Result:")
        st.json(result)
