import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from scipy.optimize import differential_evolution
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

# ============================================================
# LOAD ALL ASSETS (Phase 1 + Phase 5)
# ============================================================

best_model = joblib.load("best_model.joblib")
scaler = joblib.load("scaler.joblib")
feature_cols = joblib.load("feature_cols.joblib")

failure_clf = joblib.load("best_failure_classifier.joblib")
label_encoder = joblib.load("label_encoder.joblib")

# Load FEA engineering database
df_raw = pd.read_excel("21.xlsx")

# ============================================================
# CLEAN + FEATURE ENGINEERING (same rules as phases 1–6)
# ============================================================

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

df_raw = df_raw.rename(columns=lambda x: str(x).replace("\n"," ").replace('"',"").replace(" ","_"))
df_raw = df_raw.rename(columns={"fy×Area":"fy_Area"})
df_raw = enrich(df_raw)

# Build NN database for Phase-6 code checks
NN = NearestNeighbors(n_neighbors=1)
NN.fit(df_raw[feature_cols])


# ============================================================
# PHASE 2 — DETERMINISTIC INVERSE DESIGN (LIVE INSIDE GUI)
# ============================================================

var_names = ["H","bf","tw","tf","L","ho","s","fy"]

bounds = np.array([
    [300,700],
    [120,270],
    [6,15],
    [10,25],
    [6000,21000],
    [200,560],
    [200,830],
    [250,460]
])

def build_features_from_x(x):
    H,bf,tw,tf,L,ho,s,fy = x
    df = pd.DataFrame([{
        "H":H,"bf":bf,"tw":tw,"tf":tf,
        "L":L,"ho":ho,"s":s,"fy":fy
    }])
    df = enrich(df)
    return df[feature_cols]

def applicability_penalty(x):
    H,bf,tw,tf,L,ho,s,fy = x
    p=0
    if not (1.25 <= H/ho <= 1.75): p+=5000
    if not (1.08 <= s/ho <= 1.50): p+=5000
    if ho > 0.8*(H + tf): p+=5000
    if (H - ho)/2 < (tf + 30): p+=5000
    if not (15 <= L/H <= 30): p+=5000
    if fy < 250 or fy > 460: p+=5000
    return p

def objective(x, wu_target):
    feats = build_features_from_x(x)
    wu_pred = best_model.predict(scaler.transform(feats))[0]
    return (wu_pred - wu_target)**2 + applicability_penalty(x)

def run_inverse_design(wu_target):
    result = differential_evolution(
        func=lambda x: objective(x, wu_target),
        bounds=bounds,
        popsize=20,
        maxiter=120,
        tol=1e-6,
        seed=42,
        polish=True
    )
    x_opt = result.x
    feats = build_features_from_x(x_opt)
    wu_pred = best_model.predict(scaler.transform(feats))[0]
    err_pct = abs(wu_pred - wu_target)/wu_target*100

    return {
        "Design": dict(zip(var_names, x_opt)),
        "wu_pred": float(wu_pred),
        "error_pct": float(err_pct),
        "Area": float(feats["Area"].iloc[0])
    }

# ============================================================
# PHASE 5 — FAILURE MODE PREDICTION
# ============================================================

def predict_failure_mode(des):
    df = pd.DataFrame([des])
    df = enrich(df)
    X = df[feature_cols]
    Xs = MinMaxScaler().fit_transform(X)  # local scaler
    y = failure_clf.predict(Xs)[0]
    return label_encoder.inverse_transform([y])[0]

# ============================================================
# PHASE 6 — CODE CHECK (SCI, EN, AISC)
# ============================================================

def code_check(des):
    df = pd.DataFrame([des])
    df = enrich(df)
    feats = df[feature_cols]

    wu_pred = float(best_model.predict(scaler.transform(feats))[0])
    dist, idx = NN.kneighbors(feats)
    db_row = df_raw.iloc[idx[0][0]]

    codes = {}
    for code in ["wSCI","wENM","wENA","wAISC"]:
        R = float(db_row[code]) if code in db_row else None
        if R is None or R<=0:
            codes[code] = {"R": None, "Safe": None}
        else:
            ratio = wu_pred / R
            codes[code] = {"R":R, "Safe": ratio<=1, "wu/R": ratio}

    return wu_pred, codes

# ============================================================
# STREAMLIT GUI
# ============================================================

st.title("🧠 Cellular Beam Inverse Design Tool")
st.caption("Fully automated inverse–design + code checks + failure mode prediction ❤️")

st.sidebar.header("⚙️ Designer Input")
wu = st.sidebar.number_input("Required Ultimate Load wu (kN/m)", min_value=5.0, max_value=80.0, value=30.0)

if st.sidebar.button("Run Inverse Design"):
    st.subheader("🔎 Phase 2 — Deterministic Inverse Design")
    result = run_inverse_design(wu)

    st.success("Inverse design completed!")

    st.write("### Suggested Section")
    st.json(result["Design"])

    st.metric("Predicted wu", f"{result['wu_pred']:.2f} kN/m")
    st.metric("Error (%)", f"{result['error_pct']:.2f}%")
    st.metric("Area", f"{result['Area']:.1f} mm²")

    # Phase 5
    st.subheader("🧩 Phase 5 — Failure Mode Prediction")
    fm = predict_failure_mode(result["Design"])
    st.write(f"**Predicted failure mode:** `{fm}`")

    # Phase 6
    st.subheader("🏛️ Phase 6 — Code Compliance Check")
    wu_pred, codes = code_check(result["Design"])
    st.write(f"**ML predicted ultimate load:** {wu_pred:.2f} kN/m")

    st.write("### Code Results")
    st.json(codes)

st.markdown("---")
st.info("This tool automatically runs: Phase-2 → Phase-5 → Phase-6\nNo need to run Python scripts on your laptop.")
