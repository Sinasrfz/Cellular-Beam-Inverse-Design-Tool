# ============================================================
# Cellular Beam Inverse Design Tool – Streamlit GUI
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import plotly.express as px
import plotly.graph_objects as go

from sklearn.neighbors import NearestNeighbors

# ============================================================
# CLEAN COLUMN FUNCTION (same as Phase 1)
# ============================================================

def clean_column(col):
    col = str(col)
    col = col.replace("\n", " ")
    col = col.replace('"', "")
    col = re.sub(r"\s+", "_", col).strip()
    return col


# ============================================================
# LOAD ALL MODELS & DATA (cached)
# ============================================================

@st.cache_resource
def load_all_assets():

    # 1) Load forward model + scaler + feature_cols
    model = joblib.load("ForwardModel_Results/model.joblib")
    scaler = joblib.load("ForwardModel_Results/scaler.joblib")
    feature_cols = joblib.load("ForwardModel_Results/feature_cols.joblib")

    # 2) Failure mode classifier (optional)
    try:
        clf = joblib.load("FailureMode_Results/failure_mode_classifier.joblib")
        encoder = joblib.load("FailureMode_Results/label_encoder.joblib")
    except:
        clf = None
        encoder = None

    # 3) Load code-check Excel
    df = pd.read_excel("data/21.xlsx")
    df.columns = [clean_column(c) for c in df.columns]

    # Rename for consistency
    rename_map = {
        "fy×Area": "fy_Area",
        "Applicable?": "Applicable_SCI",
        "wSCI_(kN/m)": "wSCI",
        "wEN,M_(kN/m)": "wENM",
        "wEN,A_(kN/m)": "wENA",
        "wAISC_(kN/m)": "wAISC",
        "wu,FEA/wSCI": "wu_over_SCI",
        "wu,FEA/wEN,M": "wu_over_ENM",
        "wu,FEA/wEN,A": "wu_over_ENA",
        "wu,FEA/wAISC": "wu_over_AISC",
        "Failure_mode": "FailureMode_SCI",
    }

    df.rename(columns=rename_map, inplace=True)

    # Database used for code-checking
    core_cols = ["H","bf","tw","tf","L","ho","s","fy"]
    code_cols = ["wSCI","wENM","wENA","wAISC","FailureMode_SCI"]

    db = df[core_cols + code_cols].copy()
    nbrs = NearestNeighbors(n_neighbors=1)
    nbrs.fit(db[core_cols])

    return model, scaler, feature_cols, clf, encoder, db, nbrs, df


model, scaler, feature_cols, clf, encoder, db, nbrs, df_full = load_all_assets()


# ============================================================
# UTILITY – Build ML input feature row
# ============================================================

def compute_features(H,bf,tw,tf,L,ho,s,fy):

    Area = bf*tf + (H - 2*tf)*tw
    fyA = fy * Area

    return pd.DataFrame([{
        "H":H, "bf":bf, "tw":tw, "tf":tf, "L":L, "ho":ho, "s":s, "fy":fy,
        "L/H": L/H,
        "H/ho": H/ho,
        "s/ho": s/ho,
        "tf/tw": tf/tw,
        "bf/H": bf/H,
        "Area": Area,
        "fy_Area": fyA
    }])[feature_cols]


# ============================================================
# CODE CHECK – nearest neighbor lookup
# ============================================================

def evaluate_code_resistance(x):
    cols = ["H","bf","tw","tf","L","ho","s","fy"]

    _, idx = nbrs.kneighbors([x])
    row = db.iloc[idx[0,0]]

    return {
        "SCI": row["wSCI"],
        "ENM": row["wENM"],
        "ENA": row["wENA"],
        "AISC": row["wAISC"],
        "FailureMode_SCI": row["FailureMode_SCI"]
    }


# ============================================================
# STREAMLIT LAYOUT
# ============================================================

st.set_page_config(page_title="Cellular Beam Inverse Design Tool", layout="wide")
st.title("🧬 Cellular Beam Inverse Design Tool")
st.write("Predict → Optimize → Check → Explain")


# ============================================================
# SIDEBAR – User Input
# ============================================================

st.sidebar.header("Beam Geometry Inputs")

H  = st.sidebar.number_input("H (mm)", 100, 1000, 300)
bf = st.sidebar.number_input("bf (mm)", 50, 400, 120)
tw = st.sidebar.number_input("tw (mm)", 4, 30, 6)
tf = st.sidebar.number_input("tf (mm)", 4, 30, 10)
L  = st.sidebar.number_input("L (mm)", 2000, 30000, 15000)
ho = st.sidebar.number_input("Opening diameter ho (mm)", 50, 700, 240)
s  = st.sidebar.number_input("Opening spacing s (mm)", 100, 1000, 350)
fy = st.sidebar.number_input("Steel yield strength fy (MPa)", 200, 500, 275)

st.sidebar.header("Load Target")
w_target = st.sidebar.number_input("Target distributed load wu (kN/m)", 5.0, 100.0, 32.0)


# ============================================================
# PHASE 1 – Predict wu
# ============================================================

st.subheader("📌 Phase 1 – Forward Model Prediction")

feat = compute_features(H,bf,tw,tf,L,ho,s,fy)
feat_scaled = scaler.transform(feat)
wu_pred = model.predict(feat_scaled)[0]

st.success(f"Predicted ultimate load = **{wu_pred:.3f} kN/m**")


# ============================================================
# PHASE 2 – Error
# ============================================================

st.subheader("📌 Phase 2 – Error compared to target")

err = abs(wu_pred - w_target)/w_target * 100
st.info(f"Relative error = **{err:.2f} %**")


# ============================================================
# PHASE 3 – Code checks
# ============================================================

st.subheader("📌 Phase 3 – Code Checks (SCI / EN / AISC)")

x_vec = [H,bf,tw,tf,L,ho,s,fy]
codes = evaluate_code_resistance(x_vec)

col1, col2 = st.columns(2)
col1.metric("SCI resistance",  f"{codes['SCI']:.2f} kN/m")
col1.metric("EN 1993-1-13 (Main)", f"{codes['ENM']:.2f} kN/m")
col1.metric("EN 1993-1-13 (Alt)",  f"{codes['ENA']:.2f} kN/m")
col2.metric("AISC DG31",           f"{codes['AISC']:.2f} kN/m")
col2.metric("Failure mode (SCI)",  codes['FailureMode_SCI'])


# ============================================================
# PHASE 4 – Failure mode prediction (optional)
# ============================================================

st.subheader("📌 Phase 4 – ML Failure Mode Classifier")

if clf is None:
    st.warning("Classifier not available in current repo.")
else:
    y_pred = clf.predict(feat_scaled)
    fm = encoder.inverse_transform(y_pred)[0]
    st.info(f"Predicted failure mode = **{fm}**")


# ============================================================
# PHASE 5 – Plot comparison
# ============================================================

st.subheader("📌 Plot: Code Resistances vs Predicted wu")

fig = go.Figure()
fig.add_trace(go.Bar(name="Code", x=["SCI","ENM","ENA","AISC"],
                     y=[codes["SCI"], codes["ENM"], codes["ENA"], codes["AISC"]]))
fig.add_hline(y=wu_pred, line_color="red", annotation_text="Predicted wu")
fig.add_hline(y=w_target, line_color="green", annotation_text="Target wu")
fig.update_layout(height=400)
st.plotly_chart(fig, use_container_width=True)


# ============================================================
# END
# ============================================================
st.caption("Developed with ❤️ by Sina & ChatGPT")
