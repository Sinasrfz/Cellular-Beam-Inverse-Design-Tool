import streamlit as st
import joblib
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(".")

st.title("🎯 Deterministic Inverse Design (Phase 2)")

# =============================
# Load assets
# =============================
model = joblib.load("UnifiedPipeline/Forward/best_model.joblib")
scaler = joblib.load("UnifiedPipeline/Forward/scaler.joblib")
feat_cols = joblib.load("UnifiedPipeline/Forward/feature_cols.joblib")


# ==========================================================
# Build feature vector (EXACTLY like Phase 1)
# ==========================================================
def build_features(x):
    H,bf,tw,tf,L,ho,s,fy = x
    L_H   = L/H
    H_ho  = H/ho
    s_ho  = s/ho
    tf_tw = tf/tw
    bf_H  = bf/H
    Area  = bf*tf + tw*(H - 2*tf)
    fyArea = fy * Area

    df = pd.DataFrame([{
        "H":H, "bf":bf, "tw":tw, "tf":tf,
        "L":L, "ho":ho, "s":s, "fy":fy,
        "L/H":L_H, "H/ho":H_ho, "s/ho":s_ho,
        "tf/tw":tf_tw, "bf/H":bf_H,
        "Area":Area, "fy_Area":fyArea
    }])
    return df[feat_cols]


# =================================================================
# Penalty (exactly like your Phase 2 code)
# =================================================================
def applicability_penalty(x):
    H,bf,tw,tf,L,ho,s,fy = x
    p = 0

    if not (1.25 <= H/ho <= 1.75): p += 5000
    if not (1.08 <= s/ho <= 1.50): p += 5000
    if ho > 0.8*(H + tf):          p += 5000
    if (H - ho)/2 < (tf + 30):     p += 5000
    if not (15 <= L/H <= 30):      p += 5000

    return p


# =================================================================
# Objective
# =================================================================
def objective(x, wu_target):
    feats = build_features(x)
    scaled = scaler.transform(feats)
    wu_pred = model.predict(scaled)[0]
    err = (wu_pred - wu_target)**2
    return err + applicability_penalty(x)


# =================================================================
# Run optimizer (Differential Evolution)
# =================================================================
def run_inverse_design(wu_target):
    from scipy.optimize import differential_evolution

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

    result = differential_evolution(
        func=lambda x: objective(x, wu_target),
        bounds=bounds,
        maxiter=150,
        popsize=20,
        polish=True
    )

    x_opt = result.x
    feats = build_features(x_opt)
    wu_pred = model.predict(scaler.transform(feats))[0]

    return x_opt, feats, wu_pred


# =============================
# Streamlit UI
# =============================
wu_target = st.number_input("Enter Target wu (kN/m):", 5.0, 80.0, 32.0)

if st.button("Run Deterministic Inverse Design"):
    x_opt, feats, wu_pred = run_inverse_design(wu_target)

    err = abs(wu_pred - wu_target)/wu_target*100

    st.subheader("Optimal Design Found ❤️")

    table = feats.copy()
    table["wu_pred"] = wu_pred
    table["wu_target"] = wu_target
    table["error_%"] = err

    st.dataframe(table, use_container_width=True)

    st.success(f"Error = {err:.3f}%  |  Predicted wu = {wu_pred:.3f} kN/m")

    # save
    table.to_excel("UnifiedPipeline/Inverse_Single/last_design.xlsx", index=False)
    st.info("Saved to UnifiedPipeline/Inverse_Single/last_design.xlsx")
