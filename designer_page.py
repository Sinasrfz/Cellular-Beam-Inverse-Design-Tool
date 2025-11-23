# ============================================================
# designer_page.py — Practical Inverse Design Engine (Phases 2–6)
# Refined for realistic engineering use:
#   - Fixed: L, ho, s, fy (designer inputs)
#   - Optimized: H, bf, tw, tf
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.termination import get_termination
from sklearn.neighbors import NearestNeighbors


# ============================================================
# MAIN PAGE
# ============================================================

def render(model_forward, scaler_forward, feature_cols,
           clf_failure, scaler_failure, label_encoder,
           df_full):

    st.header("🏗 Designer Tool — Practical Inverse Design")

    st.markdown("""
    This module assists engineers in selecting the **optimal cellular beam geometry**  
    for a required ultimate load (**wu_target**), given fixed project parameters.

    **You specify:**  
    - Beam span (**L**)  
    - Hole diameter (**ho**)  
    - Hole spacing (**s**)  
    - Steel grade (**fy**)  
    - Target ultimate load (**wu_target**)  

    The tool optimizes only **H**, **bf**, **tw**, and **tf**,  
    predicts the **failure mode**, and checks **code compliance** (SCI, EN, AISC).
    """)

    # -----------------------------
    # USER INPUTS
    # -----------------------------
    st.sidebar.header("🧮 Design Inputs")

    L = st.sidebar.number_input("Beam Span L (mm)", value=12000.0, step=500.0)
    ho = st.sidebar.number_input("Opening Diameter ho (mm)", value=400.0, step=10.0)
    s = st.sidebar.number_input("Opening Spacing s (mm)", value=600.0, step=10.0)
    fy = st.sidebar.number_input("Steel Yield Strength fy (MPa)", value=355.0, step=10.0)

    wu_target = st.number_input("🎯 Target Ultimate Load wu (kN/m)", min_value=5.0, max_value=200.0, value=30.0, step=1.0)

    if st.button("Run Inverse Design", type="primary"):
        run_inverse_design(wu_target, L, ho, s, fy,
                           model_forward, scaler_forward, feature_cols,
                           clf_failure, scaler_failure, label_encoder,
                           df_full)


# ============================================================
# FEATURE ENGINEERING (same as Phase 1)
# ============================================================

def build_features_vector(H, bf, tw, tf, L, ho, s, fy):
    """Constructs the full feature vector (with derived terms)."""
    df = pd.DataFrame([{
        "H": H, "bf": bf, "tw": tw, "tf": tf,
        "L": L, "ho": ho, "s": s, "fy": fy,
        "L/H": L / H,
        "H/ho": H / ho,
        "s/ho": s / ho,
        "tf/tw": tf / tw,
        "bf/H": bf / H,
        "Area": bf * tf + tw * (H - 2 * tf),
        "fy_Area": fy * (bf * tf + tw * (H - 2 * tf))
    }])
    return df


def applicability_penalty(H, bf, tw, tf, L, ho, s, fy):
    """Penalizes non-compliant geometries according to Eurocode and SCI rules."""
    p = 0
    if not (1.25 <= H / ho <= 1.75): p += 5000
    if not (1.08 <= s / ho <= 1.50): p += 5000
    if ho > 0.8 * (H + tf): p += 5000
    if (H - ho) / 2 < (tf + 30): p += 5000
    if not (15 <= L / H <= 30): p += 5000
    if not (250 <= fy <= 460): p += 5000
    return p


# ============================================================
# PHASE 2 — Deterministic Inverse Design (DE)
# ============================================================

def deterministic_inverse(wu_target, L, ho, s, fy, model, scaler, feature_cols):
    bounds = np.array([
        [300, 700],  # H
        [120, 270],  # bf
        [6, 15],     # tw
        [10, 25]     # tf
    ])

    def objective(x):
        H, bf, tw, tf = x
        feats = build_features_vector(H, bf, tw, tf, L, ho, s, fy)
        wu_pred = model.predict(scaler.transform(feats[feature_cols]))[0]
        err = (wu_pred - wu_target) ** 2
        pen = applicability_penalty(H, bf, tw, tf, L, ho, s, fy)
        return err + pen

    result = differential_evolution(
        objective, bounds=bounds,
        popsize=20, maxiter=200, tol=1e-6, seed=42, polish=True
    )

    H, bf, tw, tf = result.x
    feats = build_features_vector(H, bf, tw, tf, L, ho, s, fy)
    wu_pred = float(model.predict(scaler.transform(feats[feature_cols]))[0])
    Area = float(feats["Area"].iloc[0])
    err_pct = abs(wu_pred - wu_target) / wu_target * 100

    return {
        "design": [H, bf, tw, tf],
        "wu_pred": wu_pred,
        "Area": Area,
        "error_pct": err_pct
    }


# ============================================================
# PHASE 3 — Multi-Objective (NSGA-II)
# ============================================================

class BeamProblem(ElementwiseProblem):
    def __init__(self, wu_target, L, ho, s, fy, model, scaler, feature_cols):
        super().__init__(n_var=4, n_obj=2,
                         xl=np.array([300, 120, 6, 10]),
                         xu=np.array([700, 270, 15, 25]))
        self.wu_target = wu_target
        self.L, self.ho, self.s, self.fy = L, ho, s, fy
        self.model, self.scaler, self.feature_cols = model, scaler, feature_cols

    def _evaluate(self, x, out, *args, **kwargs):
        H, bf, tw, tf = x
        feats = build_features_vector(H, bf, tw, tf, self.L, self.ho, self.s, self.fy)
        wu_pred = self.model.predict(self.scaler.transform(feats[self.feature_cols]))[0]
        A = feats["Area"].iloc[0]

        f1 = abs(wu_pred - self.wu_target) / self.wu_target
        f2 = A / 1e4
        pen = applicability_penalty(H, bf, tw, tf, self.L, self.ho, self.s, self.fy)

        out["F"] = [f1 + pen * 1e-6, f2 + pen * 1e-6]


def run_nsga(wu_target, L, ho, s, fy, model, scaler, feature_cols):
    problem = BeamProblem(wu_target, L, ho, s, fy, model, scaler, feature_cols)
    algo = NSGA2(pop_size=80)
    term = get_termination("n_gen", 150)
    result = minimize(problem, algo, term, seed=42, verbose=False)

    X, F = result.X, result.F
    best_idx = np.argmin(F[:, 0] + F[:, 1])
    best = X[best_idx]

    feats = build_features_vector(*best, L, ho, s, fy)
    wu_best = float(model.predict(scaler.transform(feats[feature_cols]))[0])
    A_best = float(feats["Area"].iloc[0])

    return {"best_design": best, "wu_best": wu_best, "A_best": A_best, "pareto_F": F}


# ============================================================
# PHASE 5 — Failure Mode Prediction
# ============================================================

def predict_failure(H, bf, tw, tf, L, ho, s, fy, clf, scaler, encoder, feature_cols):
    df = build_features_vector(H, bf, tw, tf, L, ho, s, fy)
    X_scaled = scaler.transform(df[feature_cols])
    y = clf.predict(X_scaled)[0]
    return encoder.inverse_transform([y])[0]


# ============================================================
# PHASE 6 — Code Compliance Check
# ============================================================

# ============================================================
# PHASE 6 — Code Compliance Check (robust version)
# ============================================================

def run_code_check(H, bf, tw, tf, L, ho, s, fy, model, scaler, feature_cols, df):
    """Robust Phase-6 code-check using tolerant column matching."""
    feats = build_features_vector(H, bf, tw, tf, L, ho, s, fy)
    wu_pred = float(model.predict(scaler.transform(feats[feature_cols]))[0])

    # Temporarily clean column headers in df
    df_cols = {c: str(c).replace("\n", " ").replace('"', '').strip() for c in df.columns}
    df = df.rename(columns=df_cols)

    # Flexible search patterns for resistance columns
    def find_col(keyword):
        for c in df.columns:
            if keyword.lower() in c.lower():
                return c
        return None

    col_sci  = find_col("wSCI")
    col_enm  = find_col("wEN,M")
    col_ena  = find_col("wEN,A")
    col_aisc = find_col("wAISC")

    # Safely fit nearest neighbor model
    existing_cols = [c for c in feature_cols if c in df.columns]
    if not existing_cols:
        st.error("⚠ Feature columns not found in dataset.")
        return {}

    from sklearn.neighbors import NearestNeighbors
    NN = NearestNeighbors(n_neighbors=1)
    NN.fit(df[existing_cols])
    dist, idx = NN.kneighbors(feats[existing_cols])
    row = df.iloc[idx[0][0]]

    def safe(pred, val):
        return bool(pred <= val) if pd.notna(val) else False

    return {
        "wu_pred": wu_pred,
        "wSCI":  float(row.get(col_sci,  np.nan)),
        "wENM":  float(row.get(col_enm,  np.nan)),
        "wENA":  float(row.get(col_ena,  np.nan)),
        "wAISC": float(row.get(col_aisc, np.nan)),
        "Safe_SCI":  safe(wu_pred, row.get(col_sci,  np.nan)),
        "Safe_ENM":  safe(wu_pred, row.get(col_enm,  np.nan)),
        "Safe_ENA":  safe(wu_pred, row.get(col_ena,  np.nan)),
        "Safe_AISC": safe(wu_pred, row.get(col_aisc, np.nan)),
    }

# ============================================================
# MAIN EXECUTION PIPELINE
# ============================================================

def run_inverse_design(wu_target, L, ho, s, fy,
                       model_forward, scaler_forward, feature_cols,
                       clf_failure, scaler_failure, label_encoder,
                       df_full):

    st.subheader("🔹 Phase 2 — Deterministic Inverse Design")
    det = deterministic_inverse(wu_target, L, ho, s, fy, model_forward, scaler_forward, feature_cols)
    H, bf, tw, tf = det["design"]

    st.json({
        "Design": {"H": H, "bf": bf, "tw": tw, "tf": tf},
        "wu_pred": det["wu_pred"],
        "Area": det["Area"],
        "Error_%": det["error_pct"]
    })

    st.subheader("🔹 Phase 3 — NSGA-II Multi-Objective Optimization")
    nsga = run_nsga(wu_target, L, ho, s, fy, model_forward, scaler_forward, feature_cols)
    st.json({
        "Best Design": {k: float(v) for k, v in zip(["H", "bf", "tw", "tf"], nsga["best_design"])},
        "wu_pred": nsga["wu_best"],
        "Area": nsga["A_best"]
    })

    st.subheader("🔹 Phase 5 — Failure Mode Prediction")
    failure = predict_failure(H, bf, tw, tf, L, ho, s, fy,
                              clf_failure, scaler_failure, label_encoder, feature_cols)
    st.success(f"Predicted Failure Mode: **{failure}**")

    st.subheader("🔹 Phase 6 — Code Compliance Check")
    code = run_code_check(H, bf, tw, tf, L, ho, s, fy,
                          model_forward, scaler_forward, feature_cols, df_full)
    st.json(code)

    st.success("✔ Full inverse design completed successfully.")

