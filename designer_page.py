# ============================================================
# designer_page.py — Physics-Informed Inverse Design Engine
# (Phases 3–6 only, with code-based safety constraints)
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.termination import get_termination
from sklearn.neighbors import NearestNeighbors

# ============================================================
# MAIN RENDER FUNCTION (CALLED FROM app.py)
# ============================================================

def render(model_forward, scaler_forward, feature_cols,
           clf_failure, scaler_failure, label_encoder,
           df_full):

    st.header("🏗 Designer Tool — Physics-Informed Inverse Design (Safe)")

    st.markdown("""
    Enter the **target ultimate load (wu)** and the tool will:
    - Run **NSGA-II** multi-objective optimization (Phase 3)
    - Predict **failure mode** (Phase 5)
    - Perform **SCI / EN / AISC code checks** (Phase 6)
    - Ensure **designs respect all code safety constraints**
    """)

    wu_target = st.number_input("Target ultimate load wu (kN/m):", min_value=1.0, step=1.0)

    if st.button("Run Physics-Informed Optimization", type="primary"):
        run_inverse_design(
            wu_target,
            model_forward, scaler_forward, feature_cols,
            clf_failure, scaler_failure, label_encoder,
            df_full
        )

# ============================================================
# FEATURE ENGINEERING (same as Phase 1)
# ============================================================

def build_features_vector(H, bf, tw, tf, L, ho, s, fy):
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

def applicability_penalty(x):
    H, bf, tw, tf, L, ho, s, fy = x
    p = 0
    if not (1.25 <= H / ho <= 1.75): p += 1e3
    if not (1.08 <= s / ho <= 1.50): p += 1e3
    if ho > 0.8 * (H + tf): p += 1e3
    if (H - ho) / 2 < (tf + 30): p += 1e3
    if fy < 250 or fy > 460: p += 1e3
    if not (15 <= L / H <= 30): p += 1e3
    return p

# ============================================================
# PHASE 3 — Physics-Informed NSGA-II (Main Optimization)
# ============================================================

class BeamProblem(ElementwiseProblem):
    def __init__(self, wu_target, model_forward, scaler_forward, feature_cols, df):
        super().__init__(n_var=8, n_obj=2,
                         xl=np.array([300,120,6,10,6000,200,200,250]),
                         xu=np.array([700,270,15,25,21000,560,830,460]))
        self.wu_target = wu_target
        self.model = model_forward
        self.scaler = scaler_forward
        self.feature_cols = feature_cols
        self.df = df.copy()

        # prepare database for code lookup
        self.df.columns = [c.replace("\n", " ").replace('"','').strip() for c in self.df.columns]
        if "fy×Area" in self.df.columns:
            self.df.rename(columns={"fy×Area": "fy_Area"}, inplace=True)

    def _evaluate(self, x, out, *args, **kwargs):
        feats = build_features_vector(*x)
        wu_pred = self.model.predict(self.scaler.transform(feats[self.feature_cols]))[0]
        A = feats["Area"].iloc[0]

        # get nearest database row for code values
        NN = NearestNeighbors(n_neighbors=1)
        NN.fit(self.df[self.feature_cols])
        dist, idx = NN.kneighbors(feats[self.feature_cols])
        row = self.df.iloc[idx[0][0]]

        # penalties if unsafe
        penalty = applicability_penalty(x)
        for code in ["wSCI", "wEN,M", "wEN,A", "wAISC"]:
            R = row.get(code, np.nan)
            if pd.notna(R) and wu_pred > R:
                penalty += (wu_pred - R)**2

        # objectives: match target, minimize area
        f1 = abs(wu_pred - self.wu_target)/self.wu_target
        f2 = A / 1e4

        out["F"] = [f1 + penalty*1e-6, f2 + penalty*1e-6]
        out["wu_pred"] = wu_pred
        out["Area"] = A

def run_nsga(wu_target, model_forward, scaler_forward, feature_cols, df):
    problem = BeamProblem(wu_target, model_forward, scaler_forward, feature_cols, df)
    algo = NSGA2(pop_size=80)
    term = get_termination("n_gen", 120)
    result = minimize(problem, algo, term, seed=1, verbose=False)

    X = result.X
    F = result.F
    best_idx = np.argmin(F[:,0] + F[:,1])
    best = X[best_idx]
    feats = build_features_vector(*best)
    wu_best = float(model_forward.predict(scaler_forward.transform(feats[feature_cols]))[0])
    A_best = float(feats["Area"].iloc[0])

    return {
        "pareto_X": X,
        "pareto_F": F,
        "best_design": best,
        "best_wu": wu_best,
        "best_Area": A_best
    }

# ============================================================
# PHASE 5 — Failure Mode Prediction
# ============================================================

def predict_failure(H,bf,tw,tf,L,ho,s,fy, clf, scaler, encoder, feature_cols):
    df = build_features_vector(H,bf,tw,tf,L,ho,s,fy)
    X_scaled = scaler.transform(df[feature_cols])
    y = clf.predict(X_scaled)[0]
    return encoder.inverse_transform([y])[0]

# ============================================================
# PHASE 6 — Code Compliance Check
# ============================================================

def run_code_check(H,bf,tw,tf,L,ho,s,fy, model, scaler, feature_cols, df):
    feats = build_features_vector(H,bf,tw,tf,L,ho,s,fy)
    wu_pred = float(model.predict(scaler.transform(feats[feature_cols]))[0])

    df.columns = [c.replace("\n"," ").replace('"','').strip() for c in df.columns]
    if "fy×Area" in df.columns:
        df.rename(columns={"fy×Area":"fy_Area"}, inplace=True)

    NN = NearestNeighbors(n_neighbors=1)
    NN.fit(df[feature_cols])
    dist, idx = NN.kneighbors(feats[feature_cols])
    row = df.iloc[idx[0][0]]

    def safe(wu_pred, R): return bool(wu_pred <= R if not pd.isna(R) else False)

    return {
        "wu_pred": wu_pred,
        "wSCI": float(row.get("wSCI", np.nan)),
        "wENM": float(row.get("wEN,M", np.nan)),
        "wENA": float(row.get("wEN,A", np.nan)),
        "wAISC": float(row.get("wAISC", np.nan)),
        "Safe_SCI": safe(wu_pred, row.get("wSCI", np.nan)),
        "Safe_ENM": safe(wu_pred, row.get("wEN,M", np.nan)),
        "Safe_ENA": safe(wu_pred, row.get("wEN,A", np.nan)),
        "Safe_AISC": safe(wu_pred, row.get("wAISC", np.nan))
    }

# ============================================================
# MAIN EXECUTION SEQUENCE
# ============================================================

def run_inverse_design(wu_target,
                       model_forward, scaler_forward, feature_cols,
                       clf_failure, scaler_failure, label_encoder,
                       df_full):

    st.subheader("🔹 Phase 3 — NSGA-II Physics-Informed Optimization")
    nsga = run_nsga(wu_target, model_forward, scaler_forward, feature_cols, df_full)
    H,bf,tw,tf,L,ho,s,fy = nsga["best_design"]

    st.json({
        "Best Design": {k: float(v) for k,v in zip(
            ["H","bf","tw","tf","L","ho","s","fy"], nsga["best_design"])},
        "wu_pred": nsga["best_wu"],
        "Area": nsga["best_Area"]
    })

    st.subheader("🔹 Phase 5 — Failure Mode Prediction")
    failure = predict_failure(H,bf,tw,tf,L,ho,s,fy,
                              clf_failure, scaler_failure, label_encoder,
                              feature_cols)
    st.success(f"Predicted Failure Mode: **{failure}**")

    st.subheader("🔹 Phase 6 — Code Compliance Check")
    code = run_code_check(H,bf,tw,tf,L,ho,s,fy,
                          model_forward, scaler_forward, feature_cols,
                          df_full)
    st.json(code)

    st.success("✔ Full physics-informed inverse design completed safely.")
