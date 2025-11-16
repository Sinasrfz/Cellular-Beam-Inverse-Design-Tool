# ============================================================
# designer_page.py — Full Inverse Design Engine (Phases 2–6)
# Root-based version (consistent with app.py)
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
# MAIN RENDER FUNCTION (CALLED FROM app.py)
# ============================================================

def render(model_forward, scaler_forward, feature_cols,
           clf_failure, scaler_failure, label_encoder,
           df_full):

    st.header("🏗 Designer Tool — Inverse Design Engine")

    st.markdown("""
    Enter the **target ultimate load (wu)** and the tool will:
    - Run **Phase 2** deterministic inverse design  
    - Run **Phase 3** NSGA-II multi-objective optimization  
    - Predict **failure mode** (Phase 5)  
    - Perform **SCI / EN / AISC code checks** (Phase 6)  
    """)

    wu_target = st.number_input("Target ultimate load wu (kN/m):", min_value=1.0, step=1.0)

    if st.button("Run Inverse Design", type="primary"):
        run_inverse_design(
            wu_target,
            model_forward, scaler_forward, feature_cols,
            clf_failure, scaler_failure, label_encoder,
            df_full
        )


# ============================================================
# FEATURE ENGINEERING (Phase 1 rules)
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
    if not (1.25 <= H / ho <= 1.75): p += 5000
    if not (1.08 <= s / ho <= 1.50): p += 5000
    if ho > 0.8 * (H + tf): p += 5000
    if (H - ho) / 2 < (tf + 30): p += 5000
    if fy < 250 or fy > 460: p += 5000
    if not (15 <= L / H <= 30): p += 5000
    return p


# ============================================================
# PHASE 2 — Deterministic Inverse Design
# ============================================================

def deterministic_inverse(wu_target, model_forward, scaler_forward, feature_cols):
    bounds = np.array([
        [300, 700], [120, 270], [6, 15], [10, 25],
        [6000, 21000], [200, 560], [200, 830], [250, 460]
    ])

    def objective(x):
        feats = build_features_vector(*x)
        feats_scaled = scaler_forward.transform(feats[feature_cols])
        wu_pred = model_forward.predict(feats_scaled)[0]
        return (wu_pred - wu_target) ** 2 + applicability_penalty(x)

    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=100,
        popsize=15,
        seed=7,
        polish=True
    )

    x = result.x
    feats = build_features_vector(*x)
    wu_pred = float(model_forward.predict(scaler_forward.transform(feats[feature_cols]))[0])
    Area = float(feats["Area"].iloc[0])

    return {
        "design": x,
        "wu_pred": wu_pred,
        "Area": Area,
        "error_pct": abs(wu_pred - wu_target) / wu_target * 100
    }


# ============================================================
# PHASE 3 — Multi-Objective NSGA-II
# ============================================================

class BeamProblem(ElementwiseProblem):
    def __init__(self, wu_target, model_forward, scaler_forward, feature_cols):
        super().__init__(
            n_var=8, n_obj=2,
            xl=np.array([300, 120, 6, 10, 6000, 200, 200, 250]),
            xu=np.array([700, 270, 15, 25, 21000, 560, 830, 460])
        )
        self.wu_target = wu_target
        self.model = model_forward
        self.scaler = scaler_forward
        self.feature_cols = feature_cols

    def _evaluate(self, x, out, *args, **kwargs):
        feats = build_features_vector(*x)
        wu_pred = self.model.predict(self.scaler.transform(feats[self.feature_cols]))[0]
        A = feats["Area"].iloc[0]

        f1 = abs(wu_pred - self.wu_target) / self.wu_target
        f2 = A / 1e4
        pen = applicability_penalty(x)

        out["F"] = [f1 + pen * 1e-6, f2 + pen * 1e-6]
        out["wu_pred"] = wu_pred
        out["Area"] = A


def run_nsga(wu_target, model_forward, scaler_forward, feature_cols):
    problem = BeamProblem(wu_target, model_forward, scaler_forward, feature_cols)
    algo = NSGA2(pop_size=80)
    term = get_termination("n_gen", 120)

    result = minimize(problem, algo, term, seed=1, verbose=False)
    X, F = result.X, result.F

    best_idx = np.argmin(F[:, 0] + F[:, 1])
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

def predict_failure(H, bf, tw, tf, L, ho, s, fy, clf, scaler, encoder, feature_cols):
    df = build_features_vector(H, bf, tw, tf, L, ho, s, fy)
    X_scaled = scaler.transform(df[feature_cols])
    y = clf.predict(X_scaled)[0]
    return encoder.inverse_transform([y])[0]


# ============================================================
# PHASE 6 — Code Compliance Check
# ============================================================

def run_code_check(H, bf, tw, tf, L, ho, s, fy, model, scaler, feature_cols, df):
    if "fy×Area" in df.columns and "fy_Area" not in df.columns:
        df = df.rename(columns={"fy×Area": "fy_Area"})
    feats = build_features_vector(H, bf, tw, tf, L, ho, s, fy)
    wu_pred = float(model.predict(scaler.transform(feats[feature_cols]))[0])

    NN = NearestNeighbors(n_neighbors=1)
    NN.fit(df[feature_cols])
    dist, idx = NN.kneighbors(feats[feature_cols])
    row = df.iloc[idx[0][0]]

    def safe(wu_pred, R): return bool(wu_pred <= R if not pd.isna(R) else False)

    return {
        "wu_pred": wu_pred,
        "wSCI": float(row.get("wSCI", np.nan)),
        "wENM": float(row.get("wENM", np.nan)),
        "wENA": float(row.get("wENA", np.nan)),
        "wAISC": float(row.get("wAISC", np.nan)),
        "Safe_SCI": safe(wu_pred, row.get("wSCI", np.nan)),
        "Safe_ENM": safe(wu_pred, row.get("wENM", np.nan)),
        "Safe_ENA": safe(wu_pred, row.get("wENA", np.nan)),
        "Safe_AISC": safe(wu_pred, row.get("wAISC", np.nan))
    }


# ============================================================
# MAIN EXECUTION SEQUENCE
# ============================================================

def run_inverse_design(wu_target,
                       model_forward, scaler_forward, feature_cols,
                       clf_failure, scaler_failure, label_encoder,
                       df_full):

    st.subheader("🔹 Phase 2 — Deterministic Inverse Design")
    det = deterministic_inverse(wu_target, model_forward, scaler_forward, feature_cols)
    H, bf, tw, tf, L, ho, s, fy = det["design"]

    st.json({
        "Design": {"H": H, "bf": bf, "tw": tw, "tf": tf,
                   "L": L, "ho": ho, "s": s, "fy": fy},
        "wu_pred": det["wu_pred"],
        "Area": det["Area"],
        "Error_%": det["error_pct"]
    })

    st.subheader("🔹 Phase 3 — NSGA-II Multi Objective")
    nsga = run_nsga(wu_target, model_forward, scaler_forward, feature_cols)
    st.json({
        "Best Design": {k: float(v) for k, v in zip(
            ["H", "bf", "tw", "tf", "L", "ho", "s", "fy"], nsga["best_design"])},
        "wu_pred": nsga["best_wu"],
        "Area": nsga["best_Area"]
    })

    st.subheader("🔹 Phase 5 — Failure Mode Prediction")
    failure = predict_failure(
        H, bf, tw, tf, L, ho, s, fy,
        clf_failure, scaler_failure, label_encoder, feature_cols
    )
    st.success(f"Predicted Failure Mode: **{failure}**")

    st.subheader("🔹 Phase 6 — Code Compliance Check")
    code = run_code_check(
        H, bf, tw, tf, L, ho, s, fy,
        model_forward, scaler_forward, feature_cols, df_full
    )
    st.json(code)

    st.success("✔ Full inverse design workflow completed successfully.")
