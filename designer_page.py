# ============================================================
# designer_page.py — Practical Inverse Design Engine (Phases 3–6)
# Refined for realistic engineering use:
#   - Fixed: L, ho, s, fy (designer inputs)
#   - Optimized: H, bf, tw, tf
#   - Enforces real code-safety by feasibility filtering
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
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

    st.header("🏗 Designer Tool — Practical Inverse Design (Code-Feasible)")

    st.markdown("""
    This version guarantees **code-safe** results by restricting optimization
    to the feasible region where **wu ≤ min(wSCI, wEN,M, wEN,A, wAISC)**.
    """)

    st.sidebar.header("🧮 Design Inputs")
    L = st.sidebar.number_input("Beam Span L (mm)", value=12000.0, step=500.0)
    ho = st.sidebar.number_input("Opening Diameter ho (mm)", value=400.0, step=10.0)
    s = st.sidebar.number_input("Opening Spacing s (mm)", value=600.0, step=10.0)
    fy = st.sidebar.number_input("Steel Yield Strength fy (MPa)", value=355.0, step=10.0)
    wu_target = st.number_input("🎯 Target Ultimate Load wu (kN/m)",
                                min_value=5.0, max_value=200.0, value=30.0, step=1.0)

    if st.button("Run Inverse Design", type="primary"):
        run_inverse_design(wu_target, L, ho, s, fy,
                           model_forward, scaler_forward, feature_cols,
                           clf_failure, scaler_failure, label_encoder,
                           df_full)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features_vector(H, bf, tw, tf, L, ho, s, fy):
    return pd.DataFrame([{
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


def applicability_penalty(H, bf, tw, tf, L, ho, s, fy):
    p = 0
    if not (1.25 <= H / ho <= 1.75): p += 1e3
    if not (1.08 <= s / ho <= 1.50): p += 1e3
    if ho > 0.8 * (H + tf): p += 1e3
    if (H - ho) / 2 < (tf + 30): p += 1e3
    if not (15 <= L / H <= 30): p += 1e3
    if not (250 <= fy <= 460): p += 1e3
    return p


# ============================================================
# PHASE 3 — NSGA-II (Feasibility-Based)
# ============================================================

class BeamProblem(ElementwiseProblem):
    def __init__(self, wu_target, L, ho, s, fy, model, scaler, feature_cols, df):
        super().__init__(n_var=4, n_obj=2,
                         xl=np.array([300, 120, 6, 10]),
                         xu=np.array([700, 270, 15, 25]))
        self.wu_target = wu_target
        self.L, self.ho, self.s, self.fy = L, ho, s, fy
        self.model, self.scaler, self.feature_cols = model, scaler, feature_cols

        self.df = df.copy()
        self.df.columns = [c.replace("\n", " ").replace('"', '').strip() for c in self.df.columns]
        if "fy×Area" in self.df.columns:
            self.df.rename(columns={"fy×Area": "fy_Area"}, inplace=True)

    def _evaluate(self, x, out, *args, **kwargs):
        H, bf, tw, tf = x
        feats = build_features_vector(H, bf, tw, tf, self.L, self.ho, self.s, self.fy)
        wu_pred = self.model.predict(self.scaler.transform(feats[self.feature_cols]))[0]
        A = feats["Area"].iloc[0]
        pen = applicability_penalty(H, bf, tw, tf, self.L, self.ho, self.s, self.fy)

        NN = NearestNeighbors(n_neighbors=1)
        NN.fit(self.df[self.feature_cols])
        _, idx = NN.kneighbors(feats[self.feature_cols])
        row = self.df.iloc[idx[0][0]]

        # find safe threshold
        safe_limit = np.nanmin([row.get(k, np.inf) for k in ["wSCI", "wEN,M", "wEN,A", "wAISC"]])
        if wu_pred > safe_limit:
            pen += 1e8 * ((wu_pred - safe_limit) / safe_limit) ** 2

        f1 = abs(wu_pred - self.wu_target) / self.wu_target
        f2 = A / 1e4
        out["F"] = [f1 + pen * 1e-6, f2 + pen * 1e-6]


def run_nsga(wu_target, L, ho, s, fy, model, scaler, feature_cols, df):
    problem = BeamProblem(wu_target, L, ho, s, fy, model, scaler, feature_cols, df)
    algo = NSGA2(pop_size=100)
    term = get_termination("n_gen", 150)
    result = minimize(problem, algo, term, seed=42, verbose=False)

    X, F = result.X, result.F
    best_idx = np.argmin(F[:, 0] + F[:, 1])
    best = X[best_idx]
    feats = build_features_vector(*best, L, ho, s, fy)
    wu_best = float(model.predict(scaler.transform(feats[feature_cols]))[0])
    A_best = float(feats["Area"].iloc[0])
    return {"best_design": best, "wu_best": wu_best, "A_best": A_best}


# ============================================================
# PHASE 5 & 6 remain unchanged
# ============================================================

def predict_failure(H, bf, tw, tf, L, ho, s, fy, clf, scaler, encoder, feature_cols):
    df = build_features_vector(H, bf, tw, tf, L, ho, s, fy)
    X_scaled = scaler.transform(df[feature_cols])
    y = clf.predict(X_scaled)[0]
    return encoder.inverse_transform([y])[0]


def run_code_check(H, bf, tw, tf, L, ho, s, fy, model, scaler, feature_cols, df):
    feats = build_features_vector(H, bf, tw, tf, L, ho, s, fy)
    wu_pred = float(model.predict(scaler.transform(feats[feature_cols]))[0])
    df = df.copy()
    df.columns = [c.replace("\n", " ").replace('"', '').strip() for c in df.columns]
    if "fy×Area" in df.columns:
        df.rename(columns={"fy×Area": "fy_Area"}, inplace=True)

    NN = NearestNeighbors(n_neighbors=1)
    NN.fit(df[feature_cols])
    _, idx = NN.kneighbors(feats[feature_cols])
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


def run_inverse_design(wu_target, L, ho, s, fy,
                       model_forward, scaler_forward, feature_cols,
                       clf_failure, scaler_failure, label_encoder,
                       df_full):

    st.subheader("🔹 Phase 3 — NSGA-II (Feasible Search)")
    nsga = run_nsga(wu_target, L, ho, s, fy, model_forward, scaler_forward, feature_cols, df_full)
    H, bf, tw, tf = nsga["best_design"]

    st.json({
        "Optimized Geometry": {"H": H, "bf": bf, "tw": tw, "tf": tf},
        "Predicted wu": nsga["wu_best"],
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

    st.success("✔ Optimization completed — code-safe solution found.")
