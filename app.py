import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import re
import matplotlib.pyplot as plt
from io import BytesIO

# Optimization & models
from scipy.optimize import differential_evolution
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.termination import get_termination

# ==============================
# HELPERS — CLEANING / FEATURES
# ==============================

def clean_columns(df):
    df = df.copy()
    df.columns = [re.sub(r"\s+", "_", str(c).replace("\n", " ").replace('"', "")) for c in df.columns]
    rename_map = {
        "fy×Area": "fy_Area",
        "Applicable?": "Applicable_SCI",
        "wSCI_(kN/m)": "wSCI",
        "wEN,M_(kN/m)": "wENM",
        "wEN,A_(kN/m)": "wENA",
        "wAISC_(kN/m)": "wAISC",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df

def enrich(df):
    df = df.copy()
    df["L/H"]   = df["L"] / df["H"]
    df["H/ho"]  = df["H"] / df["ho"]
    df["s/ho"]  = df["s"] / df["ho"]
    df["tf/tw"] = df["tf"] / df["tw"]
    df["bf/H"]  = df["bf"] / df["H"]
    df["Area"]  = df["bf"]*df["tf"] + df["tw"]*(df["H"] - 2*df["tf"])
    df["fy_Area"] = df["fy"] * df["Area"]
    return df

# ==============================
# LOAD RAW DATA (21.xlsx)
# ==============================

@st.cache_data
def load_dataset():
    df = pd.read_excel("21.xlsx")
    df = clean_columns(df)
    df = enrich(df)
    return df

df_raw = load_dataset()

# FEATURE LIST (same as phases 1–6)
feature_cols = [
    "H","bf","tw","tf","L","ho","s","fy",
    "L/H","H/ho","s/ho","tf/tw","bf/H","Area","fy_Area"
]

# DESIGN VARIABLES (for Phase 2–3)
var_names = ["H","bf","tw","tf","L","ho","s","fy"]
bounds = np.array([
    [300,700], [120,270], [6,15], [10,25],
    [6000,21000], [200,560], [200,830], [250,460]
])

# ==============================
# PHASE 1 — TRAIN BEST FORWARD MODEL
# ==============================

def train_forward_model(df):
    df = df.dropna(subset=["wu"]).copy()
    df = enrich(df)

    X = df[feature_cols]
    y = df["wu"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    models = {
        "LGBM": LGBMRegressor(n_estimators=500, learning_rate=0.04,
                              subsample=0.85, colsample_bytree=0.85),
        "XGB": XGBRegressor(n_estimators=350, learning_rate=0.05,
                            max_depth=5, subsample=0.85, colsample_bytree=0.85),
        "RF": RandomForestRegressor(n_estimators=350)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        r2 = r2_score(y_test, pred)
        results[name] = r2

    best_name = max(results, key=lambda k: results[k])
    best_model = models[best_name]

    os.makedirs("UnifiedPipeline/ForwardModel", exist_ok=True)
    joblib.dump(best_model, "UnifiedPipeline/ForwardModel/best_model.joblib")
    joblib.dump(scaler, "UnifiedPipeline/ForwardModel/scaler.joblib")
    joblib.dump(feature_cols, "UnifiedPipeline/ForwardModel/feature_cols.joblib")

    return best_model, scaler

# ==============================
# PHASE 2 — DETERMINISTIC INVERSE
# ==============================

def build_features_from_x(x):
    df = pd.DataFrame([{
        "H":x[0], "bf":x[1], "tw":x[2], "tf":x[3],
        "L":x[4], "ho":x[5], "s":x[6], "fy":x[7]
    }])
    return enrich(df)[feature_cols]

def applicability_penalty(x):
    H,bf,tw,tf,L,ho,s,fy = x
    p = 0
    if not (1.25 <= H/ho <= 1.75): p += 5000
    if not (1.08 <= s/ho <= 1.50): p += 5000
    if ho > 0.8*(H + tf): p += 5000
    if (H - ho)/2 < (tf + 30): p += 5000
    if not (15 <= L/H <= 30): p += 5000
    if fy < 250 or fy > 460: p += 5000
    return p

def inverse_design(wu_target, model, scaler):
    def obj(x):
        feats = build_features_from_x(x)
        wu_pred = model.predict(scaler.transform(feats))[0]
        return (wu_pred - wu_target)**2 + applicability_penalty(x)

    result = differential_evolution(
        obj, bounds=bounds, popsize=20, maxiter=120, seed=42
    )

    x_opt = result.x
    feats = build_features_from_x(x_opt)
    wu_pred = model.predict(scaler.transform(feats))[0]

    return {
        "Design": {k: float(v) for k,v in zip(var_names, x_opt)},
        "wu_pred": float(wu_pred),
        "error_%": float(abs(wu_pred - wu_target)/wu_target * 100),
        "Area": float(feats["Area"].iloc[0])
    }

# ==============================
# PHASE 3 — NSGA-II MULTI-OBJECTIVE
# ==============================

class BeamProblem(ElementwiseProblem):
    def __init__(self, wu_target, model, scaler):
        super().__init__(n_var=8, n_obj=2, n_constr=0,
                         xl=bounds[:,0], xu=bounds[:,1])
        self.wu_target = wu_target
        self.model = model
        self.scaler = scaler

    def _evaluate(self, x, out, **kw):
        feats = build_features_from_x(x)
        wu_pred = self.model.predict(self.scaler.transform(feats))[0]
        area = feats["Area"].iloc[0]
        f1 = abs(wu_pred - self.wu_target) / self.wu_target
        f2 = area / 1e4
        pen = applicability_penalty(x)
        out["F"] = [f1 + pen*1e-6, f2 + pen*1e-6]

def run_nsga(wu_target, model, scaler):
    algo = NSGA2(pop_size=80)
    term = get_termination("n_gen", 80)
    prob = BeamProblem(wu_target, model, scaler)
    res = minimize(prob, algo, term, seed=42, verbose=False)
    return res

# ==============================
# PHASE 6 — CODE CHECK (NEAREST NEIGHBOR)
# ==============================

from sklearn.neighbors import NearestNeighbors
NN = NearestNeighbors(n_neighbors=1)
NN.fit(df_raw[feature_cols])

def code_check(H,bf,tw,tf,L,ho,s,fy, model, scaler):
    feats = enrich(pd.DataFrame([{
        "H":H,"bf":bf,"tw":tw,"tf":tf,
        "L":L,"ho":ho,"s":s,"fy":fy
    }]))[feature_cols]

    wu_pred = float(model.predict(scaler.transform(feats))[0])
    dist, idx = NN.kneighbors(feats)
    row = df_raw.iloc[idx[0][0]]

    return {
        "wSCI": {"R": float(row["wSCI"]), "Safe": wu_pred <= row["wSCI"]},
        "wENM": {"R": float(row["wENM"]), "Safe": wu_pred <= row["wENM"]},
        "wENA": {"R": float(row["wENA"]), "Safe": wu_pred <= row["wENA"]},
        "wAISC": {"R": float(row["wAISC"]), "Safe": wu_pred <= row["wAISC"]},
    }

# ==============================
# STREAMLIT GUI
# ==============================

st.title("🧠 Cellular Beam Inverse Design Tool")
st.write("A unified AI–Physics inverse design engine for cellular beams ❤️")

wu_input = st.number_input("Enter target ultimate load (wu) [kN/m]:",
                           min_value=5.0, max_value=300.0, value=40.0)

if st.button("RUN FULL DESIGN"):
    st.success("Starting full pipeline...")

    # ---- PHASE 1 ----
    model, scaler = train_forward_model(df_raw)
    st.success("Phase 1 done ✔")

    # ---- PHASE 2 ----
    det = inverse_design(wu_input, model, scaler)
    st.subheader("Phase 2 — Deterministic Design")
    st.json(det)

    # ---- PHASE 3 ----
    st.subheader("Phase 3 — NSGA-II Multi-Objective")
    res = run_nsga(wu_input, model, scaler)
    st.write(f"Pareto solutions: {res.F.shape[0]}")
    st.write(res.F)

    # ---- PHASE 6 ----
    st.subheader("Phase 6 — Code Check (SCI / EN / AISC)")
    d = det["Design"]
    cc = code_check(d["H"],d["bf"],d["tw"],d["tf"],
                    d["L"],d["ho"],d["s"],d["fy"],
                    model, scaler)
    st.json(cc)

    st.success("ALL PHASES COMPLETED ❤️")
