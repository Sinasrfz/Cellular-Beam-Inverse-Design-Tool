import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.termination import get_termination
import matplotlib.pyplot as plt
import sys
sys.path.append(".")

st.title("🌈 Multi-Objective Inverse Design (Phase 3)")

model = joblib.load("UnifiedPipeline/Forward/best_model.joblib")
scaler = joblib.load("UnifiedPipeline/Forward/scaler.joblib")
feat_cols = joblib.load("UnifiedPipeline/Forward/feature_cols.joblib")


# ----------------------------
# Build features
# ----------------------------
def build_features(x):
    H,bf,tw,tf,L,ho,s,fy = x
    L_H   = L/H
    H_ho  = H/ho
    s_ho  = s/ho
    tf_tw = tf/tw
    bf_H  = bf/H
    Area  = bf*tf + tw*(H - 2*tf)
    fyA = fy * Area
    df = pd.DataFrame([{
        "H":H, "bf":bf, "tw":tw, "tf":tf,
        "L":L, "ho":ho, "s":s, "fy":fy,
        "L/H":L_H, "H/ho":H_ho, "s/ho":s_ho,
        "tf/tw":tf_tw, "bf/H":bf_H,
        "Area":Area, "fy_Area":fyA
    }])
    return df[feat_cols]


def constraint_penalty(x):
    H,bf,tw,tf,L,ho,s,fy = x
    p = 0
    if not (1.25 <= H/ho <= 1.75): p += 1e3
    if not (1.08 <= s/ho <= 1.50): p += 1e3
    if ho > 0.8*(H + tf):          p += 1e3
    if not (15 <= L/H <= 30):      p += 1e3
    return p


class BeamInverseProblem(ElementwiseProblem):
    def __init__(self, target):
        super().__init__(
            n_var=8, n_obj=2,
            xl=np.array([300,120,6,10,6000,200,200,250]),
            xu=np.array([700,270,15,25,21000,560,830,460])
        )
        self.target = target

    def _evaluate(self, x, out, **kw):
        feats = build_features(x)
        wu_pred = model.predict(scaler.transform(feats))[0]
        Area = feats["Area"].iloc[0]
        f1 = abs(wu_pred - self.target) / self.target
        f2 = Area/1e4
        penalty = constraint_penalty(x)
        out["F"] = [f1 + penalty*1e-6, f2 + penalty*1e-6]
        out["wu_pred"] = wu_pred
        out["Area"] = Area


wu_target = st.number_input("Enter target wu:", 5.0, 80.0, 32.0)

if st.button("Run NSGA-II Optimization ❤️"):
    problem = BeamInverseProblem(wu_target)
    algo = NSGA2(pop_size=80)
    term = get_termination("n_gen", 150)

    st.write("Running optimization… please wait.")

    res = minimize(problem, algo, term, seed=42, verbose=False)

    F = res.F
    X = res.X

    df = pd.DataFrame(X, columns=["H","bf","tw","tf","L","ho","s","fy"])
    df["f1_load_error"] = F[:,0]
    df["f2_area"] = F[:,1]
    df["wu_pred"] = [problem.evaluate(x, return_values_of=["wu_pred"]) for x in X]

    st.subheader("Pareto Solutions")
    st.dataframe(df, use_container_width=True)

    # Save
    df.to_excel("UnifiedPipeline/Inverse_Multi/Pareto_Solutions.xlsx", index=False)
    st.info("Saved to UnifiedPipeline/Inverse_Multi/Pareto_Solutions.xlsx")

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(F[:,0], F[:,1], c="blue")
    ax.set_xlabel("Load Error Fraction")
    ax.set_ylabel("Area/1e4")
    ax.set_title("Pareto Front")
    st.pyplot(fig)
