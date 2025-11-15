# utils/inverse_multi.py

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem


def build_features(x):
    H,bf,tw,tf,L,ho,s,fy = x
    L_H   = L/H
    H_ho  = H/ho
    s_ho  = s/ho
    tf_tw = tf/tw
    bf_H  = bf/H
    Area  = bf*tf + tw*(H - 2*tf)
    fy_Area = fy * Area

    return pd.DataFrame([{
        "H":H,"bf":bf,"tw":tw,"tf":tf,"L":L,"ho":ho,"s":s,"fy":fy,
        "L/H":L_H,"H/ho":H_ho,"s/ho":s_ho,"tf/tw":tf_tw,"bf/H":bf_H,
        "Area":Area,"fy_Area":fy_Area
    }])


def constraint_penalty(x):
    H,bf,tw,tf,L,ho,s,fy = x
    p = 0.0
    if not(1.25 <= H/ho <= 1.75): p += 1e3
    if not(1.08 <= s/ho <= 1.50): p += 1e3
    if ho > 0.8*(H+tf): p += 1e3
    if L/H < 15 or L/H > 30: p += 1e3
    return p


class MultiObjectiveProblem(ElementwiseProblem):
    def __init__(self, model, scaler, feature_cols, wu_target):
        super().__init__(n_var=8, n_obj=2, n_constr=0,
                         xl=np.array([300,120,6,10,6000,200,250,275]),
                         xu=np.array([700,270,15,25,21000,500,800,460]))
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.wu_target = wu_target

    def _evaluate(self, x, out, *args, **kwargs):

        feats = build_features(x)[self.feature_cols]
        feats_scaled = self.scaler.transform(feats)

        wu_pred = self.model.predict(feats_scaled)[0]
        Area    = feats["Area"].iloc[0]

        f1 = abs(wu_pred - self.wu_target)/self.wu_target
        f2 = Area / 1e4

        p = constraint_penalty(x)

        out["F"] = [f1 + p*1e-6, f2 + p*1e-6]
        out["wu_pred"] = wu_pred
        out["Area"] = Area


def run_nsga2(model, scaler, feature_cols, wu_target):
    problem = MultiObjectiveProblem(model, scaler, feature_cols, wu_target)
    algorithm = NSGA2(pop_size=70)

    res = minimize(problem, algorithm,
                   ("n_gen", 120),
                   seed=42,
                   verbose=False)

    return res
