# utils/inverse_single.py

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution


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
    penalty = 0.0

    if not(1.25 <= H/ho <= 1.75): penalty += 1e3
    if not(1.08 <= s/ho <= 1.50): penalty += 1e3
    if ho > 0.8*(H+tf): penalty += 1e3
    if L/H < 15 or L/H > 30: penalty += 1e3

    return penalty


def single_objective_inverse(model, scaler, feature_cols, target_wu):
    bounds = [
        (300,700), (120,270), (6,15), (10,25),
        (6000,21000), (200,500), (250,800), (275,460)
    ]

    def objective(x):
        feats = build_features(x)
        feats = feats[feature_cols]
        feats_scaled = scaler.transform(feats)

        wu_pred = model.predict(feats_scaled)[0]
        loss = (wu_pred - target_wu)**2 + constraint_penalty(x)
        return loss

    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=150,
        popsize=20,
        seed=42,
        polish=True
    )

    x_opt = result.x
    feats = build_features(x_opt)[feature_cols]
    wu_pred = model.predict(scaler.transform(feats))[0]

    return x_opt, wu_pred
