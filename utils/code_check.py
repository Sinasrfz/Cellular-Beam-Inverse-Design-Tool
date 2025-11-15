# utils/code_check.py

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def prepare_code_database(df):
    cols = ["H","bf","tw","tf","L","ho","s","fy"]
    code_cols = ["wSCI","wENM","wENA","wAISC",
                 "FailureMode_SCI","FailureMode_ENM","FailureMode_ENA","FailureMode_AISC"]

    db = df[cols + code_cols].copy()
    nbrs = NearestNeighbors(n_neighbors=1).fit(db[cols])
    return db, nbrs


def lookup_code_resistance(x, db, nbrs):
    x = np.array(x).reshape(1,-1)
    dist, idx = nbrs.kneighbors(x)
    row = db.iloc[idx[0][0]]

    return {
        "wSCI":  row["wSCI"],
        "wENM":  row["wENM"],
        "wENA":  row["wENA"],
        "wAISC": row["wAISC"],
        "fm_SCI":  row["FailureMode_SCI"],
        "fm_ENM":  row["FailureMode_ENM"],
        "fm_ENA":  row["FailureMode_ENA"],
        "fm_AISC": row["FailureMode_AISC"],
    }
