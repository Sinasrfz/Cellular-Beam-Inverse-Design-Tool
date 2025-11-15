import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

st.title("📘 Code-Based Resistance Check (SCI • EN • AISC)")

# Load assets
model = joblib.load("UnifiedPipeline/Forward/best_model.joblib")
scaler = joblib.load("UnifiedPipeline/Forward/scaler.joblib")
feat_cols = joblib.load("UnifiedPipeline/Forward/feature_cols.joblib")
df = pd.read_excel("UnifiedPipeline/CodeDB/codedb_clean.xlsx")

NN = NearestNeighbors(n_neighbors=1)
NN.fit(df[feat_cols])

def build_feats(x):
    H,bf,tw,tf,L,ho,s,fy = x
    L_H   = L/H
    H_ho  = H/ho
    s_ho  = s/ho
    tf_tw = tf/tw
    bf_H  = bf/H
    Area  = bf*tf + tw*(H - 2*tf)
    fyA = fy * Area
    row = pd.DataFrame([{
        "H":H, "bf":bf, "tw":tw, "tf":tf,
        "L":L, "ho":ho, "s":s, "fy":fy,
        "L/H":L_H, "H/ho":H_ho, "s/ho":s_ho,
        "tf/tw":tf_tw, "bf/H":bf_H,
        "Area":Area, "fy_Area":fyA
    }])
    return row[feat_cols]

def lookup(x):
    feats = build_feats(x)
    scaled = scaler.transform(feats)
    _, idx = NN.kneighbors(scaled)
    return df.iloc[idx[0][0]]

H = st.number_input("H", 100.0, 1000.0, 300.0)
bf = st.number_input("bf", 60.0, 400.0, 120.0)
tw = st.number_input("tw", 3.0, 40.0, 6.0)
tf = st.number_input("tf", 3.0, 40.0, 10.0)
L  = st.number_input("L", 2000.0, 30000.0, 15000.0)
ho = st.number_input("ho", 50.0, 600.0, 230.0)
s  = st.number_input("s", 50.0, 1200.0, 350.0)
fy = st.number_input("fy", 150.0, 700.0, 280.0)

if st.button("Check Codes"):
    x = [H,bf,tw,tf,L,ho,s,fy]
    feats = build_feats(x)
    wu_pred = model.predict(scaler.transform(feats))[0]

    row = lookup(x)

    st.subheader(f"Predicted ML Ultimate Load = {wu_pred:.2f} kN/m")

    table = pd.DataFrame({
        "Code": ["SCI","EN-M","EN-A","AISC"],
        "Resistance": [row["wSCI"], row["wENM"], row["wENA"], row["wAISC"]],
        "Failure Mode": [
            row["FailureMode_SCI"],
            row.get("Failure_mode_EN_M"),
            row.get("Failure_mode_EN_A"),
            row.get("Failure_mode_AISC")
        ],
        "wu/R": [
            wu_pred/row["wSCI"],
            wu_pred/row["wENM"],
            wu_pred/row["wENA"],
            wu_pred/row["wAISC"]
        ]
    })

    st.dataframe(table, use_container_width=True)

    fig, ax = plt.subplots()
    ax.bar(table["Code"], table["Resistance"])
    ax.axhline(wu_pred, color="red", linestyle="--", label="ML wu_pred")
    ax.legend()
    ax.set_ylabel("kN/m")
    st.pyplot(fig)
