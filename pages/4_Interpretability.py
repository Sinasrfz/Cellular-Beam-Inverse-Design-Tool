import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🔎 Interpretability Analysis (SHAP & Diagnostics)")

model = joblib.load("UnifiedPipeline/Forward/best_model.joblib")
scaler = joblib.load("UnifiedPipeline/Forward/scaler.joblib")
feat_cols = joblib.load("UnifiedPipeline/Forward/feature_cols.joblib")

df = pd.read_excel("data/full_database.xlsx")

X = df[feat_cols]
X_scaled = scaler.transform(X)
pred = model.predict(X_scaled)

st.subheader("Parity Plot")
fig, ax = plt.subplots()
ax.scatter(df["wu"], pred, alpha=0.4)
ax.plot([df["wu"].min(), df["wu"].max()],
        [df["wu"].min(), df["wu"].max()], "r--")
st.pyplot(fig)

explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(X_scaled)

st.subheader("SHAP Global Importance")
fig = plt.figure()
shap.summary_plot(shap_vals, X, plot_type="bar", show=False)
st.pyplot(fig)

st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(df[feat_cols+["wu"]].corr(), cmap="coolwarm")
st.pyplot(fig)
