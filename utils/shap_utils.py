# utils/shap_utils.py

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_shap(model, X_scaled, X_raw):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_scaled)
    return shap_values


def plot_shap_summary(shap_values, X_raw):
    plt.figure(figsize=(7,5))
    shap.summary_plot(shap_values, X_raw, show=False)
    plt.tight_layout()


def plot_shap_bar(shap_values, X_raw):
    plt.figure(figsize=(7,5))
    shap.summary_plot(shap_values, X_raw, plot_type="bar", show=False)
    plt.tight_layout()
