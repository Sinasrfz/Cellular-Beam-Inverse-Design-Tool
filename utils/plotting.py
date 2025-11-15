# utils/plotting.py

import matplotlib.pyplot as plt
import numpy as np


def plot_response_comparison(target, wu_pred):
    plt.figure(figsize=(5,4))
    plt.bar(["Target","Predicted"], [target, wu_pred], color=["gray","blue"])
    plt.ylabel("Ultimate Load (kN/m)")
    plt.title("Inverse Design Result")
    plt.grid(alpha=0.3)
    plt.tight_layout()


def plot_pareto(F):
    plt.figure(figsize=(6,5))
    plt.scatter(F[:,0], F[:,1], c="blue", s=40)
    plt.xlabel("Load Error")
    plt.ylabel("Area / 1e4")
    plt.title("NSGA-II Pareto Front")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_code_bars(code_dict):
    keys = ["wSCI","wENM","wENA","wAISC"]
    values = [code_dict[k] for k in keys]

    plt.figure(figsize=(6,4))
    plt.bar(keys, values, color="orange")
    plt.ylabel("Resistance (kN/m)")
    plt.title("Code-Based Resistances")
    plt.grid(alpha=0.2)
    plt.tight_layout()
