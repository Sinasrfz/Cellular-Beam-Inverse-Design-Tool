# =============================================================
# methodology_page.py — Scientific & Engineering Methodology
# Root-based version (consistent with app.py and designer_page.py)
# =============================================================

import streamlit as st
import os


def render():

    st.header("📚 Methodology — Cellular Beam Inverse Design Pipeline")
    st.markdown("---")

    st.markdown("""
    This tool implements a **six-phase Physics-Informed + AI + Optimization**
    pipeline for inverse design of cellular beams.  
    It reflects the same structure used in your research workflow.
    """)

    # =========================================================
    # PHASE 1 — FORWARD SURROGATE MODEL
    # =========================================================
    st.subheader("🔵 Phase 1 — Forward Surrogate Model")
    st.markdown("""
    **Goal:** Learn a fast surrogate model mapping  
    👉 **Design → Ultimate Load (wu)**  
    based on the full FEA-generated database (`original_data.xlsx`).

    **Methods used:**
    - LightGBM  
    - XGBoost  
    - Random Forest  
    → Best model selected automatically by **highest R²**.

    **Outputs stored in root:**
    - `best_model.joblib`
    - `scaler.joblib`
    - `feature_cols.joblib`
    - `model_report.json`
    """)

    if os.path.exists("best_model.joblib"):
        st.success("✔ Phase-1 model found")
    else:
        st.warning("⚠ Phase-1 assets missing")

    st.markdown("---")

    # =========================================================
    # PHASE 2 — Deterministic Inverse Design (DE)
    # =========================================================
    st.subheader("🟢 Phase 2 — Deterministic Inverse Design (Differential Evolution)")
    st.markdown("""
    **Goal:** For a target `wu_target`, find the design vector  
    👉 **[H, bf, tw, tf, L, ho, s, fy]**  
    such that the surrogate model predicts `wu ≈ wu_target`.

    **Method:**
    - Differential Evolution (DE)
    - Physics-informed penalties:
        - s/ho, H/ho, L/H Eurocode limits  
        - fy range constraints  
        - Geometric feasibility

    **Outputs (root):**
    - `inverse_result.json`
    - `inverse_result.xlsx`
    """)

    if os.path.exists("inverse_result.json"):
        st.success("✔ Phase-2 result found")
    else:
        st.info("Run the Designer Tool to generate Phase-2 results")

    st.markdown("---")

    # =========================================================
    # PHASE 3 — Multi-objective NSGA-II
    # =========================================================
    st.subheader("🟣 Phase 3 — Multi-Objective Optimization (NSGA-II)")
    st.markdown("""
    **Goal:** Explore trade-offs between:  
    - f₁: |wu_pred − wu_target| / wu_target  
    - f₂: Area / 1e4  

    **Method:**
    - NSGA-II (population 120, generations 250)
    - Physics-informed constraints identical to Phase-2  
    - Produces **Pareto front** of optimal designs

    **Outputs (root):**
    - `pareto_clean.xlsx`
    - `best_compromise.json`
    - `ParetoFront.png`
    """)

    if os.path.exists("ParetoFront.png"):
        st.success("✔ Phase-3 Pareto front found")
    else:
        st.warning("⚠ Phase-3 results missing")

    st.markdown("---")

    # =========================================================
    # PHASE 4 — Interpretability
    # =========================================================
    st.subheader("🟠 Phase 4 — Interpretability & Diagnostics")
    st.markdown("""
    **Goal:** Explain surrogate model behaviour via:  
    - Parity plot (pred vs true)  
    - Error distribution & residuals  
    - Feature importance  
    - SHAP values (global & local)  
    - Correlation heatmaps  
    - Pareto variable distributions

    **Typical outputs (root):**
    - `plot_parity.png`
    - `plot_error_hist.png`
    - `plot_residuals.png`
    - `plot_corr.png`
    - `plot_shap_summary.png`
    - `plot_shap_bar.png`
    - `plot_pareto_dist.png`
    """)

    shap_ok = any("shap" in f.lower() for f in os.listdir(".")) if os.path.isdir(".") else False
    if shap_ok:
        st.success("✔ SHAP / interpretability plots detected")
    else:
        st.warning("⚠ Phase-4 outputs not found")

    st.markdown("---")

    # =========================================================
    # PHASE 5 — FAILURE MODE CLASSIFIER
    # =========================================================
    st.subheader("🟥 Phase 5 — Failure Mode Classifier")
    st.markdown("""
    **Goal:** Predict failure mode for any design  
    👉 **{WPS, WPB, BGS, BGB, VBT}**

    **Method:**
    - Trained on enriched Phase-1 dataset  
    - Auto-selects best among LightGBM, XGBoost, RF

    **Outputs (root):**
    - `best_failure_classifier.joblib`
    - `scaler_failure.joblib`
    - `label_encoder.joblib`
    """)

    if os.path.exists("best_failure_classifier.joblib"):
        st.success("✔ Failure classifier found")
    else:
        st.warning("⚠ Failure classifier missing")

    st.markdown("---")

    # =========================================================
    # PHASE 6 — CODE-BASED SAFETY CHECKS
    # =========================================================
    st.subheader("🟤 Phase 6 — Code-Based Hybrid Safety Checks")
    st.markdown("""
    **Goal:** Compare ML-predicted ultimate load against:  
    - SCI P355 resistance  
    - EN 1993-1-13 (M & A)  
    - AISC DG31  

    **Method:**
    - Nearest-neighbour lookup in the FEA database  
    - Retrieve:
        - wSCI  
        - wENM  
        - wENA  
        - wAISC  
    - Compute safety ratios (wu_pred / R_code)

    **Outputs (root):**
    - `codecheck_result.json`
    - `codecheck_report.xlsx`
    - `code_comparison.png`
    """)

    if os.path.exists("code_comparison.png"):
        st.success("✔ Code-check results found")
    else:
        st.warning("⚠ Phase-6 code-check outputs missing")

    st.markdown("---")

    # =========================================================
    # END NOTE
    # =========================================================
    st.info("""
    ✅ This methodology summary matches the implemented pipeline.  
    ✅ All assets are expected in the **GitHub root directory**.  
    ✅ Phases 2–6 can be executed directly through the **Designer Tool** page.
    """)
