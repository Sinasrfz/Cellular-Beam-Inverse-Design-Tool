import streamlit as st

st.set_page_config(
    page_title="Cellular Beam Inverse Design Tool",
    layout="wide"
)

st.title("🧠 Cellular Beam Inverse Design Tool")
st.write("Welcome! Use the sidebar to access:")
st.markdown("""
- **Designer Tool** → Predict geometry for a given ultimate load (wu)  
- **Inverse Multi-objective Optimization** → Pareto front search  
- **Code Checks** → SCI • EN • AISC resistances  
- **Interpretability** → SHAP, correlations, diagnostics  
- **Database Explorer** → Raw dataset viewer  
""")

st.success("The system is ready. Select a page from the left menu ❤️")
