import streamlit as st
import pandas as pd

st.title("📂 Database Explorer")

df = pd.read_excel("data/full_database.xlsx")
st.dataframe(df, use_container_width=True)

st.download_button(
    "Download dataset",
    data=df.to_csv(index=False),
    file_name="full_database.csv"
)
