import streamlit as st
import pandas as pd

st.set_page_config(page_title="Simple CSV Viewer", layout="wide")
st.title("💓 Heart Failure Dataset Viewer (Safe Version)")

uploaded_file = st.file_uploader("Upload your Heart Failure CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Raw Data")
        st.dataframe(df)

        st.subheader("🔍 Dataset Info")
        st.write("Number of Rows:", df.shape[0])
        st.write("Number of Columns:", df.shape[1])

        if "DEATH_EVENT" in df.columns:
            st.subheader("📊 DEATH_EVENT Distribution")
            death_count = df["DEATH_EVENT"].value_counts().rename(index={0: "Survived", 1: "Died"})
            st.bar_chart(death_count)
        else:
            st.warning("⚠️ 'DEATH_EVENT' column not found in your dataset.")

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
else:
    st.info("📤 Please upload a CSV file to proceed.")
