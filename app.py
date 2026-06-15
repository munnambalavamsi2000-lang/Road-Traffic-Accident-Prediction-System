# -*- coding: utf-8 -*-
"""
🚦 Advanced Road Traffic Accident (RTA) Prediction Deployment
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")

# ==========================
# CONFIG
# ==========================
MODEL_PATH = r"C:\Smart_Accident_Risk_Classification\data_set\rta_best_pipeline.pkl"
SCHEMA_PATH = r"C:\Smart_Accident_Risk_Classification\data_set\rta_feature_schema.json"


# ==========================
# LOAD MODEL & SCHEMA
# ==========================
try:
    pipeline = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

try:
    with open(SCHEMA_PATH, "r") as f:
        schema = json.load(f)
except Exception as e:
    st.error(f"❌ Error loading schema: {e}")
    st.stop()

numeric_features = schema.get("numeric", [])
categorical_features = schema.get("categorical", [])
target_feature = schema.get("target", "accident_occurred")
expected_features = numeric_features + categorical_features

# ==========================
# STREAMLIT UI
# ==========================
st.set_page_config(page_title="RTA Prediction", layout="wide")



st.title("🚦 Advanced Road Traffic Accident (RTA) Prediction")
st.write("""
Upload traffic data and get **predicted accident likelihood** with interactive visualizations.
""")

uploaded_file = st.file_uploader("Upload CSV/XLSX file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        st.success(f" File loaded successfully! Shape: {df.shape}")
        st.dataframe(df.head(10))
    except Exception as e:
        st.error(f" Error loading file: {e}")
        st.stop()

    # Fill missing columns with 0
    missing_cols = [c for c in expected_features if c not in df.columns]
    if missing_cols:
        st.warning(f" Missing columns: {missing_cols} (filled with 0).")
        for col in missing_cols:
            df[col] = 0

    # Select columns to predict
    X_input = df[[c for c in expected_features if c in df.columns]]

    # ==========================
    # MAKE PREDICTIONS
    # ==========================
    try:
        y_pred = pipeline.predict(X_input)
        df["predicted_accident"] = ["Accident" if int(x)==1 else "No Accident" for x in y_pred]
        st.success("✅ Predictions completed!")

        # Download predictions
        st.download_button(
            "Download Predictions",
            df.to_csv(index=False).encode('utf-8'),
            file_name="RTA_Predictions.csv",
            mime="text/csv"
        )

        # ==========================
        # DASHBOARD
        # ==========================
        st.subheader("📊 Prediction Summary")
        counts = df["predicted_accident"].value_counts()
        percentages = df["predicted_accident"].value_counts(normalize=True) * 100
        summary_df = pd.DataFrame({"Count": counts, "Percentage": percentages})

        # Display summary table and pie chart side-by-side
        col1, col2 = st.columns([1,1])
        with col1:
            st.dataframe(summary_df)

        with col2:
            fig1, ax1 = plt.subplots(figsize=(4,4))
            ax1.pie(
                counts,
                labels=counts.index,
                autopct="%1.1f%%",
                colors=["#FF6B6B", "#4ECDC4"],
                startangle=90,
                textprops={'fontsize':10}
            )
            ax1.set_title("Accident Prediction Distribution", fontsize=12)
            st.pyplot(fig1)

        # ==========================
        # FEATURE IMPORTANCE (if supported)
        # ==========================
        if hasattr(pipeline.named_steps['clf'], 'feature_importances_'):
            st.subheader("🌟 Top Feature Importances")
            def get_feature_names(preprocessor):
                numeric_feats = preprocessor.transformers_[0][2]
                cat_transformer = preprocessor.transformers_[1][1]
                cat_feats = preprocessor.transformers_[1][2]
                onehot = cat_transformer.named_steps['onehot']
                cat_names = onehot.get_feature_names_out(cat_feats)
                return list(numeric_feats) + list(cat_names)

            feature_names = get_feature_names(pipeline.named_steps['preprocessor'])
            importances = pipeline.named_steps['clf'].feature_importances_
            fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
            fi_df = fi_df.sort_values(by="importance", ascending=False).head(15)
            
            fig2, ax2 = plt.subplots(figsize=(10,5))
            sns.barplot(data=fi_df, x="importance", y="feature", palette="viridis", ax=ax2)
            st.pyplot(fig2)

        # ==========================
        # INTERACTIVE EXPLORATION
        # ==========================
        st.subheader("Explore Your Data")
        col1, col2 = st.columns(2)
        with col1:
            feature_to_plot = st.selectbox("Select numeric feature for histogram", numeric_features)
            fig3, ax3 = plt.subplots()
            sns.histplot(df[feature_to_plot], kde=True, bins=20, color="#4ECDC4", ax=ax3)
            ax3.set_title(f"Histogram of {feature_to_plot}")
            st.pyplot(fig3)
        with col2:
            st.write("Filter dataset by predicted accident:")
            accident_filter = st.selectbox("Show rows", ["All", "Accident", "No Accident"])
            if accident_filter != "All":
                st.dataframe(df[df["predicted_accident"]==accident_filter])

    except Exception as e:
        st.error(f"Error during prediction: {e}")

# ==========================
# FOOTER WITH NAME & BATCH
# ==========================
st.markdown("---")
st.markdown("**Created by:** Munnam Balavamsi")
