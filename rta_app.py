# -- coding: utf-8 --
"""
🚦 RTA Prediction & Risk Dashboard
Author: K SAI PRAGNA
Features: Modular, Interactive, Visual Analytics, ML Report
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

# =========================
# SEABORN STYLE
# =========================
sns.set_style("whitegrid")
sns.set_context("notebook")

# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(
    page_title="RTA Prediction & Risk Dashboard",
    layout="wide",
    page_icon="🚦",
    initial_sidebar_state="expanded"
)

# =========================
# PATHS
# =========================
MODEL_PATH = Path(r"C:\Users\MyPc\OneDrive\Desktop\LIVE PROJECT\CODES\RTA_OUTPUT_FILES\rta_best_pipeline.pkl")
SCHEMA_PATH = Path(r"C:\Users\MyPc\OneDrive\Desktop\LIVE PROJECT\CODES\RTA_OUTPUT_FILES\rta_feature_schema.json")
LOGO_PATH = Path(r"C:\Users\MyPc\OneDrive\Desktop\LIVE PROJECT\AISPRY.jpg")

# =========================
# UTILITY FUNCTIONS
# =========================
@st.cache_data
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_schema(path):
    with open(path, "r") as f:
        return json.load(f)

def preprocess_data(df, expected_features):
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0
    return df[expected_features]

def get_final_estimator(pipeline):
    if hasattr(pipeline, "named_steps"):
        final_step_name = list(pipeline.named_steps.keys())[-1]
        return pipeline.named_steps[final_step_name]
    return pipeline

def get_feature_names(preprocessor):
    numeric_feats = preprocessor.transformers_[0][2]
    cat_transformer = preprocessor.transformers_[1][1]
    cat_feats = preprocessor.transformers_[1][2]
    onehot = cat_transformer.named_steps['onehot']
    cat_names = onehot.get_feature_names_out(cat_feats)
    return list(numeric_feats) + list(cat_names)

def plot_feature_importance(estimator, preprocessor, top_n=15):
    if hasattr(estimator, 'feature_importances_'):
        feature_names = get_feature_names(preprocessor)
        importances = estimator.feature_importances_
        fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        fi_df = fi_df.sort_values(by="Importance", ascending=False).head(top_n)
        fig, ax = plt.subplots(figsize=(10,5))
        sns.barplot(x="Importance", y="Feature", data=fi_df, palette="mako", ax=ax)
        ax.set_title(f"Top {top_n} Feature Importances", fontsize=14)
        st.pyplot(fig)

def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="RdBu", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

def show_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.background_gradient(cmap="YlGn"))

# =========================
# LOAD MODEL & SCHEMA
# =========================
try:
    pipeline = load_model(MODEL_PATH)
    schema = load_schema(SCHEMA_PATH)
except Exception as e:
    st.error(f"❌ Loading Error: {e}")
    st.stop()

numeric_features = schema.get("numeric", [])
categorical_features = schema.get("categorical", [])
target_feature = schema.get("target", "accident_occurred")
expected_features = numeric_features + categorical_features

# =========================
# PAGE HEADER
# =========================
st.image(LOGO_PATH, width=220)
st.title("🚦 RTA Prediction & Risk Dashboard")
st.markdown("Upload your traffic dataset to get **accident predictions**, **risk assessment**, feature importance, and ML evaluation reports.")

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload CSV/XLSX file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        st.success(f"✅ Dataset loaded! Shape: {df.shape}")
        st.dataframe(df.head(10))
    except Exception as e:
        st.error(f"❌ Could not load file: {e}")
        st.stop()

    # Preprocess and fill missing features
    X_input = preprocess_data(df, expected_features)

    # =========================
    # PREDICTIONS
    # =========================
    try:
        y_pred = pipeline.predict(X_input)
        df["predicted_accident"] = ["Accident" if x==1 else "No Accident" for x in y_pred]
        df["Accident_Risk"] = np.where(y_pred==1, "High Risk", "Low Risk")
        st.success("✅ Predictions Completed!")

        # Download button
        st.download_button(
            "⬇ Download Prediction Results",
            df.to_csv(index=False).encode("utf-8"),
            file_name="RTA_Predictions.csv",
            mime="text/csv"
        )

        # =========================
        # SUMMARY DASHBOARD
        # =========================
        st.subheader("📊 Prediction Summary")
        counts = df["predicted_accident"].value_counts()
        percentages = df["predicted_accident"].value_counts(normalize=True) * 100
        summary_df = pd.DataFrame({"Count": counts, "Percentage (%)": percentages.round(2)})
        st.dataframe(summary_df.style.background_gradient(cmap="PuBu"))

        # Horizontal bar chart
        st.subheader("🧾 Accident Distribution")
        col1, col2 = st.columns(2)
        with col1: 
            fig_bar, ax_bar = plt.subplots(figsize=(4,2))
            colors = ["#6BCB77", "#FF6F61"]
            sns.barplot(x=counts.values, y=counts.index, palette=colors, ax=ax_bar)
            ax_bar.set_xlabel("Count")
            ax_bar.set_title("Accident Distribution", fontsize=10)
            st.pyplot(fig_bar)

        # =========================
        # FEATURE IMPORTANCE
        # =========================
        final_estimator = get_final_estimator(pipeline)
        st.subheader("🌟 Feature Importance")
        plot_feature_importance(final_estimator, pipeline.named_steps['preprocessor'])

        # =========================
        # MODEL EVALUATION
        # =========================
        
        st.subheader("🧾 Model Evaluation")
        col3, col4 = st.columns(2)
        with col3:
            plot_confusion(df[target_feature] if target_feature in df.columns else np.zeros(len(df)), y_pred)
        with col4:
                show_classification_report(df[target_feature] if target_feature in df.columns else np.zeros(len(df)), y_pred)

        # =========================
        # DATA EXPLORATION
        # =========================
        st.subheader("🔍 Explore Your Data")
        col5, col6,col7 = st.columns(3)

        with col6:
            feature_to_plot = st.selectbox("Select Numeric Feature for Histogram", numeric_features)
            fig3, ax3 = plt.subplots(figsize=(8,6))
            sns.histplot(df[feature_to_plot], kde=True, color="darkblue", bins=20, ax=ax3)
            ax3.set_title(f"Histogram of {feature_to_plot}", fontsize=10, fontweight='bold' )
            st.pyplot(fig3)
            
        st.subheader("🚦 MODEL EXPLORATION & PREDICTION FILTERING ")
        filter_option = st.selectbox("Filter rows by prediction", ["All"] + list(df["predicted_accident"].unique()))
        if filter_option != "All":
            st.dataframe(df[df["predicted_accident"]==filter_option])
        else:
            st.dataframe(df.head(10))
    except Exception as e:
        st.error(f"Prediction failed: {e}")


# =========================
# FOOTER
# =========================
footer_html = """
<div style="text-align: center; margin-top: 30px; font-size: 14px; color: white;">
<hr style="border-top:1px solid #bbb;">
<p><strong>Created By:</strong> K SAI PRAGNA &nbsp;</p>
<p><strong>Project:</strong> Road Traffic Accident Analysis & ML Pipeline</p>
<p>© 2025 All rights reserved</p>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
