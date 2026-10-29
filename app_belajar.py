import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Mobile Price Classification App",
    page_icon="📱",
    layout="wide"
)

st.title("📱 Mobile Price Classification & Prediction App")
st.markdown("Aplikasi interaktif untuk eksplorasi data, pelatihan model machine learning, prediksi, dan evaluasi Confusion Matrix.")

# --- SIDEBAR: Upload Dataset ---
st.sidebar.header("📁 Upload Dataset")
train_file = st.sidebar.file_uploader("Upload Data Training (CSV)", type=["csv"])
test_file = st.sidebar.file_uploader("Upload Data Testing (CSV)", type=["csv"])

tab1, tab2, tab3, tab4 = st.tabs(["📊 Eksplorasi Data (EDA)", "⚙️ Training & Evaluasi Model", "🔮 Prediksi & Confusion Matrix", "ℹ️ Informasi"])

if train_file is not None:
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file) if test_file is not None else None

    # ==========================================
    # TAB 1: EDA
    # ==========================================
    with tab1:
        st.subheader("Ringkasan Data Training")
        col1, col2 = st.columns(2)
        col1.metric("Jumlah Baris Training", train_df.shape[0])
        col2.metric("Jumlah Kolom / Fitur", train_df.shape[1])

        st.dataframe(train_df.head(10), use_container_width=True)
        st.markdown("---")
        
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if 'price_range' in train_df.columns:
            sns.countplot(ax=axes[0], x='price_range', hue='price_range', data=train_df, palette='viridis', legend=False)
            axes[0].set_title('Distribusi Price Range (Data Latih)')
            axes[0].set_xlabel('Price Range')
            axes[0].set_ylabel('Jumlah')

        if 'battery_power' in train_df.columns:
            sns.kdeplot(ax=axes[1], data=train_df, x='battery_power', label='Train Data', fill=True, alpha=0.5, color='blue')
            if test_df is not None and 'battery_power' in test_df.columns:
