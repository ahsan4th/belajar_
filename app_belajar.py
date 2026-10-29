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
                sns.kdeplot(ax=axes[1], data=test_df, x='battery_power', label='Test Data', fill=True, alpha=0.5, color='orange')
            axes[1].set_title('Perbandingan Distribusi Battery Power')
            axes[1].set_xlabel('Battery Power')
            axes[1].set_ylabel('Density')
            axes[1].legend()
        
        plt.tight_layout()
        st.pyplot(fig)

    # ==========================================
    # TAB 2: Model Training
    # ==========================================
    with tab2:
        st.subheader("Pelatihan Model Machine Learning")
        
        if 'price_range' not in train_df.columns:
            st.error("Kolom target 'price_range' tidak ditemukan di data training.")
        else:
            test_size = st.slider("Ukuran Data Validasi (Test Size)", 0.1, 0.4, 0.2, 0.05)
            
            X = train_df.drop(columns=['price_range'])
            if 'id' in X.columns:
                X = X.drop(columns=['id'])
            y = train_df['price_range']

            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            models = {
                'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), True),
                'Random Forest': (RandomForestClassifier(random_state=42, n_estimators=100), False),
                'XGBoost': (xgb.XGBClassifier(random_state=42, eval_metric='mlogloss'), False),
                'Linear Discriminant Analysis (LDA)': (LinearDiscriminantAnalysis(), True)
            }

            if st.button("Mulai Latih Model", type="primary"):
                results = {}
                trained_models = {}
                val_cm = {}

                with st.spinner("Melatih model..."):
                    for name, (clf, use_scaled) in models.items():
                        X_tr = X_train_scaled if use_scaled else X_train
                        X_vl = X_val_scaled if use_scaled else X_val
                        
                        clf.fit(X_tr, y_train)
                        preds = clf.predict(X_vl)
                        
                        results[name] = accuracy_score(y_val, preds)
                        trained_models[name] = (clf, use_scaled)
                        val_cm[name] = confusion_matrix(y_val, preds)

                st.session_state['trained_models'] = trained_models
                st.session_state['scaler'] = scaler
                st.session_state['results'] = results
                st.session_state['val_cm'] = val_cm
                st.session_state['feature_columns'] = list(X.columns)

                st.success("Pelatihan selesai!")

            if 'results' in st.session_state:
                res_df = pd.DataFrame(list(st.session_state['results'].items()), columns=['Model', 'Akurasi'])
                res_df = res_df.sort_values(by='Akurasi', ascending=False).reset_index(drop=True)
                
                col_res1, col_res2 = st.columns([1, 2])
                with col_res1:
                    st.write("### Perbandingan Akurasi")
                    st.dataframe(res_df.style.format({'Akurasi': '{:.4f}'}))
                
                with col_res2:
                    fig_acc, ax_acc = plt.subplots(figsize=(6, 3))
                    sns.barplot(data=res_df, x='Akurasi', y='
