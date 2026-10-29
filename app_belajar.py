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
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Mobile Price Classification App",
    page_icon="📱",
    layout="wide"
)

st.title("📱 Mobile Price Classification & Prediction App")
st.markdown("Aplikasi interaktif untuk analisis eksplorasi data, pelatihan model machine learning, dan prediksi rentang harga ponsel.")

# --- SIDEBAR: Upload Dataset ---
st.sidebar.header("📁 Upload Dataset")
train_file = st.sidebar.file_uploader("Upload Data Training (CSV)", type=["csv"])
test_file = st.sidebar.file_uploader("Upload Data Testing (CSV)", type=["csv"])

# Tab Navigasi
tab1, tab2, tab3, tab4 = st.tabs(["📊 Eksplorasi Data (EDA)", "⚙️ Training & Evaluasi Model", "🔮 Prediksi Data Uji", "ℹ️ Informasi"])

if train_file is not None:
    # Membaca data training
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
        st.subheader("Visualisasi Fitur")
        
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Visualisasi 1: Distribusi target price_range
        if 'price_range' in train_df.columns:
            sns.countplot(ax=axes[0], x='price_range', hue='price_range', data=train_df, palette='viridis', legend=False)
            axes[0].set_title('Distribusi Price Range (Data Latih)')
            axes[0].set_xlabel('Price Range')
            axes[0].set_ylabel('Jumlah')
        else:
            axes[0].text(0.5, 0.5, "Kolom 'price_range' tidak ditemukan", ha='center', va='center')

        # Visualisasi 2: Distribusi battery_power
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
            
            # Persiapan Data
            X = train_df.drop(columns=['price_range'])
            if 'id' in X.columns:
                X = X.drop(columns=['id'])
            y = train_df['price_range']

            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            # Standardisasi
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Inisialisasi Model
            models = {
                'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), True),
                'Random Forest': (RandomForestClassifier(random_state=42, n_estimators=100), False),
                'XGBoost': (xgb.XGBClassifier(random_state=42, eval_metric='mlogloss'), False),
                'Linear Discriminant Analysis (LDA)': (LinearDiscriminantAnalysis(), True)
            }

            if st.button("Mulai Latih Model", type="primary"):
                results = {}
                trained_models = {}

                with st.spinner("Melatih model..."):
                    for name, (clf, use_scaled) in models.items():
                        X_tr = X_train_scaled if use_scaled else X_train
                        X_vl = X_val_scaled if use_scaled else X_val
                        
                        clf.fit(X_tr, y_train)
                        preds = clf.predict(X_vl)
                        acc = accuracy_score(y_val, preds)
                        results[name] = acc
                        trained_models[name] = (clf, use_scaled)

                # Simpan model ke session_state agar dapat dipakai di Tab Prediksi
                st.session_state['trained_models'] = trained_models
                st.session_state['scaler'] = scaler
                st.session_state['results'] = results

                st.success("Pelatihan selesai!")

            # Menampilkan hasil pelatihan jika sudah ada
            if 'results' in st.session_state:
                res_df = pd.DataFrame(list(st.session_state['results'].items()), columns=['Model', 'Akurasi'])
                res_df = res_df.sort_values(by='Akurasi', ascending=False).reset_index(drop=True)
                
                col_res1, col_res2 = st.columns([1, 2])
                with col_res1:
                    st.write("### Perbandingan Akurasi")
                    st.dataframe(res_df.style.format({'Akurasi': '{:.4f}'}))
                
                with col_res2:
                    fig_acc, ax_acc = plt.subplots(figsize=(6, 3))
                    sns.barplot(data=res_df, x='Akurasi', y='Model', palette='mako', ax=ax_acc)
                    ax_acc.set_xlim(0, 1.0)
                    for p in ax_acc.patches:
                        width = p.get_width()
                        ax_acc.annotate(f'{width:.3f}', (width + 0.02, p.get_y() + p.get_height() / 2.),
                                        ha='center', va='center')
                    st.pyplot(fig_acc)

    # ==========================================
    # TAB 3: Prediksi Data Uji (Testing)
    # ==========================================
    with tab3:
        st.subheader("Prediksi pada Data Testing")
        if test_df is None:
            st.info("Silakan unggah file Data Testing (CSV) pada sidebar untuk menjalankan prediksi.")
        elif 'trained_models' not in st.session_state:
            st.warning("Silakan latih model terlebih dahulu pada tab 'Training & Evaluasi Model'.")
        else:
            model_options = list(st.session_state['trained_models'].keys())
            selected_model_name = st.selectbox("Pilih Model untuk Prediksi:", model_options)

            if st.button("Jalankan Prediksi"):
                model, use_scaled = st.session_state['trained_models'][selected_model_name]
                scaler = st.session_state['scaler']

                X_test_data = test_df.drop(columns=['id']) if 'id' in test_df.columns else test_df.copy()

                if use_scaled:
                    X_test_input = scaler.transform(X_test_data)
                else:
                    X_test_input = X_test_data

                test_predictions = model.predict(X_test_input)

                test_results = test_df.copy()
                test_results['predicted_price_range'] = test_predictions

                st.write(f"### Hasil Prediksi Menggunakan {selected_model_name}")
                st.dataframe(test_results.head(15), use_container_width=True)

                # Tombol Download
                csv_data = test_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Hasil Prediksi (CSV)",
                    data=csv_data,
                    file_name="hasil_prediksi_price_range.csv",
                    mime="text/csv"
                )

                # Distribusi Prediksi
                st.write("### Distribusi Hasil Prediksi")
                fig_pred, ax_pred = plt.subplots(figsize=(7, 4))
                sns.countplot(x='predicted_price_range', hue='predicted_price_range', data=test_results, palette='viridis', legend=False, ax=ax_pred)
                ax_pred.set_title('Distribusi Prediksi Price Range pada Data Uji')
                ax_pred.set_xlabel('Price Range')
                ax_pred.set_ylabel('Jumlah')
                st.pyplot(fig_pred)

else:
    st.info("👋 Silakan unggah file `train.csv` (dan opsional `test.csv`) di sidebar untuk memulai.")

# ==========================================
# TAB 4: Info
# ==========================================
with tab4:
    st.markdown("""
    ### Keterangan Fitur dan Target:
    - **battery_power**: Kapasitas total baterai (mAh)
    - **ram**: Memori RAM (MB)
    - **px_height / px_width**: Resolusi layar
    - **price_range (Target)**:
        - `0`: Low Cost (Murah)
        - `1`: Medium Cost (Menengah)
        - `2`: High Cost (Tinggi)
        - `3`: Very High Cost (Sangat Tinggi)
    """)
