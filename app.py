"""
NexBank Credit Decision Engine
Menggunakan Model XGBoost dari PyCaret (AUC: 0.9785)
beserta integrasi Business Rules (OJK Collectibility)
"""

import streamlit as st
import pandas as pd
import os
import time
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

# --- IMPORT MODULE LOKAL SELALU DI LUAR TRY-EXCEPT ---
# Agar sistem Fallback selalu kenal dengan fungsi-fungsi ini
from src.rules import hitung_kolektibilitas_ojk
from src.preprocessing import DataPreprocessor
from src.modeling import CreditRiskModel

# Import PyCaret untuk load model
try:
    from pycaret.classification import load_model, predict_model
    PYCARET_AVAILABLE = True
except ImportError:
    PYCARET_AVAILABLE = False

st.set_page_config(page_title="Credit Risk Analysis System", page_icon="🏦", layout="wide")


def compute_metrics(y_true, y_pred, y_prob):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob) if y_prob is not None else 0.0
    }


def _print_evaluation_summary(results):
    print("\n" + "="*60)
    print(" EVALUASI TOP 3 ALGORITMA PYCARET / MANUAL")
    print("="*60)
    for result in results:
        print(
            f"{result['name']} -> Accuracy: {result['accuracy']:.4f}, "
            f"Precision: {result['precision']:.4f}, Recall: {result['recall']:.4f}, "
            f"F1: {result['f1']:.4f}, ROC AUC: {result['roc_auc']:.4f}"
        )
    print("="*60 + "\n")


def evaluate_top_algorithms(df, pycaret_pipeline=None):
    df = df.dropna(subset=['loan_status']).copy()
    raw_X = df.drop('loan_status', axis=1)
    y = df['loan_status']

    preprocessor = DataPreprocessor()
    X_processed, y_processed = preprocessor.fit_transform(df)

    evaluation_results = []

    if pycaret_pipeline is not None:
        try:
            y_pred = pycaret_pipeline.predict(raw_X)
            y_prob = pycaret_pipeline.predict_proba(raw_X)[:, 1]
        except Exception:
            y_prob = None
        evaluation_results.append({
            'name': 'PyCaret XGBoost',
            **compute_metrics(y, y_pred, y_prob)
        })

    candidate_models = [
        ('Random Forest', RandomForestClassifier(class_weight='balanced', random_state=42)),
        ('XGBoost (manual)', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)),
        ('Logistic Regression', LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear', random_state=42))
    ]

    for name, model in candidate_models:
        model.fit(X_processed, y_processed)
        y_pred = model.predict(X_processed)
        y_prob = model.predict_proba(X_processed)[:, 1] if hasattr(model, 'predict_proba') else None
        evaluation_results.append({
            'name': name,
            **compute_metrics(y_processed, y_pred, y_prob)
        })

    # Pastikan urutan tertinggi berdasarkan ROC AUC
    evaluation_results.sort(key=lambda entry: entry['roc_auc'], reverse=True)
    _print_evaluation_summary(evaluation_results)

    best_result = evaluation_results[0] if evaluation_results else None
    return evaluation_results, best_result

# --- 1. TAHAP LOAD MODEL ---
@st.cache_resource
def load_system():
    """Load model PyCaret atau fallback ke Random Forest manual."""

    file_path = "loan_data.csv"
    if not os.path.exists(file_path) and os.path.exists("data/loan_data.csv"):
        file_path = "data/loan_data.csv"

    try:
        df = pd.read_csv(file_path)
        df = df.dropna(subset=['loan_status'])
    except FileNotFoundError:
        return None, None, None, "File loan_data.csv tidak ditemukan!", [], None

    X = df.drop('loan_status', axis=1)
    template_df = X.iloc[[0]].copy()

    if PYCARET_AVAILABLE and os.path.exists("models/best_pycaret_model.pkl"):
        try:
            model = load_model('models/best_pycaret_model')
            evaluation_results, best_model = evaluate_top_algorithms(df, pycaret_pipeline=model)
            return model, template_df, "PyCaret", "Sistem Siap! (Model: XGBoost - AUC 0.9785)", evaluation_results, best_model
        except Exception as e:
            st.warning(f"Gagal load model PyCaret: {e}. Menggunakan Random Forest manual...")

    preprocessor = DataPreprocessor()
    X_processed, y = preprocessor.fit_transform(df)

    model = CreditRiskModel()
    model.train(X_processed, y)
    evaluation_results, best_model = evaluate_top_algorithms(df, pycaret_pipeline=None)

    return (preprocessor, model), template_df, "RandomForest", "Sistem Siap! (Model: Random Forest Manual)", evaluation_results, best_model

model_data, template_df, model_type, status_msg, evaluation_results, best_model = load_system()

# --- 2. TAHAP UI & INPUT DASHBOARD ---
st.title("🏦 Credit Risk Analysis System")
st.markdown("Sistem Penilaian Risiko Pinjaman Berbasis Machine Learning")
st.markdown(f"**{status_msg}**")

if best_model is not None:
    st.markdown("### 🔍 Ringkasan Evaluasi Model")
    st.markdown(f"**Algoritma Terbaik:** {best_model['name']} dengan ROC AUC **{best_model['roc_auc']:.3f}**")
    st.markdown("Model metrics dihitung pada data training / dataset yang tersedia.")
    if len(evaluation_results) > 0:
        metrics_df = pd.DataFrame(evaluation_results)[['name','accuracy','precision','recall','f1','roc_auc']].round(4)
        metrics_df.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']
        st.table(metrics_df)

st.divider()

if model_data is None:
    st.error(status_msg)
    st.stop()

# Sidebar untuk Input
with st.sidebar:
    st.header("📝 Form Input Data")
    app_name = st.text_input("Nama Aplikan", value="Andi")
    age = st.number_input("Umur (Tahun)", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Jenis Kelamin", ["male", "female"])
    education = st.selectbox("Pendidikan", ["High School", "Associate", "Bachelor", "Master", "Doctorate"], index=2)
    income = st.number_input("Pendapatan Tahunan ($)", min_value=1000, value=70000, step=1000)
    loan_intent = st.selectbox("Tujuan Pinjaman", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
    loan_amount = st.number_input("Jumlah Pinjaman ($)", min_value=1000, value=8000, step=1000)
    loan_int_rate = st.number_input("Suku Bunga (%)", min_value=1.0, value=11.0, step=0.1)

    st.markdown("---")
    st.subheader("Data Tambahan")
    emp_length = st.number_input("Lama Bekerja (Tahun)", min_value=0, max_value=50, value=5)
    home_ownership = st.selectbox("Status Kepemilikan Rumah", ["RENT", "OWN", "MORTGAGE"], index=2)
    credit_score = st.number_input("Skor Kredit", min_value=300, max_value=850, value=640)
    hari_tunggakan = st.number_input("Riwayat Tunggakan (Hari)", min_value=0, value=0)
    durasi_kredit = st.number_input("Durasi Histori Kredit (Tahun)", min_value=0, value=6)
    riwayat_default = st.selectbox("Pernah Gagal Bayar Sebelumnya?", ["No", "Yes"])

    analyze_btn = st.button("🚀 Jalankan Analisis", type="primary", use_container_width=True)

# --- 3. TAHAP DEPLOYMENT / PREDIKSI ---
if analyze_btn:
    with st.spinner("Memproses Analisis Risiko..."):
        time.sleep(0.8)

        dti_ratio = loan_amount / income if income > 0 else 1.0

        # --- DYNAMIC TEMPLATE MATCHING ---
        input_raw = template_df.copy()

        # Timpa nilainya satu per satu HANYA JIKA kolomnya ada
        if 'person_age' in input_raw.columns: input_raw['person_age'] = age
        if 'person_gender' in input_raw.columns: input_raw['person_gender'] = gender
        if 'person_education' in input_raw.columns: input_raw['person_education'] = education
        if 'person_income' in input_raw.columns: input_raw['person_income'] = income

        # Mengatasi nama kolom pengalaman kerja yang berbeda
        if 'person_emp_exp' in input_raw.columns: input_raw['person_emp_exp'] = emp_length
        elif 'person_emp_length' in input_raw.columns: input_raw['person_emp_length'] = emp_length

        if 'person_home_ownership' in input_raw.columns: input_raw['person_home_ownership'] = home_ownership
        if 'loan_amnt' in input_raw.columns: input_raw['loan_amnt'] = loan_amount
        if 'loan_intent' in input_raw.columns: input_raw['loan_intent'] = loan_intent
        if 'loan_int_rate' in input_raw.columns: input_raw['loan_int_rate'] = loan_int_rate
        if 'loan_percent_income' in input_raw.columns: input_raw['loan_percent_income'] = dti_ratio
        if 'cb_person_cred_hist_length' in input_raw.columns: input_raw['cb_person_cred_hist_length'] = durasi_kredit
        if 'credit_score' in input_raw.columns: input_raw['credit_score'] = credit_score

        # Mengatasi kolom riwayat gagal bayar (mengambil dari input form)
        if 'cb_person_default_on_file' in input_raw.columns:
            input_raw['cb_person_default_on_file'] = 'Y' if riwayat_default == 'Yes' else 'N'
        elif 'previous_loan_defaults_on_file' in input_raw.columns:
            input_raw['previous_loan_defaults_on_file'] = riwayat_default

        if riwayat_default == 'Yes' or hari_tunggakan > 0:
            warning_text = []
            if riwayat_default == 'Yes':
                warning_text.append("Nasabah memiliki riwayat gagal bayar sebelumnya dan harus diperlakukan konservatif sesuai prinsip OJK/BI.")
            if hari_tunggakan > 0:
                warning_text.append(f"Nasabah memiliki tunggakan saat ini sebesar {hari_tunggakan} hari, yang secara langsung memengaruhi kolektibilitas.")
            st.warning(" ".join(warning_text))

        # PREDIKSI berdasarkan tipe model
        if model_type == "PyCaret":
            # PyCaret model: gunakan predict_model
            predictions = predict_model(model_data, data=input_raw)

            # CRITICAL FIX: prediction_score adalah probabilitas untuk kelas yang DIPREDIKSI
            # Bukan selalu probabilitas untuk kelas 1 (default)
            pred_label = predictions['prediction_label'].values[0]
            pred_score = predictions['prediction_score'].values[0]

            # Jika diprediksi default (label=1), score sudah benar sebagai PD
            # Jika diprediksi lancar (label=0), PD = 1 - score
            if pred_label == 1:
                pd_value = pred_score
            else:
                pd_value = 1 - pred_score

        else:
            # Random Forest manual
            preprocessor, ml_model = model_data
            input_processed = preprocessor.transform(input_raw)
            pd_value = ml_model.predict_default_prob(input_processed)

        pd_value = max(0.01, min(pd_value, 0.99))
        pd_percent = pd_value * 100

        # Menghitung Expected Loss
        lgd_rate = 0.45
        expected_loss = loan_amount * pd_value * lgd_rate

        # --- LOGIKA KEPUTUSAN OJK (MEMANGGIL rules.py) ---
        # UPGRADE: Sekarang memasukkan income, loan_amount, dan dti_ratio
        # untuk risk adjustment dan dynamic threshold
        kol_str, decision, decision_color, reason, pd_adjusted = hitung_kolektibilitas_ojk(
            pd_value=pd_value,
            hari_tunggakan=hari_tunggakan,
            riwayat_default=riwayat_default,
            income=income,
            loan_amount=loan_amount,
            dti_ratio=dti_ratio
        )

        # Update PD dengan nilai yang sudah di-adjust (termasuk safety check)
        pd_value = pd_adjusted
        pd_percent = pd_value * 100

        # Recalculate Expected Loss dengan PD yang sudah di-adjust
        expected_loss = loan_amount * pd_value * lgd_rate

        # --- 4. TATA LETAK HASIL ---
        st.markdown("### Detail Aplikan")
        st.markdown(f"""
        <div style='background-color: #2c3e50; padding: 20px; border-radius: 8px; color: white; margin-bottom: 25px;'>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px;'>
                <div><strong>Nama:</strong> {app_name}</div>
                <div><strong>Tujuan Pinjaman:</strong> {loan_intent}</div>
                <div><strong>Umur:</strong> {age}</div>
                <div><strong>Jumlah Pinjaman:</strong> ${loan_amount:,.2f}</div>
                <div><strong>Pendapatan:</strong> ${income:,.2f}</div>
                <div><strong>Suku Bunga Pinjaman:</strong> {loan_int_rate}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if decision_color == "success":
            st.success(f"**Rekomendasi: {decision} ({kol_str})**\n\n*Alasan: {reason}*")
        elif decision_color == "warning":
            st.warning(f"**Rekomendasi: {decision} ({kol_str})**\n\n*Alasan: {reason}*")
        else:
            st.error(f"**Rekomendasi: {decision} ({kol_str})**\n\n*Alasan: {reason}*")

        col_text, col_chart = st.columns([1, 1])

        with col_text:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color: #c0392b;'>Potensi Kerugian (Expected Loss): ${expected_loss:,.2f}</h3>", unsafe_allow_html=True)
            st.markdown(f"**Probabilitas Gagal Bayar (PD): {pd_percent:.2f}%**")

        with col_chart:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = pd_percent,
                number = {'suffix': "%", 'font': {'size': 40, 'color': '#2c3e50'}},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': "rgba(41, 128, 185, 0.8)"},
                    'steps': [
                        {'range': [0, 15], 'color': "rgba(46, 204, 113, 0.4)"},
                        {'range': [15, 30], 'color': "rgba(241, 196, 15, 0.4)"},
                        {'range': [30, 100], 'color': "rgba(231, 76, 60, 0.4)"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': pd_percent
                    }
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

        st.info(f"Aplikan bernama **{app_name}** memiliki probabilitas gagal bayar sebesar **{pd_percent:.2f}%**. Keputusan akhir sistem: **{decision} ({kol_str})**.")

else:
    st.write("Silakan isi form di *Sidebar* sebelah kiri dan klik **Jalankan Analisis** untuk melihat detail profil risiko nasabah.")