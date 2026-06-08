"""
NexBank Credit Decision Engine
Menggunakan Model XGBoost dari PyCaret (AUC: 0.9785)
beserta integrasi Business Rules (OJK Collectibility)
"""

import streamlit as st
import pandas as pd
import os
import time
import json
from datetime import datetime
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

# --- IMPORT MODULE LOKAL SELALU DI LUAR TRY-EXCEPT ---
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

# --- INITIALIZE SESSION STATE ---
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []


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


def evaluate_top_algorithms(
    X_train_processed=None,
    y_train=None,
    X_test_processed=None,
    y_test=None,
    X_test_raw=None,
    pycaret_pipeline=None
):
    evaluation_results = []

    # Evaluasi PyCaret model jika tersedia
    if pycaret_pipeline is not None and X_test_raw is not None and y_test is not None:
        try:
            y_pred = pycaret_pipeline.predict(X_test_raw)
            y_prob = pycaret_pipeline.predict_proba(X_test_raw)[:, 1]
        except Exception:
            y_prob = None
        evaluation_results.append({
            'name': 'PyCaret XGBoost',
            **compute_metrics(y_test, y_pred, y_prob)
        })

    # SELALU evaluasi model manual untuk perbandingan
    if X_train_processed is not None and y_train is not None and X_test_processed is not None and y_test is not None:
        candidate_models = [
            ('Random Forest', RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100)),
            ('XGBoost (manual)', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)),
            ('Logistic Regression', LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear', random_state=42))
        ]

        for name, model in candidate_models:
            try:
                model.fit(X_train_processed, y_train)
                y_pred = model.predict(X_test_processed)
                y_prob = model.predict_proba(X_test_processed)[:, 1] if hasattr(model, 'predict_proba') else None
                evaluation_results.append({
                    'name': name,
                    **compute_metrics(y_test, y_pred, y_prob)
                })
            except Exception as e:
                print(f"Warning: Failed to evaluate {name}: {e}")

    evaluation_results.sort(key=lambda entry: entry['roc_auc'], reverse=True)
    _print_evaluation_summary(evaluation_results)

    best_result = evaluation_results[0] if evaluation_results else None
    return evaluation_results, best_result


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

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df['loan_status'],
        random_state=42
    )

    X = df.drop('loan_status', axis=1)
    template_df = X.iloc[[0]].copy()

    if PYCARET_AVAILABLE and os.path.exists("models/best_pycaret_model.pkl"):
        try:
            model = load_model('models/best_pycaret_model')

            # Prepare processed data untuk evaluasi model manual
            preprocessor = DataPreprocessor()
            X_train_processed, y_train = preprocessor.fit_transform(train_df)
            X_test_processed = preprocessor.transform(test_df.drop('loan_status', axis=1))
            y_test = test_df['loan_status']

            # Evaluasi semua model (PyCaret + manual)
            evaluation_results, best_model = evaluate_top_algorithms(
                X_train_processed=X_train_processed,
                y_train=y_train,
                X_test_processed=X_test_processed,
                y_test=y_test,
                X_test_raw=test_df.drop('loan_status', axis=1),
                pycaret_pipeline=model
            )
            return model, template_df, "PyCaret", "Sistem Siap! (Model: XGBoost - AUC 0.9785)", evaluation_results, best_model
        except Exception as e:
            st.warning(f"Gagal load model PyCaret: {e}. Menggunakan Random Forest manual...")

    preprocessor = DataPreprocessor()
    X_train_processed, y_train = preprocessor.fit_transform(train_df)
    X_test_processed = preprocessor.transform(test_df.drop('loan_status', axis=1))
    y_test = test_df['loan_status']

    model = CreditRiskModel()
    model.train(X_train_processed, y_train)
    evaluation_results, best_model = evaluate_top_algorithms(
        X_train_processed=X_train_processed,
        y_train=y_train,
        X_test_processed=X_test_processed,
        y_test=y_test,
        X_test_raw=None,
        pycaret_pipeline=None
    )

    return (preprocessor, model), template_df, "RandomForest", "Sistem Siap! (Model: Random Forest Manual)", evaluation_results, best_model


model_data, template_df, model_type, status_msg, evaluation_results, best_model = load_system()

# --- UI SETUP ---
st.title("🏦 Credit Risk Analysis System")
st.markdown("Sistem Penilaian Risiko Pinjaman Berbasis Machine Learning")
st.markdown(f"**{status_msg}**")

if model_data is None:
    st.error(status_msg)
    st.stop()

st.divider()

# --- CREATE TABS ---
tab_analytics, tab_metrics = st.tabs(["📊 Main Analytics", "📈 Metrics & History"])

# ============================================================================
# TAB 1: MAIN ANALYTICS
# ============================================================================
with tab_analytics:
    st.header("📝 Credit Risk Assessment Form")
    
    # Create form columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        app_name = st.text_input("Nama Aplikan", value="Andi", key="app_name")
        age = st.number_input("Umur (Tahun)", min_value=18, max_value=100, value=30, key="age")
        gender = st.selectbox("Jenis Kelamin", ["male", "female"], key="gender")
        income = st.number_input("Pendapatan Tahunan ($)", min_value=1000, value=70000, step=1000, key="income")
    
    with col2:
        education = st.selectbox("Pendidikan", ["High School", "Associate", "Bachelor", "Master", "Doctorate"], index=2, key="education")
        loan_intent = st.selectbox("Tujuan Pinjaman", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"], key="loan_intent")
        loan_amount = st.number_input("Jumlah Pinjaman ($)", min_value=1000, value=8000, step=1000, key="loan_amount")
        loan_int_rate = st.number_input("Suku Bunga (%)", min_value=1.0, value=11.0, step=0.1, key="loan_int_rate")
    
    with col3:
        emp_length = st.number_input("Lama Bekerja (Tahun)", min_value=0, max_value=50, value=5, key="emp_length")
        home_ownership = st.selectbox("Status Kepemilikan Rumah", ["RENT", "OWN", "MORTGAGE"], index=2, key="home_ownership")
        credit_score = st.number_input("Skor Kredit", min_value=300, max_value=850, value=640, key="credit_score")
        hari_tunggakan = st.number_input("Riwayat Tunggakan (Hari)", min_value=0, value=0, key="hari_tunggakan")
    
    durasi_kredit = st.number_input("Durasi Histori Kredit (Tahun)", min_value=0, value=6, key="durasi_kredit")
    riwayat_default = st.selectbox("Pernah Gagal Bayar Sebelumnya?", ["No", "Yes"], key="riwayat_default")
    
    col_btn_left, col_btn_right = st.columns([2, 2])
    with col_btn_left:
        analyze_btn = st.button("🚀 Jalankan Analisis", type="primary", use_container_width=True)
    with col_btn_right:
        clear_btn = st.button("🗑️ Bersihkan Riwayat", use_container_width=True)
    
    if clear_btn:
        st.session_state.analysis_history = []
        st.success("✓ Riwayat telah dihapus!")
        st.rerun()
    
    st.divider()
    
    # --- ANALYSIS EXECUTION ---
    if analyze_btn:
        with st.spinner("Memproses Analisis Risiko..."):
            time.sleep(0.8)

            dti_ratio = loan_amount / income if income > 0 else 1.0

            # --- DYNAMIC TEMPLATE MATCHING ---
            input_raw = template_df.copy()

            if 'person_age' in input_raw.columns: input_raw['person_age'] = age
            if 'person_gender' in input_raw.columns: input_raw['person_gender'] = gender
            if 'person_education' in input_raw.columns: input_raw['person_education'] = education
            if 'person_income' in input_raw.columns: input_raw['person_income'] = income

            if 'person_emp_exp' in input_raw.columns: input_raw['person_emp_exp'] = emp_length
            elif 'person_emp_length' in input_raw.columns: input_raw['person_emp_length'] = emp_length

            if 'person_home_ownership' in input_raw.columns: input_raw['person_home_ownership'] = home_ownership
            if 'loan_amnt' in input_raw.columns: input_raw['loan_amnt'] = loan_amount
            if 'loan_intent' in input_raw.columns: input_raw['loan_intent'] = loan_intent
            if 'loan_int_rate' in input_raw.columns: input_raw['loan_int_rate'] = loan_int_rate
            if 'loan_percent_income' in input_raw.columns: input_raw['loan_percent_income'] = dti_ratio
            if 'cb_person_cred_hist_length' in input_raw.columns: input_raw['cb_person_cred_hist_length'] = durasi_kredit
            if 'credit_score' in input_raw.columns: input_raw['credit_score'] = credit_score

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
                predictions = predict_model(model_data, data=input_raw)
                pred_label = predictions['prediction_label'].values[0]
                pred_score = predictions['prediction_score'].values[0]
                if pred_label == 1:
                    pd_value = pred_score
                else:
                    pd_value = 1 - pred_score
            else:
                preprocessor, ml_model = model_data
                input_processed = preprocessor.transform(input_raw)
                pd_value = ml_model.predict_default_prob(input_processed)

            pd_value = max(0.01, min(pd_value, 0.99))
            pd_percent = pd_value * 100

            lgd_rate = 0.45
            expected_loss = loan_amount * pd_value * lgd_rate

            # --- OJK RISK DECISION ---
            kol_str, decision, decision_color, reason, pd_adjusted = hitung_kolektibilitas_ojk(
                pd_value=pd_value,
                hari_tunggakan=hari_tunggakan,
                riwayat_default=riwayat_default,
                income=income,
                loan_amount=loan_amount,
                dti_ratio=dti_ratio
            )

            pd_value = pd_adjusted
            pd_percent = pd_value * 100
            expected_loss = loan_amount * pd_value * lgd_rate

            # --- STORE IN SESSION HISTORY ---
            record = {
                "timestamp": datetime.now().isoformat(),
                "applicant": {
                    "name": app_name,
                    "age": age,
                    "gender": gender,
                    "education": education,
                    "income": income,
                    "emp_length": emp_length,
                    "home_ownership": home_ownership,
                    "credit_score": credit_score
                },
                "loan": {
                    "intent": loan_intent,
                    "amount": loan_amount,
                    "interest_rate": loan_int_rate,
                    "dti_ratio": float(dti_ratio)
                },
                "history": {
                    "delinquency_days": hari_tunggakan,
                    "previous_default": riwayat_default,
                    "credit_history_length": durasi_kredit
                },
                "risk_analysis": {
                    "pd_original": float(pd_value),
                    "pd_adjusted": float(pd_adjusted),
                    "collectibility": kol_str,
                    "decision": decision,
                    "reason": reason,
                    "expected_loss": float(expected_loss),
                    "model_type": model_type
                }
            }
            st.session_state.analysis_history.append(record)

            # --- DISPLAY RESULTS ---
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
        st.info("💡 Isi form di atas dan klik **Jalankan Analisis** untuk memulai penilaian risiko kredit.")

# ============================================================================
# TAB 2: METRICS & HISTORY (JSON)
# ============================================================================
with tab_metrics:
    st.header("📈 Model Evaluation Metrics")

    if len(evaluation_results) > 0:
        st.markdown("### 🔍 Top 3 Algorithm Performance")
        st.markdown("Metrik evaluasi dihitung pada data holdout test yang dipisahkan sebelum pelatihan.")

        # Ambil top 3 algoritma berdasarkan ROC AUC (sudah disort di evaluate_top_algorithms)
        top_3_results = evaluation_results[:3]

        # Tampilkan informasi best model
        if best_model is not None:
            st.success(f"🏆 **Best Model:** {best_model['name']} dengan ROC AUC **{best_model['roc_auc']:.4f}**")

        # Buat tabel untuk top 3
        metrics_df = pd.DataFrame(top_3_results)[['name','accuracy','precision','recall','f1','roc_auc']].round(4)
        metrics_df.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
        metrics_df.index = ['🥇 Rank 1', '🥈 Rank 2', '🥉 Rank 3'][:len(metrics_df)]

        st.table(metrics_df)

        # Tampilkan detail masing-masing model
        st.markdown("#### 📊 Detail Performance per Model")
        cols = st.columns(min(3, len(top_3_results)))
        for idx, result in enumerate(top_3_results):
            with cols[idx]:
                rank_emoji = ['🥇', '🥈', '🥉'][idx]
                st.markdown(f"**{rank_emoji} {result['name']}**")
                st.metric("ROC AUC", f"{result['roc_auc']:.4f}")
                st.metric("Accuracy", f"{result['accuracy']:.4f}")
                st.metric("F1-Score", f"{result['f1']:.4f}")
    else:
        st.warning("Tidak ada hasil evaluasi model yang tersedia.")
    
    st.divider()
    st.header("📋 Session Analysis History (JSON)")
    
    if len(st.session_state.analysis_history) == 0:
        st.info("Belum ada hasil analisis dalam sesi ini. Kembali ke tab 'Main Analytics' dan lakukan analisis.")
    else:
        # Display history count
        st.metric("Total Analyses", len(st.session_state.analysis_history))
        
        # Show JSON viewer
        st.subheader("Raw JSON Data")
        json_data = {
            "session_info": {
                "total_analyses": len(st.session_state.analysis_history),
                "export_time": datetime.now().isoformat()
            },
            "analyses": st.session_state.analysis_history
        }
        
        st.json(json_data, expanded=False)
        
        # Download button
        json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
        st.download_button(
            label="📥 Download JSON History",
            data=json_str,
            file_name=f"credit_risk_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        # Display summary table
        st.subheader("Analysis Summary Table")
        summary_data = []
        for record in st.session_state.analysis_history:
            summary_data.append({
                "Waktu": record["timestamp"],
                "Nama Aplikan": record["applicant"]["name"],
                "Pinjaman": f"${record['loan']['amount']:,.0f}",
                "Suku Bunga": f"{record['loan']['interest_rate']:.1f}%",
                "PD": f"{record['risk_analysis']['pd_adjusted']*100:.2f}%",
                "Kolektibilitas": record["risk_analysis"]["collectibility"],
                "Keputusan": record["risk_analysis"]["decision"]
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
