# Changelog - Credit Risk Classifier

## [Fixed] - 2026-06-08

### Masalah yang Ditemukan:
1. **Logika Prediksi Terbalik**: PyCaret prediction_score tidak diinterpretasikan dengan benar
2. **Threshold Terlalu Tinggi**: Nasabah dengan PD 37.55% dan expected loss $1,689 masih APPROVED
3. **Python Version Mismatch**: App berjalan di Python 3.13 padahal PyCaret ada di Python 3.11

### Perbaikan yang Dilakukan:

#### 1. Fix Prediction Logic (`app.py` lines 143-158)
**SEBELUM:**
```python
if pred_label == 1:
    pd_value = pred_score
else:
    pd_value = 1 - pred_score  # SALAH: menghasilkan PD terbalik
```

**SESUDAH:**
```python
# PyCaret 3.x: prediction_score adalah probabilitas kelas positif (1 = default)
if 'prediction_score' in predictions.columns:
    pd_value = predictions['prediction_score'].values[0]
elif 'prediction_score_1' in predictions.columns:
    pd_value = predictions['prediction_score_1'].values[0]
```

#### 2. Recalibrate Risk Thresholds (`src/rules.py` lines 36-41)
**SEBELUM:**
```python
THRESHOLDS = {
    'MACET': 0.80,        # Terlalu tinggi
    'DIRAGUKAN': 0.65,
    'KURANG_LANCAR': 0.50,
    'DPK': 0.40           # PD 37.55% masih lolos
}
```

**SESUDAH:**
```python
THRESHOLDS = {
    'MACET': 0.70,        # Lebih konservatif
    'DIRAGUKAN': 0.50,
    'KURANG_LANCAR': 0.30, # PD >= 30% → REJECT
    'DPK': 0.15           # PD >= 15% → CONDITIONAL
}
```

#### 3. Python Environment Setup
- File `run_streamlit.bat` sudah menggunakan Python 3.11 (credit_env)
- Pastikan selalu jalankan menggunakan: `run_streamlit.bat`
- Jangan gunakan: `streamlit run app.py` langsung (akan pakai Python 3.13)

### Hasil Setelah Perbaikan:
- PD 37.55% → Kol 3 (Kurang Lancar) → **REJECTED** ✅
- PD < 15% → Kol 1 (Lancar) → **APPROVED**
- PD 15-30% → Kol 2 (DPK) → **CONDITIONAL APPROVAL**
- PD >= 30% → Kol 3+ → **REJECTED**

### Cara Menjalankan:
```bash
# Windows: Double-click atau via CMD
run_streamlit.bat

# Atau manual:
C:\Users\User\miniconda3\envs\credit_env\python.exe -m streamlit run app.py
```

### Catatan:
- Model PyCaret (XGBoost) memerlukan Python 3.11
- Base rate default di dataset ~22%, threshold disesuaikan dengan standar perbankan konservatif
- Expected Loss = Loan Amount × PD × LGD (45%)
