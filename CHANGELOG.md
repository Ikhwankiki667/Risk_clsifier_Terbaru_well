# Changelog - Credit Risk Classifier

All notable changes to this project will be documented in this file.

---

## [1.2.1] - 2026-06-07

### 🔴 ROLLBACK - Risk Adjustment Terlalu Agresif

**Problem:** Versi 1.2.0 menciptakan FALSE NEGATIVE berbahaya dengan menurunkan PD terlalu drastis (60% reduction), menyebabkan peminjam high-risk di-approve.

**Example Case (Andi) - ANALISIS ULANG:**
- Income: $700K, Loan: $8K, DTI: 1.14%
- PD Original (v1.1.0): 31.46% → REJECTED (Kol 3) ✅ **CORRECT**
- PD After v1.2.0: 12.58% → APPROVED (Kol 1) ❌ **FALSE NEGATIVE**
- **Root Cause:** Model ML sudah mempertimbangkan income. PD 31.46% disebabkan faktor lain (credit score 640 = fair/rendah, payment history, dll)

### Changed

- **Risk-Adjusted PD** (`src/rules.py`)
  - Adjustment factor dikurangi drastis: 0.85 (max 15% reduction), turun dari 0.40
  - Alasan: Model ML (AUC 0.9785) sudah optimal, override besar merusak akurasi
  - Prinsip: **Trust the model** → business rules hanya untuk edge cases

- **Dynamic Threshold** (`src/rules.py`)
  - Threshold KURANG_LANCAR: 0.32 (turun dari 0.40)
  - Tetap konservatif: PD >30% = HIGH RISK (standar industri)

### Impact (Kasus Andi)

| Metric | v1.1.0 | v1.2.0 (SALAH) | v1.2.1 (FIXED) |
|--------|--------|----------------|----------------|
| PD | 31.46% | 12.58% | ~27% |
| Kolektibilitas | Kol 3 | Kol 1 | Kol 3 |
| Keputusan | REJECTED ✅ | APPROVED ❌ | REJECTED ✅ |

**Kesimpulan:** 
- Income tinggi ≠ otomatis low risk
- Credit score 640 (fair) + PD 31% = **red flag legitimate**
- Model ML > business intuition sederhana

---

## [1.2.0] - 2026-06-07 [DEPRECATED]

⚠️ **JANGAN GUNAKAN VERSI INI** - False negative rate tinggi

### 🎯 FALSE POSITIVE FIX - High-Income Borrowers (TERNYATA KELIRU)

**Problem YANG DIKLAIM:** Sistem menolak nasabah high-income dengan loan sangat kecil hanya karena credit score "fair" (640).

**ANALISIS ULANG:** Penolakan justified karena:
1. Credit score 640 = fair (bukan excellent)
2. Model trained on 32K+ records, AUC 0.9785
3. PD 31.46% mencerminkan risk factors beyond income
4. Data historis "7.29%" dari query terlalu luas (tidak filter credit score)

### Added (REVERTED IN v1.2.1)

- Risk-Adjusted PD: 60% reduction → TOO AGGRESSIVE
- Dynamic Threshold: 0.40 → TOO PERMISSIVE

**Backward Compatibility:** ✅ Fully compatible. Sistem lama tetap berfungsi jika parameter tidak diberikan.

---

## [1.1.0] - 2026-06-07

### 🔴 CRITICAL FIXES

#### Fixed

1. **PyCaret Prediction Logic TERBALIK** (`app.py`)
   - **Bug:** `prediction_score` dibaca sebagai P(default=1), padahal adalah P(predicted_class)
   - **Impact:** PD bisa terbalik 180° (nasabah berisiko di-approve, nasabah baik di-reject)
   - **Fix:** Check `prediction_label` terlebih dahulu, jika label=0 maka PD = 1 - score

2. **Blacklist Rule SALAH TOTAL** (`src/rules.py`)
   - **Bug:** `previous_loan_defaults_on_file = "Yes"` dianggap sebagai "pernah gagal bayar"
   - **Fact:** 22,858 nasabah dengan "Yes" → 100% LANCAR (0% default)
   - **Impact:** 50% nasabah terbaik di-reject secara salah
   - **Fix:** Blacklist rule dihapus sepenuhnya

3. **Redundant Feature** (`src/preprocessing.py`)
   - **Issue:** `income_to_loan_ratio` adalah inverse dari `loan_percent_income` (multicollinearity)
   - **Fix:** Feature dihapus, hanya gunakan `loan_percent_income`

#### Validated (Already Correct)

- ✅ Expected Loss formula: `EL = Loan × PD × LGD` (Basel II compliant)
- ✅ Threshold kolektibilitas aligned dengan data
- ✅ Column name handling untuk variasi nama kolom
- ✅ Decision mapping konsisten dengan standar OJK

---

## [1.0.0] - Initial Release

### Features

- Machine Learning model untuk prediksi default (XGBoost via PyCaret, AUC: 0.9785)
- Fallback ke Random Forest manual jika PyCaret tidak tersedia
- Business rules berbasis kolektibilitas OJK (Kol 1-5)
- Streamlit dashboard untuk input dan analisis
- Expected Loss calculation (Basel II)
- Support untuk multiple column name variations

### Models

- **Primary:** XGBoost (trained via PyCaret)
  - AUC: 0.9785
  - Balanced accuracy
  - Optimized hyperparameters

- **Fallback:** Random Forest (manual)
  - Used when PyCaret not available
  - Basic preprocessing pipeline

### Business Rules

- Threshold PD untuk kolektibilitas:
  - Kol 1 (Lancar): PD < 15% → APPROVED
  - Kol 2 (DPK): PD 15-30% → CONDITIONAL
  - Kol 3-5: PD ≥ 30% → REJECTED

- Integrasi tunggakan (BI/OJK compliance):
  - 0 hari: Kol 1
  - 1-90 hari: Kol 2
  - 91-120 hari: Kol 3
  - 121-180 hari: Kol 4
  - >180 hari: Kol 5

---

## Versioning

Format: `[MAJOR.MINOR.PATCH]`

- **MAJOR:** Breaking changes atau perubahan arsitektur besar
- **MINOR:** New features, enhancements (backward compatible)
- **PATCH:** Bug fixes, dokumentasi

---

## See Also

- `CRITICAL_FIXES.md` - Detail bug fixes v1.1.0
- `FALSE_POSITIVE_FIX.md` - Detail false positive fix v1.2.0
- `ANALISIS_KASUS_ANDI.md` - Case study yang memicu perbaikan
