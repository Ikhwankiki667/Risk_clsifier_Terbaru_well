# Analisis Kasus: Penolakan Andi (TIDAK MASUK AKAL)

**Tanggal Analisis:** 2026-06-08

## Profil Aplikan

| Parameter | Nilai | Assessment |
|-----------|-------|------------|
| Nama | Andi | - |
| Income | $700,000 | **EXCELLENT** (Top 5% earners) |
| Loan Amount | $8,000 | Very Low |
| DTI Ratio | 1.14% | **EXCELLENT** (standar max: 43%) |
| Credit Score | 640 | Fair (580-669 range) |
| Interest Rate | 11% | Medium-High |
| Tunggakan | 0 hari | **EXCELLENT** |

## Keputusan Sistem

- **PD (Probability of Default):** 31.46%
- **Kolektibilitas:** Kol 3 (Kurang Lancar)
- **Keputusan:** 🟠 REJECTED
- **Expected Loss:** $1,132.61

## Masalah: Prediksi TIDAK MASUK AKAL

### 1. Bukti dari Data Historis

**Query:** Profil serupa (income ≥$200K, loan ≤$15K)
- **Total records:** 658
- **Default rate aktual:** **7.29%**
- **Prediksi model untuk Andi:** **31.46%**
- **Selisih:** **4.3x OVERESTIMATE** ❌

**Query:** High income bracket (≥$100K)
- **Total records:** 6,784
- **Default rate aktual:** **6.47%**

### 2. Perbandingan DTI

| Segmen | DTI Range | Andi |
|--------|-----------|------|
| High income historis (≥$200K) | Mean: 3.55%, Max: 7.40% | **1.14%** (lebih baik) |
| Standard max DTI perbankan | 43% | **1.14%** (97% lebih rendah) |

**Kesimpulan:** DTI Andi termasuk **SANGAT EXCELLENT**, namun model memprediksi PD 4x lipat lebih tinggi dari data historis.

### 3. Akar Masalah

#### a. Model Overfitting pada Credit Score
```python
# Credit score 640 (fair range) terlalu dihukum
# Padahal untuk income $700K:
#   - Loan hanya 1.14% dari income
#   - Kapasitas bayar sangat tinggi
#   - Risiko gagal bayar seharusnya MINIMAL
```

#### b. Feature Imbalance
Model tidak cukup mempertimbangkan:
- **Income-to-Loan Ratio** (700K:8K = 87.5x)
- **DTI Ratio** (1.14% vs standar 43%)
- **Debt Service Coverage** (ability to pay)

#### c. Threshold Terlalu Rigid
```python
# src/rules.py:35
'KURANG_LANCAR': 0.30  # PD >= 30% → REJECT
```
Threshold flat 30% tidak mempertimbangkan:
- Kapasitas bayar (income)
- Relatif loan size
- DTI ratio

## Rekomendasi Perbaikan

### Opsi 1: Dynamic Threshold (Berbasis DTI)
```python
def get_dynamic_threshold(dti_ratio):
    """Adjust threshold based on debt-to-income ratio"""
    if dti_ratio < 0.05:  # DTI < 5% (excellent)
        return 0.50  # More tolerant
    elif dti_ratio < 0.15:  # DTI < 15% (good)
        return 0.40
    elif dti_ratio < 0.30:  # DTI < 30% (acceptable)
        return 0.30
    else:  # DTI >= 30% (high)
        return 0.20  # More strict
```

### Opsi 2: Risk-Adjusted PD
```python
def adjust_pd_by_capacity(pd_value, income, loan_amount):
    """Adjust PD based on repayment capacity"""
    loan_to_income = loan_amount / income
    
    if loan_to_income < 0.02:  # Loan < 2% of income
        adjustment_factor = 0.5  # Reduce PD by 50%
    elif loan_to_income < 0.05:  # Loan < 5% of income
        adjustment_factor = 0.7  # Reduce PD by 30%
    else:
        adjustment_factor = 1.0  # No adjustment
    
    return pd_value * adjustment_factor
```

### Opsi 3: Multi-Factor Override
```python
# Override rejection jika semua kondisi terpenuhi:
if (
    pd_value < 0.40 and
    dti_ratio < 0.05 and
    income >= 200000 and
    hari_tunggakan == 0
):
    # Downgrade dari Kol 3 → Kol 2 (Conditional Approval)
    final_kol = 2
```

## Kesimpulan

**Keputusan menolak Andi adalah FALSE POSITIVE yang merugikan bank:**

1. ✅ **Income $700K** → Top earner, sangat mampu bayar
2. ✅ **DTI 1.14%** → 38x lebih baik dari standar (43%)
3. ✅ **Loan $8K** → Hanya 1.14% dari income
4. ✅ **No tunggakan** → Track record bersih
5. ❌ **Credit score 640** → Fair (bukan poor)

**Data historis:** Profil serupa default rate **7.29%**, bukan 31.46%

**Rekomendasi:** Sistem perlu:
- Mempertimbangkan **debt service capacity** (income vs loan)
- Dynamic threshold berbasis DTI
- Feature engineering untuk income-to-loan ratio
- Retrain model dengan weighted features

---

**Status:** BUTUH PERBAIKAN URGENT ⚠️
