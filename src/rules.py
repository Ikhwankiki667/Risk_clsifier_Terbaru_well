def adjust_pd_by_capacity(pd_value, income, loan_amount, dti_ratio):
    """
    Menyesuaikan PD berdasarkan kapasitas bayar nasabah.

    PENTING: Adjustment sangat konservatif (max 15%) untuk menghormati
    prediksi model ML. Model sudah mempertimbangkan income, jadi adjustment
    besar akan merusak akurasi.

    Args:
        pd_value: Probabilitas default dari model ML
        income: Pendapatan tahunan nasabah
        loan_amount: Jumlah pinjaman
        dti_ratio: Debt-to-Income ratio

    Returns:
        Adjusted PD value
    """
    loan_to_income = loan_amount / income if income > 0 else 1.0

    # Kategori kapasitas bayar - ADJUSTMENT KONSERVATIF
    if dti_ratio < 0.05 and loan_to_income < 0.02:
        # EXCELLENT: DTI <5% dan loan <2% dari income
        adjustment_factor = 0.85  # Kurangi PD max 15%
    elif dti_ratio < 0.10 and loan_to_income < 0.05:
        # VERY GOOD: DTI <10% dan loan <5% dari income
        adjustment_factor = 0.90  # Kurangi PD max 10%
    elif dti_ratio < 0.15:
        # GOOD: DTI <15%
        adjustment_factor = 0.95  # Kurangi PD max 5%
    else:
        # STANDARD: Tidak ada adjustment
        adjustment_factor = 1.0

    adjusted_pd = pd_value * adjustment_factor
    return max(0.01, min(adjusted_pd, 0.99))


def get_dynamic_threshold(dti_ratio):
    """
    Menentukan threshold kolektibilitas berdasarkan DTI ratio.

    REVISI: Threshold adjustment minimal untuk menghormati prediksi model ML.
    Standar industri: PD >30% = HIGH RISK, tidak boleh di-approve.

    Args:
        dti_ratio: Debt-to-Income ratio (loan_amount / income)

    Returns:
        Dictionary threshold untuk setiap kategori
    """
    if dti_ratio < 0.05:
        # DTI <5%: Kapasitas sangat tinggi, sedikit lebih toleran
        return {
            'MACET': 0.70,
            'DIRAGUKAN': 0.50,
            'KURANG_LANCAR': 0.32,  # Sedikit lebih toleran (+2%)
            'DPK': 0.15
        }
    else:
        # DTI ≥5%: Standar konservatif (tidak ada adjustment)
        return {
            'MACET': 0.70,
            'DIRAGUKAN': 0.50,
            'KURANG_LANCAR': 0.30,  # Threshold standar
            'DPK': 0.15
        }


def hitung_kolektibilitas_ojk(pd_value, hari_tunggakan, riwayat_default='No',
                               income=None, loan_amount=None, dti_ratio=None):
    """
    Fungsi Diskrit untuk mengelompokkan risiko ke dalam 5 tingkatan Kolektibilitas (Kol).
    Menggabungkan historis (fakta tunggakan), kebijakan bank (riwayat masa lalu), dan prediktif (Machine Learning).

    UPGRADE: Sekarang mempertimbangkan kapasitas bayar (income, DTI) untuk menghindari false positive.

    Args:
        pd_value: Probabilitas default dari model
        hari_tunggakan: Jumlah hari tunggakan
        riwayat_default: Riwayat default (tidak digunakan)
        income: Pendapatan tahunan (optional, untuk adjustment)
        loan_amount: Jumlah pinjaman (optional, untuk adjustment)
        dti_ratio: Debt-to-Income ratio (optional, untuk dynamic threshold)
    """
    # 1. Validasi Input (Keamanan Sistem)
    if not (0.0 <= pd_value <= 1.0):
        raise ValueError("Probabilitas Default (PD) harus berada di rentang 0.0 - 1.0")

    # 2. RISK-ADJUSTED PD (NEW!)
    # Jika data kapasitas bayar tersedia, sesuaikan PD
    if income is not None and loan_amount is not None and dti_ratio is not None:
        pd_original = pd_value
        pd_value = adjust_pd_by_capacity(pd_value, income, loan_amount, dti_ratio)

        # Log adjustment jika signifikan (>20% perubahan)
        if abs(pd_original - pd_value) / pd_original > 0.20:
            print(f"INFO: Risk Adjustment: PD {pd_original:.2%} -> {pd_value:.2%} (DTI: {dti_ratio:.2%})")

    original_pd = pd_value  # Simpan untuk debug

    # 3. BLACKLIST RULE DIHAPUS
    # Alasan: Analisis data menunjukkan 'previous_loan_defaults_on_file=Yes'
    # BUKAN indikator gagal bayar. 50.8% nasabah dengan Yes justru 100% lancar.
    # Kolom ini kemungkinan berarti "pernah ada loan record" bukan "pernah default"
    #
    # TIDAK DIGUNAKAN: parameter riwayat_default diabaikan

    # 4. Tentukan Skor Historis (Aturan Baku BI/OJK)
    if hari_tunggakan > 180:
        kol_hist = 5  # Macet
    elif hari_tunggakan > 120:
        kol_hist = 4  # Diragukan
    elif hari_tunggakan > 90:
        kol_hist = 3  # Kurang Lancar
    elif hari_tunggakan > 0:
        kol_hist = 2  # Dalam Perhatian Khusus (DPK)
    else:
        kol_hist = 1  # Lancar

    # 5. Tentukan Skor Prediksi AI dengan DYNAMIC THRESHOLD (NEW!)
    # Threshold disesuaikan berdasarkan DTI ratio nasabah
    if dti_ratio is not None:
        THRESHOLDS = get_dynamic_threshold(dti_ratio)
    else:
        # Fallback ke threshold standar konservatif
        THRESHOLDS = {
            'MACET': 0.70,
            'DIRAGUKAN': 0.50,
            'KURANG_LANCAR': 0.30,
            'DPK': 0.15
        }

    if pd_value >= THRESHOLDS['MACET']:
        kol_ai = 5
    elif pd_value >= THRESHOLDS['DIRAGUKAN']:
        kol_ai = 4
    elif pd_value >= THRESHOLDS['KURANG_LANCAR']:
        kol_ai = 3
    elif pd_value >= THRESHOLDS['DPK']:
        kol_ai = 2
    else:
        kol_ai = 1

    # 6. KEPUTUSAN FINAL: Pendekatan Konservatif
    final_kol = max(kol_hist, kol_ai)

    # 7. Kamus Pemetaan Output
    mapping = {
        1: ("Kol 1 (Lancar)", "🟢 APPROVED", "success", "Tidak ada tunggakan berjalan, profil risiko AI sangat rendah."),
        2: ("Kol 2 (Dalam Perhatian Khusus)", "🟡 CONDITIONAL APPROVAL", "warning", "Terdapat riwayat tunggakan 1-90 hari atau peringatan risiko menengah dari AI."),
        3: ("Kol 3 (Kurang Lancar)", "🟠 REJECTED", "error", "Tunggakan 91-120 hari atau profil risiko AI tinggi (Ditolak)."),
        4: ("Kol 4 (Diragukan)", "🔴 REJECTED", "error", "Tunggakan 121-180 hari atau profil risiko AI sangat tinggi (Ditolak)."),
        5: ("Kol 5 (Macet)", "⛔ REJECTED", "error", "Nasabah terindikasi macet permanen secara historis atau dari tebakan AI.")
    }

    # Safe Return (Fallback jika final_kol di luar 1-5)
    return mapping.get(
        final_kol, 
        ("Tidak Diketahui", "⚠️ ERROR", "error", "Terjadi kesalahan pada sistem pemetaan kolektibilitas.")
    )