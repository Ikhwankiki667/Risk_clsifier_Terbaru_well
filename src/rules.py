def adjust_pd_by_capacity(pd_value, income, loan_amount, dti_ratio):
    """
    Menyesuaikan PD berdasarkan kapasitas bayar nasabah.

    REVISI v1.3: Adjustment MINIMAL (max 5%) untuk menghormati prediksi model ML.
    Model sudah mempertimbangkan income dalam training, jadi adjustment besar
    akan merusak akurasi dan menyebabkan false negative berbahaya.

    Args:
        pd_value: Probabilitas default dari model ML
        income: Pendapatan tahunan nasabah
        loan_amount: Jumlah pinjaman
        dti_ratio: Debt-to-Income ratio

    Returns:
        Adjusted PD value
    """
    loan_to_income = loan_amount / income if income > 0 else 1.0

    # Kategori kapasitas bayar - ADJUSTMENT SANGAT KONSERVATIF
    if dti_ratio < 0.02 and loan_to_income < 0.01:
        # EXCEPTIONAL: DTI <2% dan loan <1% dari income (kasus ekstrem)
        adjustment_factor = 0.95  # Kurangi PD max 5%
    elif dti_ratio < 0.05 and loan_to_income < 0.02:
        # EXCELLENT: DTI <5% dan loan <2% dari income
        adjustment_factor = 0.97  # Kurangi PD max 3%
    else:
        # STANDARD: Tidak ada adjustment, percaya model ML
        adjustment_factor = 1.0

    adjusted_pd = pd_value * adjustment_factor
    return max(0.01, min(adjusted_pd, 0.99))


def get_dynamic_threshold(dti_ratio):
    """
    Menentukan threshold kolektibilitas berdasarkan DTI ratio.

    REVISI v1.3: Threshold STANDAR untuk semua DTI ratio.
    Dynamic threshold dihapus karena menyebabkan false negative.
    Standar industri: PD >30% = HIGH RISK, tidak boleh di-approve.

    Args:
        dti_ratio: Debt-to-Income ratio (loan_amount / income)

    Returns:
        Dictionary threshold untuk setiap kategori
    """
    # Threshold STANDAR untuk semua nasabah (tidak ada dynamic adjustment)
    return {
        'MACET': 0.70,
        'DIRAGUKAN': 0.50,
        'KURANG_LANCAR': 0.30,
        'DPK': 0.15
    }


def hitung_kolektibilitas_ojk(pd_value, hari_tunggakan, riwayat_default='No',
                               income=None, loan_amount=None, dti_ratio=None):
    """
    Fungsi Diskrit untuk mengelompokkan risiko ke dalam 5 tingkatan Kolektibilitas (Kol).
    Menggabungkan historis (fakta tunggakan), kebijakan bank (riwayat masa lalu), dan prediktif (Machine Learning).

    REVISI v1.3: Risk adjustment minimal + safety check untuk riwayat gagal bayar.

    Args:
        pd_value: Probabilitas default dari model
        hari_tunggakan: Jumlah hari tunggakan
        riwayat_default: Riwayat default ('Yes'/'No')
        income: Pendapatan tahunan (optional, untuk adjustment)
        loan_amount: Jumlah pinjaman (optional, untuk adjustment)
        dti_ratio: Debt-to-Income ratio (optional, untuk dynamic threshold)
    """
    # 1. Validasi Input (Keamanan Sistem)
    if not (0.0 <= pd_value <= 1.0):
        raise ValueError("Probabilitas Default (PD) harus berada di rentang 0.0 - 1.0")

    # 2. RISK-ADJUSTED PD (konservatif, max 5% adjustment)
    # Jika data kapasitas bayar tersedia, sesuaikan PD
    if income is not None and loan_amount is not None and dti_ratio is not None:
        pd_original = pd_value
        pd_value = adjust_pd_by_capacity(pd_value, income, loan_amount, dti_ratio)

        # Log adjustment jika signifikan (>2% perubahan)
        if abs(pd_original - pd_value) / pd_original > 0.02:
            print(f"INFO: Risk Adjustment: PD {pd_original:.2%} -> {pd_value:.2%} (DTI: {dti_ratio:.2%})")

    original_pd = pd_value  # Simpan untuk debug

    # 3. SAFETY CHECK: Riwayat Gagal Bayar
    # Meskipun kolom 'previous_loan_defaults_on_file' tidak reliabel (100% Yes = lancar),
    # jika user MANUAL input "Yes" di form, kita tetap waspada.
    # Minimum PD untuk nasabah dengan riwayat default adalah 10%
    has_default_history = (riwayat_default == 'Yes' or riwayat_default == 'Y')

    if has_default_history and pd_value < 0.10:
        print(f"WARNING: Nasabah dengan riwayat gagal bayar memiliki PD terlalu rendah ({pd_value:.2%}). Dinaikkan ke 10% minimum.")
        pd_value = 0.10

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

    # 5. Tentukan Skor Prediksi AI dengan THRESHOLD STANDAR
    # Threshold tetap untuk semua nasabah (no dynamic adjustment)
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

    # 7. SAFETY ENFORCEMENT: Riwayat default minimum Kol 2
    # Nasabah dengan riwayat gagal bayar tidak boleh dapat Kol 1 (full approval)
    if has_default_history and final_kol < 2:
        print(f"INFO: Nasabah dengan riwayat default di-upgrade dari Kol {final_kol} ke Kol 2 (Conditional Approval)")
        final_kol = 2

    # 8. Kamus Pemetaan Output
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