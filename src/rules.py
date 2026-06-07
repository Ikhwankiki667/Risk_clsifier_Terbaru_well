def hitung_kolektibilitas_ojk(pd_value, hari_tunggakan, riwayat_default='No'):
    """
    Fungsi Diskrit untuk mengelompokkan risiko ke dalam 5 tingkatan Kolektibilitas (Kol).
    Menggabungkan historis (fakta tunggakan), kebijakan bank (riwayat masa lalu), dan prediktif (Machine Learning).
    """
    # 1. Validasi Input (Keamanan Sistem)
    if not (0.0 <= pd_value <= 1.0):
        raise ValueError("Probabilitas Default (PD) harus berada di rentang 0.0 - 1.0")

    # 2. BLACKLIST RULE DIHAPUS
    # Alasan: Analisis data menunjukkan 'previous_loan_defaults_on_file=Yes'
    # BUKAN indikator gagal bayar. 50.8% nasabah dengan Yes justru 100% lancar.
    # Kolom ini kemungkinan berarti "pernah ada loan record" bukan "pernah default"
    #
    # TIDAK DIGUNAKAN: parameter riwayat_default diabaikan

    # 3. Tentukan Skor Historis (Aturan Baku BI/OJK)
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

    # 4. Tentukan Skor Prediksi AI (Risk Appetite Thresholds)
    # Batas ambang risiko yang lebih konservatif untuk melindungi bank
    # Threshold disesuaikan dengan standar perbankan: reject jika PD > 30%
    THRESHOLDS = {
        'MACET': 0.70,        # PD >= 70% → Macet (sangat tinggi)
        'DIRAGUKAN': 0.50,    # PD >= 50% → Diragukan (tinggi)
        'KURANG_LANCAR': 0.30, # PD >= 30% → Kurang Lancar (menengah-tinggi) - REJECT
        'DPK': 0.15           # PD >= 15% → Dalam Perhatian Khusus (menengah) - CONDITIONAL
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

    # 5. KEPUTUSAN FINAL: Pendekatan Konservatif
    final_kol = max(kol_hist, kol_ai)

    # 6. Kamus Pemetaan Output
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