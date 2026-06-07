"""
Test Case: Andi - High Income Borrower dengan Riwayat Gagal Bayar
Memverifikasi bahwa sistem tidak memberikan false negative untuk nasabah berisiko.
"""

import sys
import io

# Fix encoding untuk Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from src.rules import hitung_kolektibilitas_ojk

def test_andi_case():
    """
    Profil Andi:
    - Income: $700,000 (sangat tinggi)
    - Loan: $8,000 (sangat kecil, DTI = 1.14%)
    - Riwayat: Pernah gagal bayar (Yes)
    - Tunggakan saat ini: 0 hari

    EXPECTED: Minimal Kol 2 (Conditional Approval) karena riwayat gagal bayar.
    PD seharusnya >= 10% (safety check aktif).
    """

    print("="*60)
    print("TEST CASE: Andi (High Income + Default History)")
    print("="*60)

    # Simulasi PD rendah dari model (misalnya 1% karena income tinggi)
    pd_from_model = 0.01  # 1%

    income = 700000
    loan_amount = 8000
    dti_ratio = loan_amount / income  # 1.14%
    hari_tunggakan = 0
    riwayat_default = 'Yes'

    print(f"\nInput:")
    print(f"  PD dari Model: {pd_from_model:.2%}")
    print(f"  Income: ${income:,}")
    print(f"  Loan Amount: ${loan_amount:,}")
    print(f"  DTI Ratio: {dti_ratio:.2%}")
    print(f"  Riwayat Default: {riwayat_default}")
    print(f"  Hari Tunggakan: {hari_tunggakan}")

    # Panggil fungsi kolektibilitas
    kol_str, decision, decision_color, reason = hitung_kolektibilitas_ojk(
        pd_value=pd_from_model,
        hari_tunggakan=hari_tunggakan,
        riwayat_default=riwayat_default,
        income=income,
        loan_amount=loan_amount,
        dti_ratio=dti_ratio
    )

    print(f"\nHasil:")
    print(f"  Kolektibilitas: {kol_str}")
    print(f"  Keputusan: {decision}")
    print(f"  Alasan: {reason}")

    # Verifikasi
    print(f"\n{'='*60}")
    if "APPROVED" in decision and "CONDITIONAL" not in decision:
        print("[FAIL] Nasabah dengan riwayat default tidak boleh full APPROVED!")
        print("   Seharusnya minimal CONDITIONAL APPROVAL atau REJECTED.")
        return False
    else:
        print("[PASS] Sistem mendeteksi risiko dengan benar.")
        return True

def test_normal_high_income():
    """
    Profil: High income TANPA riwayat gagal bayar
    EXPECTED: Boleh APPROVED jika PD rendah
    """

    print("\n" + "="*60)
    print("TEST CASE: High Income Tanpa Default History (Control)")
    print("="*60)

    pd_from_model = 0.01  # 1%
    income = 700000
    loan_amount = 8000
    dti_ratio = loan_amount / income
    hari_tunggakan = 0
    riwayat_default = 'No'

    print(f"\nInput:")
    print(f"  PD dari Model: {pd_from_model:.2%}")
    print(f"  Riwayat Default: {riwayat_default}")

    kol_str, decision, decision_color, reason = hitung_kolektibilitas_ojk(
        pd_value=pd_from_model,
        hari_tunggakan=hari_tunggakan,
        riwayat_default=riwayat_default,
        income=income,
        loan_amount=loan_amount,
        dti_ratio=dti_ratio
    )

    print(f"\nHasil:")
    print(f"  Kolektibilitas: {kol_str}")
    print(f"  Keputusan: {decision}")

    print(f"\n{'='*60}")
    if "APPROVED" in decision:
        print("[PASS] Nasabah kredibel boleh disetujui.")
        return True
    else:
        print("[FAIL] Nasabah kredibel seharusnya disetujui.")
        return False

if __name__ == "__main__":
    print("\n[TEST] MEMULAI REGRESSION TEST...")
    print("Verifikasi fix untuk kasus Andi (false negative)\n")

    result1 = test_andi_case()
    result2 = test_normal_high_income()

    print("\n" + "="*60)
    print("RINGKASAN TEST")
    print("="*60)
    print(f"Test 1 (Andi - Default History): {'[PASS]' if result1 else '[FAIL]'}")
    print(f"Test 2 (Control - No Default): {'[PASS]' if result2 else '[FAIL]'}")
    print(f"\nOverall: {'[SEMUA TEST PASS]' if (result1 and result2) else '[ADA TEST YANG GAGAL]'}")
    print("="*60)
