# Credit Risk Classifier

Sistem analisis risiko kredit berbasis machine learning untuk memprediksi probabilitas gagal bayar (default) nasabah dan memberikan rekomendasi persetujuan kredit menggunakan aturan kolektibilitas OJK.

## Fitur Utama

- **Model Machine Learning**: XGBoost dengan PyCaret (AUC: 0.9785)
- **Web Dashboard**: Aplikasi Streamlit untuk analisis risiko real-time
- **Sistem Kolektibilitas OJK**: Klasifikasi risiko 5 tingkat (Kol 1-5)
- **Modular Architecture**: Preprocessing, modeling, dan rules terpisah
- **Fallback System**: Random Forest manual jika PyCaret tidak tersedia

## Struktur Proyek

```
Credit-Risk-Classifier/
├── app.py                          # Aplikasi Streamlit (deployment)
├── data/
│   └── loan_data.csv              # Dataset utama
├── models/
│   └── best_pycaret_model.pkl     # Model XGBoost terlatih
├── src/
│   ├── eda.ipynb                  # Exploratory Data Analysis
│   ├── preprocessing.py           # Data cleaning & transformation
│   ├── modeling.py                # Random Forest fallback model
│   └── rules.py                   # Logika kolektibilitas OJK
├── isengisengPyCaret.ipynb        # Training dengan PyCaret AutoML
├── requirements.txt               # Dependencies
└── README.md
```

## Setup untuk Teman

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/Credit-Risk-Classifier.git
cd Credit-Risk-Classifier
```

### 2. Buat Virtual Environment (Opsional tapi Disarankan)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi

Ada 3 cara menggunakan proyek ini:

#### A. Menjalankan Web Dashboard (Recommended)
```bash
streamlit run app.py
```
Dashboard akan terbuka di browser untuk input data nasabah dan analisis risiko real-time.

#### B. Training Model dengan PyCaret
```bash
jupyter notebook isengisengPyCaret.ipynb
```
Jalankan semua cell untuk:
- Training dengan PyCaret AutoML
- Comparing 15+ algoritma
- Hyperparameter tuning
- Menyimpan model terbaik ke `models/`

#### C. Exploratory Data Analysis
```bash
jupyter notebook src/eda.ipynb
```
Analisis data, visualisasi distribusi, dan insight tentang dataset.

## Dataset

**File**: `data/loan_data.csv`
- Fitur: `person_age`, `person_income`, `person_emp_length`, `loan_amnt`, `loan_intent`, `loan_int_rate`, `credit_score`, dll
- Target: `loan_status` (0 = Lancar, 1 = Gagal Bayar)
- Total sampel: ~32,000 records
- Handling imbalanced data dengan SMOTE dan class weighting

## Model Performance

**Model Terbaik: XGBoost Classifier (via PyCaret)**
- **AUC Score**: 0.9785
- **Accuracy**: 93%+
- **Recall untuk kelas Gagal Bayar**: Optimal untuk mendeteksi nasabah berisiko
- Training time: ~2-3 menit dengan AutoML

**Fallback Model: Random Forest**
- Digunakan jika PyCaret tidak tersedia
- `class_weight='balanced'` untuk menangani imbalanced data
- Accuracy: ~85-88%

## Sistem Kolektibilitas OJK

Aplikasi menggunakan aturan Bank Indonesia/OJK untuk klasifikasi risiko:

| Kolektibilitas | Hari Tunggakan | Probabilitas AI | Status |
|----------------|----------------|-----------------|---------|
| Kol 1 (Lancar) | 0 | < 8% | ✅ APPROVED |
| Kol 2 (DPK) | 1-90 | 8-15% | ⚠️ CONDITIONAL |
| Kol 3 (Kurang Lancar) | 91-120 | 15-30% | ❌ REJECTED |
| Kol 4 (Diragukan) | 121-180 | 30-50% | ❌ REJECTED |
| Kol 5 (Macet) | > 180 | > 50% | ⛔ REJECTED |

Sistem mengambil **skor terburuk** antara historis tunggakan dan prediksi AI untuk mitigasi risiko maksimal.

## Requirements

- Python 3.8+
- RAM minimal 4GB (untuk PyCaret training)
- Dependencies: PyCaret, Scikit-learn, Streamlit, Plotly, Pandas, NumPy

## Troubleshooting

### Error: `ModuleNotFoundError`
```bash
pip install -r requirements.txt
```

### Error: PyCaret installation
Jika ada masalah dengan PyCaret, install secara manual:
```bash
pip install pycaret[full]
```

### Error: Model tidak ditemukan
Jika `models/best_pycaret_model.pkl` tidak ada:
1. Jalankan `isengisengPyCaret.ipynb` untuk training model
2. Atau biarkan `app.py` otomatis training Random Forest fallback

### Streamlit tidak bisa diakses
```bash
pip install streamlit plotly
streamlit run app.py
```

### Jupyter Kernel tidak ditemukan
```bash
python -m ipykernel install --user --name=venv
```

## Author

Tugas kuliah - Credit Risk Analysis dengan Machine Learning dan OJK Compliance
