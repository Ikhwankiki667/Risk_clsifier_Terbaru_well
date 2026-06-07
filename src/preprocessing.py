import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

class DataPreprocessor:
    """
    Kelas Modular untuk membersihkan dan mentransformasi data mentah
    menjadi format yang bisa dibaca oleh algoritma Machine Learning.
    Kini dilengkapi dengan fitur pembersihan anomali data otomatis saat training.
    """
    def __init__(self):
        self.preprocessor = None
        self.numeric_features = []
        self.categorical_features = []

    def fit_transform(self, df):
        """
        Mempelajari pola data (fit) sekaligus mengubahnya (transform).
        Dipanggil HANYA SAAT TRAINING.

        Melakukan pembersihan data (data cleaning) terhadap anomali logis:
        1. Menghapus baris jika usia nasabah > 100 tahun.
        2. Menghapus baris jika pengalaman kerja > 60 tahun.
        3. Menghapus baris jika pengalaman kerja >= usia nasabah itu sendiri.

        OPTIMASI BARU:
        4. Outlier capping untuk income dan loan amount (99th percentile)
        5. Feature engineering: debt_burden, income_to_loan_ratio
        6. Log transformation untuk fitur skewed (income, age, emp_exp)
        """
        # --- TAHAP 1: PEMBERSIHAN DATA (AUTOMATIC CLEANING) ---
        df_clean = df.copy()
        rows_before = len(df_clean)

        # Deteksi otomatis kolom pengalaman kerja (bisa berupa 'person_emp_exp' atau 'person_emp_length')
        if 'person_emp_exp' in df_clean.columns:
            emp_col = 'person_emp_exp'
        elif 'person_emp_length' in df_clean.columns:
            emp_col = 'person_emp_length'
        else:
            emp_col = None

        # 1. Bersihkan usia tidak rasional
        if 'person_age' in df_clean.columns:
            df_clean = df_clean[df_clean['person_age'] <= 100]

        # 2. Bersihkan masa kerja tidak rasional
        if emp_col and emp_col in df_clean.columns:
            df_clean = df_clean[df_clean[emp_col] <= 60]

        # 3. Bersihkan jika masa kerja melebihi umur nasabah
        if 'person_age' in df_clean.columns and emp_col and emp_col in df_clean.columns:
            df_clean = df_clean[df_clean[emp_col] < df_clean['person_age']]

        # VALIDASI: Log berapa banyak data yang dibersihkan
        rows_after = len(df_clean)
        pct_removed = (1 - rows_after/rows_before) * 100 if rows_before > 0 else 0
        if pct_removed > 10:
            print(f"⚠️ Warning: {pct_removed:.1f}% data dihapus saat cleaning ({rows_before} → {rows_after})")

        # --- TAHAP 1B: OUTLIER CAPPING (NEW!) ---
        # Cap nilai ekstrem ke 99th percentile untuk mencegah outlier merusak model
        for col in ['person_income', 'loan_amnt']:
            if col in df_clean.columns:
                upper_limit = df_clean[col].quantile(0.99)
                df_clean[col] = df_clean[col].clip(upper=upper_limit)

        # --- TAHAP 1C: FEATURE ENGINEERING (NEW!) ---
        # Buat fitur turunan yang memiliki prediksi power lebih tinggi

        # 1. Income to Loan Ratio (semakin tinggi = semakin aman)
        if 'person_income' in df_clean.columns and 'loan_amnt' in df_clean.columns:
            df_clean['income_to_loan_ratio'] = df_clean['person_income'] / (df_clean['loan_amnt'] + 1)

        # 2. Debt Burden (beban cicilan per tahun)
        if all(col in df_clean.columns for col in ['loan_amnt', 'loan_int_rate', 'person_income']):
            df_clean['debt_burden'] = (df_clean['loan_amnt'] * df_clean['loan_int_rate'] / 100) / (df_clean['person_income'] + 1)

        # --- TAHAP 1D: LOG TRANSFORMATION FOR SKEWED FEATURES (NEW!) ---
        # Transformasi log untuk menormalkan distribusi yang sangat skewed
        skewed_features = ['person_income', 'person_age']
        if emp_col:
            skewed_features.append(emp_col)

        for col in skewed_features:
            if col in df_clean.columns:
                df_clean[f'{col}_log'] = np.log1p(df_clean[col])

        # --- TAHAP 2: PEMISAHAN FITUR & TARGET ---
        # Validasi kolom target
        if 'loan_status' not in df_clean.columns:
            raise ValueError("Kolom target 'loan_status' tidak ditemukan di dataset!")

        # Pisahkan fitur (X) dan target (y) menggunakan data yang sudah disterilkan
        X = df_clean.drop('loan_status', axis=1)
        y = df_clean['loan_status']

        self.numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Pipeline untuk angka: Isi data kosong dengan median, lalu standarisasi (Scaling)
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Pipeline untuk teks: Isi data kosong dengan modus, lalu ubah jadi angka (OneHot)
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Gabungkan kedua pipeline
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, self.numeric_features),
                ('cat', cat_transformer, self.categorical_features)
            ])

        # Eksekusi pemrosesan data X
        X_processed = self.preprocessor.fit_transform(X)
        
        return X_processed, y

    def transform(self, df_input):
        """
        Mengubah data baru berdasarkan pola yang sudah dipelajari.
        Dipanggil SAAT PREDIKSI NASABAH BARU (Inference).
        Pada tahap ini kita TIDAK membuang baris data agar aplikasi tidak crash.

        PENTING: Feature engineering harus dilakukan dengan cara yang sama seperti saat training!
        """
        if self.preprocessor is None:
            raise Exception("Preprocessor belum di-fit dengan data training!")

        # Buat copy untuk tidak merusak input original
        df_transformed = df_input.copy()

        # Deteksi kolom pengalaman kerja
        if 'person_emp_exp' in df_transformed.columns:
            emp_col = 'person_emp_exp'
        elif 'person_emp_length' in df_transformed.columns:
            emp_col = 'person_emp_length'
        else:
            emp_col = None

        # --- REPLIKASI FEATURE ENGINEERING (HARUS SAMA DENGAN fit_transform!) ---

        # 1. Income to Loan Ratio
        if 'person_income' in df_transformed.columns and 'loan_amnt' in df_transformed.columns:
            df_transformed['income_to_loan_ratio'] = df_transformed['person_income'] / (df_transformed['loan_amnt'] + 1)

        # 2. Debt Burden
        if all(col in df_transformed.columns for col in ['loan_amnt', 'loan_int_rate', 'person_income']):
            df_transformed['debt_burden'] = (df_transformed['loan_amnt'] * df_transformed['loan_int_rate'] / 100) / (df_transformed['person_income'] + 1)

        # 3. Log Transformation
        skewed_features = ['person_income', 'person_age']
        if emp_col:
            skewed_features.append(emp_col)

        for col in skewed_features:
            if col in df_transformed.columns:
                df_transformed[f'{col}_log'] = np.log1p(df_transformed[col])

        return self.preprocessor.transform(df_transformed)