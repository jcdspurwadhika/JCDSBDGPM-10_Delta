# Bank Marketing Campaign(Smart Telemarketing Strategy): Predicting the Potential of Deposit based on Profile Customers Using Machine Learning to Maximize Bank Profitability

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://qecwrrk6hniuhxnievssnx.streamlit.app/)
[![Tableau Dashboard](https://img.shields.io/badge/Tableau-Public-blue)](https://public.tableau.com/views/FinproDelta-GroupPM10BankMarketingCampaign/BankMarketingCampaignDashboard?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)
![Python](https://img.shields.io/badge/Python-3.12-blue)

**Final Project – Data Science & Machine Learning Program** 
**Group Delta (JCDSBDGPM10+AM08):**
- Alifsya Salam
- Salma Almira Kuswihandono
- Wahyu Eki Sepriansyah

---

## Project Overview

Institusi perbankan seringkali mengandalkan kampanye pemasaran langsung (telemarketing) untuk menawarkan produk Deposito Berjangka (Term Deposit). Namun, strategi memanggil seluruh nasabah sangat tidak efisien dan memakan biaya tinggi, dengan tingkat konversi rata-rata hanya sekitar 11%.  

Project ini membangun **model klasifikasi prediktif** berbasis Machine Learning yang bertujuan untuk menyeleksi calon nasabah potensial, sehingga bank dapat menekan biaya operasional telemarketing (mengurangi *false alarm*) sekaligus memaksimalkan *conversion rate*.

---

## Dataset

Dataset yang digunakan bersumber dari data kampanye institusi perbankan Portugis (UCI Bank Marketing Dataset) yang diakses melalui Kaggle.
- **Kaggle Data Source**: https://www.kaggle.com/datasets/volodymyrgavrysh/bank-marketing-campaigns-dataset
- **Total Data:** 41,176 baris (setelah menghapus 12 baris duplikat)
- **Fitur:** 20 fitur (kolom `duration` dihapus untuk mencegah *data leakage*)
- **Conversion Rate:** ~**11.3%** (Yes) vs **88.7%** (No), menunjukkan data yang sangat *imbalanced* (Rasio 7.9:1).

**Fitur utama**:
- Demografi & Sosial: `age`, `job`, `marital`, `education`.
- Informasi Finansial: `default`, `housing`, `loan`.
- Riwayat Kampanye: `campaign`, `pdays`, `previous`, `poutcome`.
- Indikator Ekonomi Makro: `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`.

---

## Business Objective & Metric

### Objective
1. Membangun model klasifikasi yang memprediksi probabilitas nasabah berlangganan deposito berjangka.
2. Memberikan rekomendasi strategis bagi tim *Sales/Telemarketing* untuk memprioritaskan target kontak.

### Metriks Utama: F1-Score
Dalam konteks data yang sangat *imbalanced* ini, evaluasi difokuskan pada **F1-Score**. F1-Score menjaga keseimbangan harmonis antara:
- **Precision:** Meminimalkan biaya operasional akibat memanggil nasabah yang tidak tertarik (False Positive).
- **Recall:** Meminimalkan kerugian hilangnya potensi pendapatan deposito karena tidak menelepon nasabah potensial (False Negative).

---

## Project Workflow

### 1) Data Preparation & Cleaning
- Pengecekan *missing value* dan duplikasi (menghapus 12 data duplikat).
- Menghapus fitur `duration` karena durasi telepon hanya diketahui setelah transaksi terjadi (*Data Leakage*).

### 2) Exploratory Data Analysis (EDA)
- Analisis *Conversion Rate* berdasarkan riwayat kampanye (`poutcome`), menunjukkan hasil "success" sebelumnya meningkatkan probabilitas hingga 65.1%.
- Distribusi konversi berdasarkan faktor eksternal makroekonomi (korelasi negatif kuat -0.8920 antara konversi dan nilai `euribor3m`).
- Identifikasi titik jenuh panggilan (*Campaign Fatigue*), di mana konversi turun drastis di bawah 5% setelah panggilan ke-3.

### 3) Preprocessing & Feature Engineering
- **Feature Engineering:** Pembuatan fitur turunan seperti `contacted_before`, `previous_success`, `is_success_month`, dan `euribor_low`.
- **Encoding:** Menggunakan `OneHotEncoder` untuk fitur berkardinalitas rendah dan `BinaryEncoder` untuk fitur berkardinalitas tinggi (`job`, `education`) melalui `ColumnTransformer`.
- **Scaling:** Menggunakan `RobustScaler` untuk fitur numerik yang kontinu.

### 4) Modeling (Benchmark)
Model algoritma yang dievaluasi dengan 5-Fold Cross Validation:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- LightGBM
- AdaBoost
- Gradient Boosting
- XGBoost

### 5) Hyperparameter Tuning & Threshold Optimization
- Menggunakan `RandomizedSearchCV` (100 iterasi) pada model terbaik (XGBoost) dengan metrik optimasi F1-Score.
- Pengaturan penyeimbang *class imbalance* menggunakan argumen `scale_pos_weight`.
- Penyesuaian ambang batas klasifikasi menggunakan `TunedThresholdClassifierCV` (didapatkan *Best Threshold*: 0.519).

### 6) Model Interpretability (SHAP)
- Menggunakan **SHAP TreeExplainer** dan *summary plot* untuk melihat dampak fitur (seperti `pdays`, `nr.employed`, `euribor3m`) terhadap probabilitas prediksi model.

---

## Model Performance & Impact

### Cross Validation Benchmark (F1 Score Mean)
- **XGBClassifier: ~0.3802**
- LightGBM: ~0.3791
- Random Forest: ~0.3785
- KNeighborsClassifier: ~0.3700
- Logistic Regression: ~0.3607

### Final Tuned Model Performance
- **Model:** XGBClassifier (Tuned + Threshold Optimization)
- **Best Params:** `learning_rate=0.01`, `max_depth=7`, `n_estimators=533`, `scale_pos_weight=4.23`, `colsample_bytree=0.70`, `subsample=0.75`
- **F1-Score (Train): ~0.542 (54.2%)**
- **F1-Score (Test): ~0.511 (51.1%)**

### Business & Financial Impact
Berdasarkan asumsi matriks biaya (Biaya panggilan: $25, Profit deposito: $100):
- **Strategi Tanpa Model:** Mengalami rugi operasional hingga **-$89,900** (jika menelepon semua nasabah) atau *Opportunity Loss* **-$92,800** (jika tidak menelepon siapapun).
- **Strategi Tuned Model ML:** Model ini berhasil menghasilkan keuntungan bersih (*Net Profit*) sebesar **$24,300**.
- **Marketing ROI:** Kampanye yang dioptimalkan menghasilkan *Return on Investment* sebesar **87.10%**. Model berhasil menyelamatkan nilai ekonomi bank sebesar ~$45,000.

---

## Deployment

### Streamlit App (Prediction)
Aplikasi antarmuka web untuk melakukan klasifikasi potensi nasabah:
- Input data profil dan informasi historis kampanye pelanggan.
- Output hasil klasifikasi nasabah (Deposit / Tidak Deposit).

**Live App:** [Bank Marketing Customer Term Deposit Predictor App](https://qecwrrk6hniuhxnievssnx.streamlit.app/)

### Tableau Dashboard (Monitoring & Insights)
Dashboard analitik interaktif untuk melacak efektivitas kampanye:
- Visualisasi metrik konversi demografis (Pekerjaan, Umur).
- Pemantauan dampak ekonomi makro (Suku bunga Euribor & Consumer Price Index) terhadap tren pembukaan deposito.

**Tableau Public:** [Bank Marketing Campaign Dashboard](https://public.tableau.com/views/FinproDelta-GroupPM10BankMarketingCampaign/BankMarketingCampaignDashboard?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)
