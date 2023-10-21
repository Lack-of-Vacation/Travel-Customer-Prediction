# Travel-Customer-Prediction

[Download Dataset](https://www.kaggle.com/datasets/susant4learning/holiday-package-purchase-prediction)

# About Dataset

### Context

Perusahaan "Trips & Travel.Com" berkeinginan untuk memperluas basis pelanggannya dengan memperkenalkan berbagai tawaran paket baru. Saat ini, ada 5 jenis paket yang ditawarkan oleh perusahaan - Basic, Standard, Deluxe, Super Deluxe, dan King. Dari data tahun lalu, 18% pelanggan membeli paket-paket tersebut. Namun, biaya pemasaran cukup tinggi karena pelanggan dihubungi secara acak tanpa mempertimbangkan informasi yang tersedia. Kini, perusahaan berencana meluncurkan produk baru yaitu Wellness Tourism Package. Tujuannya adalah untuk memanfaatkan data yang ada agar biaya pemasaran lebih efisien.

### Content

Dataset ini berisi informasi tentang pelanggan yang berpotensi membeli paket liburan. 

### Tasks To Solve

Analisis data pelanggan diperlukan untuk memberikan rekomendasi kepada pembuat kebijakan dan tim pemasaran. Selain itu, model prediksi perlu dibangun untuk menentukan pelanggan potensial yang akan membeli paket liburan Wellness Tourism Package yang baru diperkenalkan.

# Insight and Recommendation From EDA

### 1. Descriptive Statistics:
Insight:
Ada 8 kolom dengan nilai null, termasuk DurationOfPitch dan TypeOfContact sebagai kolom dengan jumlah nilai null paling banyak.
Outliers dan distribusi yang skew terdeteksi di kolom DurationOfPitch, NumberOfTrips, dan MonthlyIncome.
Terdapat kesalahan entri data pada kolom Gender ("Fe Male") dan ambiguitas di kolom MaritalStatus ("Unmarried").

Rekomendasi Preprocessing:
Untuk kolom dengan nilai null, strategi imputasi perlu diterapkan. Untuk kolom numerik bisa menggunakan median dan untuk kategorikal bisa menggunakan mode.
Kesalahan entri pada kolom Gender harus dikoreksi dari "Fe Male" menjadi "Female".
Gabungkan kategori "Unmarried" ke dalam "Single" pada kolom MaritalStatus.

### 2. Univariate Analysis:
Insight:
Ketidakseimbangan kelas sangat signifikan pada kolom target ProdTaken, ini bisa mempengaruhi performa model.
Kolom DurationOfPitch dan MonthlyIncome memiliki distribusi yang skew ke kanan, menunjukkan adanya outliers yang bisa mempengaruhi model.

Rekomendasi Preprocessing:
Untuk mengatasi ketidakseimbangan kelas pada ProdTaken, teknik SMOTE atau oversampling bisa digunakan.
Transformasi logaritmik atau metode penanganan outliers lainnya perlu diterapkan pada kolom dengan distribusi yang skew.

### 3a. Multivariate Analysis: Korelasi Fitur-Label:
Insight:
Age dan MonthlyIncome menunjukkan korelasi yang lebih tinggi dengan ProdTaken, ini menandakan kedua fitur ini bisa jadi penting dalam pemodelan.

Rekomendasi Preprocessing:
Hapus kolom CustomerID dari dataset karena tidak memberikan informasi yang berguna untuk pemodelan.
Pertahankan fitur seperti Age dan MonthlyIncome yang menunjukkan korelasi tinggi dengan label.

### 3b. Multivariate Analysis: Korelasi antar Fitur:
Insight:
Tidak ada fitur yang menunjukkan korelasi yang sangat tinggi satu sama lain, ini menandakan bahwa setiap fitur memberikan informasi yang berbeda dan bisa jadi penting untuk pemodelan.

Rekomendasi Preprocessing:
Tidak perlu drop fitur hanya berdasarkan korelasi antar-fitur, karena korelasi rendah menunjukkan bahwa fitur-fitur tersebut bisa memberikan informasi yang berbeda dalam model.

### 4. Business Insight:
Insight 1: Produk "Basic" adalah yang paling sering dibeli oleh pelanggan. Ini menunjukkan tingkat penerimaan produk ini yang tinggi di pasar.
Rekomendasi Bisnis: Fokus pemasaran lebih pada produk "Basic" dengan membuat penawaran atau diskon khusus.

Insight 2: Pelanggan yang sudah menikah lebih cenderung untuk membeli produk. Ini bisa jadi karena kebutuhan akan paket liburan keluarga.
Rekomendasi Bisnis: Tawarkan paket khusus keluarga atau diskon untuk menarik segmen pelanggan ini.

Insight 3: Kepemilikan paspor berkolerasi dengan kecenderungan untuk membeli paket. Ini menunjukkan pelanggan ini lebih terbuka untuk paket yang melibatkan perjalanan lintas negara. 
Rekomendasi Bisnis: Buat penawaran khusus untuk pelanggan dengan paspor, seperti diskon atau paket eksklusif.

# Data Preprocessing
