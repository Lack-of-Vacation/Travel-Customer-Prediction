# Travel-Customer-Prediction

[Download Dataset](https://www.kaggle.com/datasets/susant4learning/holiday-package-purchase-prediction)

# About Dataset

### Context

Perusahaan "Trips & Travel.Com" berkeinginan untuk memperluas basis pelanggannya dengan memperkenalkan berbagai tawaran paket baru. Saat ini, ada 5 jenis paket yang ditawarkan oleh perusahaan - Basic, Standard, Deluxe, Super Deluxe, dan King. Dari data tahun lalu, 18% pelanggan membeli paket-paket tersebut. Namun, biaya pemasaran cukup tinggi karena pelanggan dihubungi secara acak tanpa mempertimbangkan informasi yang tersedia. Kini, perusahaan berencana meluncurkan produk baru yaitu Wellness Tourism Package. Tujuannya adalah untuk memanfaatkan data yang ada agar biaya pemasaran lebih efisien.

### Content

Dataset ini berisi informasi tentang pelanggan yang berpotensi membeli paket liburan. 

### Tasks To Solve

Analisis data pelanggan diperlukan untuk memberikan rekomendasi kepada pembuat kebijakan dan tim pemasaran. Selain itu, model prediksi perlu dibangun untuk menentukan pelanggan potensial yang akan membeli paket liburan Wellness Tourism Package yang baru diperkenalkan.

# 1. Insight and Recommendation From EDA

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

# 2. Data Preprocessing

## 1. Data Cleansing:
### A. Handle missing values
  Mengatasi missing value dalam dataset sangat penting karena nilai yang hilang dapat mengakibatkan bias dan ketidakakuratan dalam hasil analisis. Kami mengatasi missing value dalam dataset dengan melakukan imputasi. Hal ini dilakukan karena persentasenya kecil. Selanjutnya, kami menggunakan nilai mean untuk data numerik dengan distribusi yang normal (Age, NumberOfFollowUps, NumberOfChildrenVisiting), nilai median untuk data numerik dengan distribusi yang tidak normal atau skew (DurationOfPitch, NumberOfTrips, MonthlyIncome), dan nilai modus untuk data kategorik (TypeOfContact, PreferedPropertyStar). Dengan metode ini, dataset menjadi lebih lengkap dan siap untuk analisis lebih lanjut.

### B. Handle duplicated data
- Handle Duplicated Data sangat penting karena data duplikasi dapat mempengaruhi akurasi data serta mengganggu efisiensi sistem pengolahan data.  Menghandle duplikasi data meningkatkan akurasi, efisiensi, keamanan, dan kemampuan organisasi/perusahaan untuk mengambil keputusan berdasarkan data yang tepat dan bersih.
- Diketahui dalam dataframe tidak terdapat duplikasi data, sehingga menghandle kesalahan pada kolom Gender (Fe Male digabungkan ke Female), serta handle kolom MaritalStatus (Unmarried digabungkan ke Single). 
- Pada kolom Gender, terdapat kesalahan input yaitu 'Fe Male'. Sedangkan terdapat juga jawaban 'Female' sehingga untuk mencegah duplikat data 'Fe Male' digabungkan ke 'Female'. 
- Pada kolom MaritalStatus, terdapat data yang tidak konsisten yaitu Unmarried, sehingga Unmarried digabungkan ke Single karena keduanya merujuk kepada istilah yang sama, yaitu lajang.

### C. Handle outliers
- Dari EDA yang sudah dilakukan pada minggu sebelumnya, diketahui bahwa kolom DurationOfPitch, NumberOfTrips, MonthlyIncome memiliki outliers.
- Kolom MonthlyIncome memiliki distribusi yang kemiringannya (skew) cukup signifikan, maka dilakukan log-transform agar distribusinya lebih simetris.
- Mengidentifikasi outliers menggunakan metode z-score dan menetapkan outliers rendah dan tinggi sebagai acuan dalam meng-handle outliers.
- Mengganti nilai outliers dengan ketentuan: Nilai outliers rendah diganti menjadi nilai batas bawah yang belum termasuk outlier dan nilai outliers tinggi diganti menjadi nilai batas atas yang belum termasuk outlier.

### D. Feature transformation
  Transformasi fitur logaritma (log transform) adalah teknik yang berguna untuk mengatasi distribusi data yang cenderung condong (skewed) dan untuk mengurangi efek outlier. Pada dataset kami dilakukan Log Transform pada DurationOfPitch, NumberOfTrips, yang dimana log transformation itu digunakan pada data yang right-skewed setelah dilakukan feature transformation maka distribusi hasil transformasi akan mendekati distribusi normal jika dilihat dari hasil grafiknya. Selanjutnya, menggunakan StandardScaler dari scikit-learn untuk melakukan standarisasi pada kolom numerik dalam DataFrame yang disebut df_prep. Standarisasi mengubah data sehingga memiliki rata-rata 0 dan deviasi standar 1. 

### E. Feature encoding
  Dalam Feature encoding, pendekatan yang digunakan beragam tergantung pada jenis data dan karakteristik masing-masing kolom: 
- Kolom Gender memiliki dua nilai unik ('Female' dan 'Male'), sehingga kita dapat gunakan Label Encoding untuk mengubah nilai-nilai ini menjadi angka (0 dan 1).
- Kolom ProductPitched dan kolom Designation memiliki nilai yang berjenis data ordinal, sehingga kita dapat menggunakan Label Encoding untuk mengubah nilai-nilai ini menjadi angka sesuai urutannya.
- Kolom TypeofContact, Occupation dan MaritalStatus memiliki lebih dari dua nilai unik dan jenis datanya tidak ordinal. Maka akan dilakukan One Hot Encoding.
- Kolom Occupation terdapat pelanggan dengan value Free Lancer dengan jumlah 2,yang dimana di kolom Occupation terdapat value Salaried, Small Business, Large Business. Free Lancer bisa masuk ke dalam kategori Salaried atau Small Bussines. Maka value Free Lancer akan di drop.

### F. Handle class imbalance
  Ketidakseimbangan kelas dalam pemodelan klasifikasi sering menyebabkan kinerja model yang buruk, khususnya untuk kelas minoritas. Sebagai solusi, teknik oversampling menggunakan SMOTE (Synthetic Minority Over Sampling Technique) dibuat untuk menghasilkan sampel sintetis bagi kelas minoritas. Hasilnya, kedua kelas (1 dan 0) memiliki 3968 sampel, menandakan isu ketidakseimbangan telah diatasi.

## 2. Feature Engineering
### A. Feature selection
  Pada tahap awal, kita pertimbangkan untuk menghapus fitur CustomerID karena mungkin tidak berkontribusi signifikan pada model. Selain itu, ada korelasi tinggi antara NumberOfPersonVisiting dan NumberOfChildrenVisiting. Namun, untuk saat ini, kita akan mempertahankannya dan memutuskannya setelah evaluasi model.

### B. Feature extraction:
  Dalam proses ekstraksi fitur, kami memutuskan untuk membuat 4 fitur baru. Fitur “TotalFamilySize” menunjukkan jumlah keseluruhan anggota keluarga yang berkunjung, memberikan wawasan tentang pilihan akomodasi dan aktivitas. “PitchEfficiency” mengukur efektivitas presentasi berdasarkan durasi dan respons yang diterima. “AgeGroup” kategorikan usia ke dalam kelompok seperti “Young”, “MiddleAge”, dan “Senior”, memberikan gambaran tentang fase kehidupan. Sementara “HasChildren” menentukan apakah pelanggan yang berkunjung memiliki anak, mempengaruhi pilihan aktivitas dan akomodasi.

### C. Feature Tambahan
Berikut adalah empat fitur tambahan yang mungkin akan membantu meningkatkan performansi model:

1. Riwayat Pembelian: Mengenal riwayat pembelian sebelumnya dari pelanggan dapat memberikan wawasan tentang seberapa sering mereka membeli paket wisata dan jenis paket apa yang mereka beli. Hal ini dapat menjadi indikator kuat tentang minat mereka terhadap paket baru.

2. Jumlah Pengeluaran Tahunan untuk Pariwisata: Mengetahui seberapa banyak uang yang dihabiskan oleh pelanggan untuk pariwisata setiap tahunnya bisa memberikan gambaran tentang kemampuan dan keinginan mereka untuk membeli paket wisata.

3. Sumber Referensi: Informasi tentang bagaimana pelanggan mengetahui tentang perusahaan (misalnya, dari iklan, rekomendasi teman, media sosial) dapat memberikan wawasan tentang seberapa efektif saluran pemasaran tertentu dan seberapa besar kemungkinan pelanggan dari saluran tersebut untuk membeli.

4. Feedback dari Paket Sebelumnya: Jika pelanggan pernah membeli paket dari perusahaan sebelumnya, feedback atau ulasan mereka tentang pengalaman tersebut bisa menjadi indikator kuat tentang kemungkinan mereka untuk membeli lagi. Pelanggan yang memberikan ulasan positif mungkin memiliki kemungkinan lebih besar untuk membeli paket baru.

# 3. Modeling
