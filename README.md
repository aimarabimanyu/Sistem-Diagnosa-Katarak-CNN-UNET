# Laporan Skripsi - Aimar Abimayu Pratama

## Domain Proyek

Katarak adalah penyakit mata yang umum ditemui di seluruh dunia, termasuk juga di Indonesia. Penyakit ini terjadi karena lensa pada mata pengidapnya menjadi keruh dan berawan disebabkan oleh gumpalan protein (Asbell, et al., 2005). Hal ini menyebabkan penglihatan kabur dan pada beberapa kasus menyebabkan kebutaan total. Berdasarkan hasil estimasi World Health Organization (WHO) bahwa sebanyak 1 miliar orang yang mengalami gangguan pada penglihatan, 94 jutanya dikarenakan oleh katarak. Diperkirakan 10 juta orang dari seluruh dunia yang mengalami kebutaan dan 35 juta orang yang mengalami gangguan penglihatan tingkat menengah hingga parah disebabkan oleh katarak (Ang dan Afshari., 2021). Meskipun katarak dapat diobati, aksesibilitas terhadap diagnosis katarak masih menjadi masalah di Indonesia, terutama pada wilayah pedesaan yang tergolong tertinggal, terdepan, dan terluar (3T). Slit lamp yang merupakan alat untuk diagnosis katarak memerlukan teknik khusus dalam penggunaan dan harganya yang tidak terjangkau membuatnya tidak tersedia secara luas di fasilitas Kesehatan (Hutabri, et al., 2018). Oleh sebab itu, diperlukan alat yang praktis dan terjangkau dalam diagnosis katarak.

## Business Understanding

Katarak dapat dibagi menjadi beberapa jenis. Jika berdasarkan lokasi kekeruhan lensa, katarak dapat dibagi menjadi:
- Katarak Subkapsular Posterior (PSC) : kekeruhan terjadi pada bagian belakang lensa.
- Katarak Nuklear : kekeruhan terjadi pada bagian tengah lensa.
- Katarak Kortikal : kekeruhan terjadi pada bagian pinggir lensa.

Sedangkan jika berdasarkan penyebabnya, katarak dapat dibagi menjadi:
- Katarak Kongenital : katarak yang terjadi sejak lahir.
- Katarak Traumatik : katarak yang terjadi akibat trauma pada mata.
- Katarak Senilis : katarak yang terjadi akibat proses penuaan.
- Katarak Komplikasi : katarak yang terjadi akibat penyakit lain seperti diabetes, glaukoma, atau rheumoid arthritis.

Kemudian jika berdasarkan pada tingkat keparahannya, katarak dapat dibagi menjadi:
- Katarak Insipien : katarak yang masih ringan dan belum mengganggu penglihatan.
- Katarak Imatur : katarak yang sudah mengganggu penglihatan.
- Katarak Matur : katarak yang sudah sangat mengganggu penglihatan.
- Katarak Hipermatur : katarak yang sudah sangat parah dan dapat menyebabkan kebutaan.

Untuk diagnosa katarak pada mata, dapat dilakukan dengan menggunakan alat slit lamp. Alat ini dapat mendeteksi katarak dengan menggunakan cahaya yang dipantulkan oleh lensa mata. Namun, alat ini tidak tersedia secara luas di fasilitas kesehatan karena harganya yang mahal dan memerlukan teknik khusus dalam penggunaannya. Oleh sebab itu, diperlukan alat yang praktis dan terjangkau dalam diagnosis katarak. Salah satu alternatif yang ditawarkan adalah menggunakan Raspberry Pi 3B dan sensor kamera. Sensor kamera akan menangkap gambar mata dan kemudian gambar tersebut akan diolah menggunakan Raspberry Pi 3B. 

### Problem Statements
- Berdasarkan jenis2 katarak yang ada, katarak mana yang akan dideteksi?
- Bagaimana memisahkan gambar mata agar fokus pada area pupil?
- Bagaimana melakukan ekstraksi fitur dari gambar mata?
- Bagaimana memprediksi jenis katarak berdasarkan fitur yang sudah diekstraksi?

### Goals
- Mendeteksi jenis katarak berdasarkan tingkat keparahan katarak.
- Membuat model tersendiri yang mampu memisahkan area pupil dari gambar mata.
- Membuat model menggunakan yang mampu melakukan ekstraksi fitur dari gambar mata.
- Membuat kesatuan model dari yang sudah dibuat sebelumnya dan untuk memprediksi jenis katarak berdasarkan fitur yang sudah diekstraksi.

### Solution statements
- Menggunakan tingkat keparahan yaitu mata normal, katarak imatur, dan katarak matur.
- Melakukan pemodelan dengan menggunakan U-Net untuk memisahkan area pupil dari gambar mata.
- Melakukan pemodelan dengan menggunakan Convolutional Neural Network (CNN) untuk melakukan ekstraksi fitur dari gambar mata.
- Menggunakan Artificial Neural Network (ANN) untuk memprediksi jenis katarak berdasarkan fitur yang sudah diekstraksi.

## Data Understanding

Data merupakan kumpulan harga saham United Tractors pada saat perdagangan saham. Data didapat dari [Kaggle](https://www.kaggle.com/datasets/rischan/indonesia-popular-stocks) dengan rentang waktu 20 tahun dari 5 September 2000 sampai 3 Juli 2020. Frekuensi pembaruan data adalah harian, dimana baris data mewakili harga saat saham tersebut diperdagangankan. Data yang didapat berjumlah 4980 baris dan 7 kolom. Seluruh data bersifat numerik, bertipe data float, dan bersifat time-series.

### Variabel-variabel pada UNTR dataset adalah sebagai berikut:
- Date : merupakan tanggal pada data diambil.
- Open : merupakan harga saham pada pembukaan perdagangan.
- High : merupakan harga saham tertinggi pada hari tersebut.
- Low : merupakan harga saham terendah pada hari tersebut.
- Close : merupakan harga saham pada penutupan perdagangan.
- Adj Close : merupakan harga saham penutupan yang telah disesuaikan dengan kondisi pasar atau aksi korporasi.
- Volume : merupakan jumlah saham yang diperdagangkan pada hari tersebut.

### Eksplorasi Data
- Melihat deskripsi statistik dari data.
- Melihat jumlah missing value pada data menggunakan fungsi isnull().sum(). Didapatkan pada kolom Open, High, Low, Close, Adj Close, dan Volume masing2 ada sejumlah 28 data yang bernilai null. Data yang bernilai null kemudian dihapus menggunakan fungsi dropna().
- Melihat nilai yang bernilai nol (0) pada data. Didapatkan kalau pada kolom Volume ada sejumlah 165 data yang bernilai 0. Data yang bernilai 0 kemudian dihapus menggunakan code.
```python
dataset = dataset[dataset.Volume != 0]
```
- Melakukan cek outlier pada data dengan menggunakan boxplot. Dari boxplot yang didapat, tidak ditemukan adanya outlier pada data kecuali Volume. Volume tidak dilakukan pembuangan outlier karena meninjau dari tahap selanjutnya.
- Melihat visualisasi korelasi antar variabel dan memutuskan untuk menggunakan variabel Open, High, dan Low sebagi fitur dan membuang variabel Volume karena tidak memiliki korelasi dengan variabel Close.

<p>
  <img src="assets/corr.png", width="577", height="516">
</p>

- Membuang variabel Adj Close karena variabel tersebut adalah hasil dari penyesuaian harga Close dengan kondisi pasar atau aksi korporasi. Jadi tidak relevan untuk dijadikan fitur.

## Data Preparation

- Data dibagi secara acak menjadi data train dan data test dengan rasio 80:20 menggunakan fungsi train_test_split() pada library Scikit-Learn. Pembagian data dengan rasio ini ini dilakukan untuk menghindari overfitting dan underfitting, selain itu berguna juga untuk mengukur tingkat keberhasilan model dalam prediksi. 
- Melakukan scaling pada data train menggunakan MinMaxScaler. Scaling dilakukan untuk menghindari perbedaan skala antar fitur yang dapat mempengaruhi performa model. Digunakan MinMaxScaler karena fitur pada data tidak memiliki bentuk distribusi normal. Untuk menggunakan MinMaxScaler memanfaatkan fungsi MinMaxScaler() pada library Scikit-Learn.

<p>
  <img src="assets/hist.png", width="577", height="516">
</p>

## Modeling

Model yang digunakan adalah Linear Regression, K-Nearest Neighbours, Random Forest, dan Adaptive Boosting. Model-model tersebut digunakan karena dapat digunakan untuk melakukan prediksi pada data numerik. Selain itu, model-model tersebut juga dapat digunakan untuk melakukan prediksi pada data yang memiliki banyak fitur.

#### Linear Regression
Pada model Linear Regression, nilai parameter yang digunakan adalah nilai default dari fungsi LinearRegression(). Kelebihan dari model ini adalah sederhana dan mudah diinterpretasikan. Kekurangan dari model ini adalah sensitif terhadap outliers dan tidak dapat menangani non-linear data.

#### K-Nearest Neighbours
Model ini menggunakan parameter sebagi berikut: 

- n_neighbours=5, digunakan n_neighbours sebesar 5 untuk menentukan jumlah tertangga terdekat.

Kelebihan dari model ini adalah mudah diimplementasikan, dapat digunakan untuk data yang tidak linear, dan cocok untuk data dengan distribusi tidak normal. Kekurangan dari model ini adalah sensitif terhadap outliers, tidak dapat menangani data yang memiliki banyak fitur, dan sensitif terhadap skala data.

#### Random Forest
Model ini menggunakan parameter sebagai berikut: 

- n_estimators=100, parameter n_estimators bernilai 100 karena decision tree yang digunakan berjumlah 100.
- max_depth=5, parameter max_depth bernilai 5 untuk menentukan kalau kedalaman maksimum dari pohon pada ensemble.
- random_state=42, parameter random_state bernilai 42 untuk melakukan randomisasi pada model. 

Kelebihan dari model ini adalah dapat digunakan untuk data yang tidak linear, dapat digunakan untuk data yang memiliki banyak fitur, dan dapat digunakan untuk data yang memiliki banyak outliers. Kekurangan dari model ini adalah kompleks dan sulit diinterpretasikan.

#### Adaptive Boosting
Pada model ini menggunakan parameter sebagai berikut:

- n_estimators=100, parameter n_estimators bernilai 100 karena decision tree untuk ensemble learning yang digunakan berjumlah 100. 
- random_state=42, parameter random_state bernilai 42 untuk melakukan randomisasi pada tiap iterasi learning. 

Kelebihan dari model ini adalah dapat digunakan untuk data yang tidak linear, mengurangi overfitting karena mempertimbangkan model2 lemah dan menggabungkannya, dan mampu meningkatkan performa pada kasus regresi dengan mengambil kelebihan dari ensemble learning. Kekurangan dari model ini adalah rentan terhadap noise dan sensitif terhadap overfitting jika n_estimator terlalu besar atau learning_rate terlalu tinggi

Berdasarkan dari variasi model yang sudah dicoba sebelumnya, didapat kalau model yang paling baik adalah linear regression dan KNN. Dua model tersebut mendapatkan MSE training masing2 sebesar 39.75 dan 29.47, untuk MSE test masing2 akan dijelaskan pada tahap evaluasi.

## Evaluation

Metrik evaluasi menggunakan Mean Squared Error (MSE). MSE digunakan karena MSE dapat mengukur seberapa jauh rata-rata kuadrat dari error. Selain itu, MSE juga dapat mengukur seberapa dekat data dengan garis regresi. Semakin kecil nilai MSE, maka semakin baik model yang digunakan.

Formula MSE adalah sebagai berikut:

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

$$y_i = nilai\ aktual$$

$$\hat{y_i} = nilai\ prediksi$$

Untuk perbandingan, digunakan nilai MSE pada training dan testing. Nilai MSE pada training digunakan untuk melihat seberapa baik model dalam mempelajari data. Sedangkan nilai MSE pada testing digunakan untuk melihat seberapa baik model dalam memprediksi data. Dari tiap model didapatkan nilai MSE pada training dan testing masing2 adalah 39.75 dan 53.81 pada Linear Regression, 29.47 dan 67.37 pada KNN, 73.44 dan 108.44 pada Random Forest, dan 395.60 dan 468.40 pada Adaptive Boosting.

<p>
  <img src="assets/eval.png", width="635", height="506,25">
</p>

Kemudian melihat hasil prediksi dengan mengambil 5 data random pada data test. Dari hasil prediksi didapatkan kalau model Linear Regression, KNN, Random Forest, dan Adaptive Boosting nyaris setara performanya dalam memprediksi nilai ribuan. Tetapi ketika memprediksi nilai ratusan, model Linear Regression dan KNN lebih baik dibandingkan model Random Forest dan Adaptive Boosting. Bahkan model linear regression memprediksi dengan nilai yang hampir sama dengan nilai sebenarnya.

| Index | y_true        | Prediksi Linear Regression | Prediksi KNN | Prediksi Random Forest | Prediksi Adaptive Boosting |
|:------|:--------------|:--------------------------:|:------------:|:----------------------:|:--------------------------:|
| 3019  | 21800.000000  |          21519.9           |   21600.0    |        21758.6         |          22098.7           |
| 4615  | 26425.000000  |          26292.8           |   26330.0    |        26667.1         |          26640.0           |
| 2427  | 18699.800781  |          18245.0           |   18135.3    |        18316.6         |          17887.2           |
| 822   | 758.921021    |           752.7            |    777.3     |         582.6          |           1602.0           |
| 4573  | 29000.000000  |          29196.5           |   29280.0    |        29204.1         |          28983.5           |

Berdasarkan hasil yang didapat, dapat disimpulkan model Linear Regression dan KNN nyaris setara performanya. Hanya saja, model Linear Regression lebih baik dalam memprediksi data sedangkan model KNN lebih baik dalam mempelajari data. Sedangkan model Random Forest dan Adaptive Boosting memiliki performa yang lebih buruk dibandingkan model Linear Regression dan KNN. Dengan ini untuk memprediksi harga penutupan saham United Tractors, dapat digunakan model Linear Regression dengan fine-tuning.

Referensi:
 - [Hermanto, T., Nugroho, I., Sunandar, M., & Totohendarto, M. (2022). IMPLEMENTASI ALGORITMA LINEAR REGRESSION UNTUK PREDIKSI HARGA SAHAM PT. ANEKA TAMBANG TBK. Jurnal Transformatika, 19(2). doi:http://dx.doi.org/10.26623/transformatika.v19i2.4396](https://journals.usm.ac.id/index.php/transformatika/article/view/4396)
 - [Putra, J. S., Ramadhani, R. D., &amp; Burhanuddin, A. (2022). Prediksi Harga Saham Bank Bri Menggunakan algoritma linear regresion Sebagai Strategi jual Beli Saham. Journal of Dinda : Data Science, Information Technology, and Data Analytics, 2(1), 1–10. https://doi.org/10.20895/dinda.v2i1.273 ](https://doi.org/10.20895/dinda.v2i1.273)
 - [Caniago, A. I., Kaswidjanti, W., &amp; Juwairiah, J. (2021). Recurrent neural network with gate recurrent unit for stock price prediction. Telematika, 18(3), 345. https://doi.org/10.31315/telematika.v18i3.6650 ](https://doi.org/10.31315/telematika.v18i3.6650)