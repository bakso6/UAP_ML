# Klasifikasi Transportasi Militer 

## Deskripsi Proyek

Klasifikasi transportasi militer menggunakan MobileNetV2 dan InceptionV3 adalah penerapan teknik deep learning untuk mengidentifikasi dan mengklasifikasikan jenis kendaraan militer berdasarkan citra yang diambil oleh kamera atau sensor visual. Dalam sistem ini, dua arsitektur jaringan syaraf konvolusional (CNN) yang populer, yaitu MobileNetV2 dan InceptionV3, digunakan untuk menganalisis dan memproses citra kendaraan militer.

**Link dataset yang digunakan** [dataset](https://www.kaggle.com/datasets/amanrajbose/millitary-vechiles/data)

## Overview dataset

Dataset yang bersumber dari kaggle dengan link sebagai [berikut](https://www.kaggle.com/datasets/amanrajbose/millitary-vechiles/data). Dataset terdiri atas 18.596 gambar transportasi militer berbagai kategori


## Deskrisi Model

**MobileNetV2** adalah model ringan yang dirancang untuk aplikasi mobile dan perangkat dengan sumber daya terbatas, namun tetap memberikan performa tinggi dalam hal akurasi dan efisiensi komputasi. MobileNetV2 menggunakan konsep depthwise separable convolutions, yang memungkinkan model ini untuk memiliki jumlah parameter yang lebih sedikit dan lebih cepat dalam pemrosesan citra.

**InceptionV3**, di sisi lain, adalah model yang lebih besar dan lebih kompleks, yang dikenal karena kemampuannya dalam menangani berbagai skala fitur melalui arsitektur inception modules. Dengan berbagai ukuran filter dalam satu layer, InceptionV3 mampu mendeteksi fitur-fitur penting dalam citra dengan lebih mendalam dan akurat, meskipun memerlukan lebih banyak sumber daya komputasi dibandingkan MobileNetV2.

## Hasil

## Model InceptionV3
![Gambar 1](https://github.com/bakso6/UAP_ML/blob/main/gambar/InceptionV3.jpeg)

![Gambar 2](https://github.com/bakso6/UAP_ML/blob/main/gambar/InceptionV3%20(2).jpeg)

![Gambar 3](https://github.com/bakso6/UAP_ML/blob/main/gambar/InceptionV3%20(3).jpeg)

Gambar tersebut menampilkan **classification report** untuk data validasi. Berikut adalah penjelasan hasilnya:

1. **Precision**: Mengukur seberapa tepat model dalam memprediksi suatu kelas (berapa dari prediksi benar yang sebenarnya benar).
   - Precision tertinggi ada di kelas "Anti-aircraft" (0.66).
   - Precision terendah ada di kelas "Infantry fighting vehicles" (0.43).

2. **Recall**: Mengukur seberapa baik model menemukan semua data yang relevan untuk suatu kelas (berapa banyak yang benar dari semua data aktual untuk kelas tersebut).
   - Recall tertinggi ada di kelas "Infantry fighting vehicles" (0.67).
   - Recall terendah ada di kelas "Mine-protected vehicles" (0.36).

3. **F1-Score**: Rata-rata harmonis dari precision dan recall, digunakan untuk menilai keseimbangan antara keduanya.
   - F1-score tertinggi ada di kelas "Light armored vehicles" (0.59).
   - F1-score terendah ada di kelas "Mine-protected vehicles" (0.41).

4. **Support**: Jumlah data aktual untuk setiap kelas di data validasi.
   - Support terbesar ada di kelas "Light armored vehicles" dan "Armored personnel carriers" (509).
   - Support terkecil ada di kelas "Prime movers and trucks" (404).

5. **Overall Metrics**:
   - **Accuracy**: 0.52 (52%), yang menunjukkan proporsi prediksi yang benar terhadap seluruh data validasi.
   - **Macro Average**: Rata-rata sederhana dari precision, recall, dan F1-score untuk semua kelas (tanpa mempertimbangkan support).
   - **Weighted Average**: Rata-rata berbobot dari precision, recall, dan F1-score, dengan bobot sesuai jumlah support masing-masing kelas.

### Kesimpulan:
Model memiliki performa yang sedang, dengan akurasi keseluruhan 52%. Performa cukup bervariasi antar kelas, yang mungkin disebabkan oleh ketidakseimbangan data, kesulitan dalam membedakan fitur antar kelas, atau pengaturan model yang perlu diperbaiki. Anda mungkin perlu mempertimbangkan **data augmentation**, **hyperparameter tuning**, atau **perbaikan arsitektur model** untuk meningkatkan kinerja.


## Model MobileNetV2

![Gambar 1](https://github.com/bakso6/UAP_ML/blob/main/gambar/MobileNetv2%20(3).jpeg)

![Gambar 2](https://github.com/bakso6/UAP_ML/blob/main/gambar/MobileNetv2%20(2).jpeg)

![Gambar 3](https://github.com/bakso6/UAP_ML/blob/main/gambar/MobileNetv2.jpeg)

Gambar tersebut merupakan classification report dari data validasi dengan metrik evaluasi seperti precision, recall, F1-score, dan support untuk masing-masing kelas. Dibandingkan dengan laporan sebelumnya, hasil ini menunjukkan perbaikan kinerja model secara keseluruhan. Berikut adalah analisisnya:

Precision:

Mengukur ketepatan prediksi (berapa banyak prediksi benar dari total prediksi untuk kelas tersebut).
Precision tertinggi: Anti-aircraft (0.73).
Precision terendah: Self-propelled artillery (0.55).

Recall:

Mengukur sensitivitas model (berapa banyak data yang benar dari total data aktual untuk kelas tersebut).
Recall tertinggi: Light armored vehicles (0.83).
Recall terendah: Mine-protected vehicles (0.42).

F1-Score:

Rata-rata harmonis antara precision dan recall. Cocok digunakan untuk menilai keseimbangan performa model.
F1-score tertinggi: Light armored vehicles (0.72).
F1-score terendah: Mine-protected vehicles (0.50).

Support:

Jumlah sampel aktual untuk setiap kelas di data validasi.
Kelas dengan jumlah sampel terbesar: Light armored vehicles dan Armored personnel carriers (509).
Kelas dengan jumlah sampel terkecil: Prime movers and trucks (404).

Overall Metrics
Accuracy:

Akurasi model meningkat menjadi 0.64 (64%), menunjukkan proporsi prediksi yang benar dari seluruh prediksi.

Macro Avg:

Rata-rata sederhana precision, recall, dan F1-score dari semua kelas, tanpa mempertimbangkan support.
Precision, Recall, F1-Score: 0.64, 0.63, 0.63.

Weighted Avg:

Rata-rata berbobot precision, recall, dan F1-score, dengan bobot berdasarkan jumlah support untuk setiap kelas.
Precision, Recall, F1-Score: 0.64, 0.64, 0.64.

## **Perbandingan**

Peningkatan Akurasi:

Akurasi meningkat dari 0.52 menjadi 0.64, menunjukkan model ini lebih baik dibandingkan model sebelumnya.

Kinerja Kelas:

Precision, recall, dan F1-score meningkat hampir di semua kelas, terutama untuk Light armored vehicles (dari 0.59 ke 0.72 pada F1-score).

Kesalahan pada Kelas Tertentu:

Mine-protected vehicles masih memiliki performa rendah, khususnya pada recall (0.42), menunjukkan model kesulitan mengidentifikasi sampel untuk kelas ini.

## **Kesimpulan**

Model MobileNetV2 menunjukkan performa yang lebih baik dan lebih efisien dibandingkan InceptionV3 untuk tugas klasifikasi ini. Namun, masih diperlukan langkah tambahan untuk meningkatkan recall pada kelas tertentu agar hasil lebih seimbang.

## Link Model

[InceptionV3](https://colab.research.google.com/drive/1GTEjh8NHmmSU0JQalD1Vzg162CaMYY1m?usp=sharing)

[MobileNetV2](https://colab.research.google.com/drive/162bbo4jBBei6xnVJytxEuuel9oLXrvSN?usp=sharing)

## Link Web

[Streamlit](https://uapml-gusnaba.streamlit.app/)

## Tampilan Web

![Gambar 1](https://github.com/bakso6/UAP_ML/blob/main/gambar/uaapp.jpeg)

![Gambar 2](https://github.com/bakso6/UAP_ML/blob/main/gambar/uap.jpeg)

## Author

[Gusnaba Fata Kusuma](https://github.com/bakso6)




