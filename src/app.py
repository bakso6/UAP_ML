# import streamlit as st

# pg = st.navigation([
#     st.Page("Inception.py", title="Citra"),
#     ])
# pg.run()

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Judul aplikasi Streamlit
st.title("Prediksi Transportasi Militer")
st.write("Aplikasi ini menggunakan model deep learning untuk memprediksi jenis kendaraan militer berdasarkan gambar.")

# Pilihan model
model_option = st.selectbox("Pilih model yang ingin digunakan:", ["Inceptionv3", "MobileNetV2"])

# Fungsi untuk memuat dan memproses gambar
def preprocess_image(image):
    img = Image.open(image)
    img = img.convert("RGB")  # Mengonversi gambar ke mode RGB
    img = img.resize((150, 150))  # Ukuran sesuai input model
    img_array = np.array(img) / 255.0  # Normalisasi pixel ke rentang [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Menambahkan dimensi batch
    return img_array

# Fungsi untuk melakukan prediksi dengan model
def predict_image(img, model_path):
    # Daftar kelas
    class_names = [
        'Prime movers and trucks', 'Mine-protected vehicles', 'Light utility vehicles', 'Tanks',
        'Self-propelled artillery', 'Infantry fighting vehicles', 'Anti-aircraft',
        'Light armored vehicles', 'Armored combat support vehicles', 'Armored personnel carriers'
    ]

    # Preprocessing gambar
    img_array = preprocess_image(img)

    # Memuat model yang dipilih
    model = tf.keras.models.load_model(model_path)

    # Melakukan prediksi
    prediction = model.predict(img_array)

    # Menghitung probabilitas dan hasil prediksi
    predicted_class = class_names[np.argmax(prediction)]
    probability = np.max(tf.nn.softmax(prediction[0]))  # Probabilitas kelas terpilih

    return predicted_class, probability

# Bagian untuk mengunggah atau menangkap gambar
tab1, tab2 = st.tabs(["Unggah Gambar", "Gunakan Kamera"])

with tab1:
    upload = st.file_uploader("Unggah gambar kendaraan militer (format: jpg, png, jpeg):", type=['jpg', 'png', 'jpeg'])

with tab2:
    camera_image = st.camera_input("Ambil gambar menggunakan kamera:")

if st.button("Prediksi", type="primary"):
    if upload is not None or camera_image is not None:
        # Prioritaskan gambar yang diunggah jika keduanya tersedia
        image = upload if upload is not None else camera_image
        
        st.image(image, caption="Gambar yang diproses", use_column_width=True)
        st.subheader("Hasil Prediksi:")

        # Menampilkan progress bar saat memproses
        with st.spinner('Memproses gambar untuk prediksi...'):
            # Memilih model berdasarkan opsi pengguna
            if model_option == "Inceptionv3":
                model_path = "D:/KULIAHHHHHH/SEMESTER 7/PRAK ML/UAP/src/Inceptionv3.h5"
            else:
                model_path = "D:/KULIAHHHHHH/SEMESTER 7/PRAK ML/UAP/src/mobilenetv2.h5"

            predicted_class, probability = predict_image(image, model_path)

        st.write(f"**Prediksi:** {predicted_class}")
        st.write(f"**Probabilitas:** {probability * 100:.2f}%")

        # Pesan tambahan berdasarkan probabilitas
        if probability > 0.8:
            st.success("Prediksi sangat yakin!")
        elif probability > 0.5:
            st.warning("Prediksi cukup yakin.")
        else:
            st.error("Prediksi kurang yakin, coba unggah gambar lain.")
    else:
        st.write("Silakan unggah atau ambil gambar terlebih dahulu!")
