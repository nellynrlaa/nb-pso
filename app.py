import streamlit as st
import joblib
import numpy as np

# Load vectorizer dan model
vectorizer = joblib.load('tfidf_vectorizer.pkl')
artifacts = joblib.load('nb_pso_artifacts.pkl')

model = artifacts['model']
selected_features_idx = artifacts['selected_features_idx']

# Fungsi prediksi sederhana (preprocessing ringan)
def prediksi_komentar(teks):
    teks = teks.lower().strip()  # casefold dan hapus spasi
    tfidf = vectorizer.transform([teks])
    tfidf_selected = tfidf[:, selected_features_idx]
    hasil = model.predict(tfidf_selected)[0]
    return hasil

# UI Streamlit
st.title("Deteksi Komentar Cyberbullying")
st.markdown("Model: **Naive Bayes + PSO**")

input_teks = st.text_area("Masukkan komentar:", height=150)

if st.button("Prediksi"):
    if input_teks.strip() != "":
        hasil = prediksi_komentar(input_teks)
        st.success(f"**Hasil prediksi:** {hasil}")
    else:
        st.warning("Komentar tidak boleh kosong.")
