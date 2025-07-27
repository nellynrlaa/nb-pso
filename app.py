import streamlit as st
import joblib
import numpy as np

# Load vectorizer dan model
vectorizer = joblib.load('tfidf_vectorizer.pkl')
artifacts = joblib.load('nb_pso_artifacts.pkl')

model = artifacts['model']
selected_features_idx = artifacts['selected_features_idx']

# Mapping label numerik ke teks
label_mapping = {
    0: 'noncyberbullying',
    1: 'cyberbullying'
}

# Fungsi prediksi sederhana
def prediksi_komentar(teks):
    teks = teks.lower().strip()
    tfidf = vectorizer.transform([teks])
    tfidf_selected = tfidf[:, selected_features_idx]
    pred_label = model.predict(tfidf_selected)[0]
    return label_mapping.get(pred_label, "Label tidak diketahui")

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
