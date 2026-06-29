# ⚙️ AI Risk Simulator - Final UAS

## Machine Learning + XAI + MCDM SAW + MLOps

AI Risk Simulator adalah aplikasi berbasis **Machine Learning** untuk memprediksi tingkat risiko kegagalan mesin berdasarkan parameter:

- 🌡️ Suhu Mesin
- 🔩 Getaran Mesin

Sistem ini mengintegrasikan proses **Machine Learning Prediction, Explainable AI (XAI), Sistem Pendukung Keputusan MCDM SAW, dan MLOps** dalam satu dashboard Streamlit.


## 🚀 Fitur Utama

✅ **Machine Learning Prediction**  
Melakukan prediksi skor risiko kegagalan mesin menggunakan model yang telah dilatih.

✅ **Model Persistence (MLOps)**  
Model dan scaler disimpan dalam file `.joblib` sehingga aplikasi dapat melakukan prediksi tanpa training ulang.

✅ **What-If Simulation**  
Pengguna dapat mengubah nilai suhu dan getaran untuk melihat perubahan risiko secara langsung.

✅ **Data Drift Monitoring**  
Mendeteksi perubahan data input yang berbeda dari data training.

✅ **Input Validation**  
Menangani input ekstrem agar sistem tetap stabil.

✅ **Privacy & Anonymization**  
Menghapus data sensitif sebelum diproses.

✅ **MCDM SAW Recommendation**  
Menghasilkan ranking alternatif berdasarkan nilai risiko dan efisiensi.

✅ **Explainable AI (XAI)**  
Menampilkan kontribusi fitur yang memengaruhi hasil prediksi.

✅ **Uncertainty Estimation**  
Memberikan rentang kepercayaan berdasarkan nilai RMSE model.
