# Speech Emotion Recognition Web Application

Aplikasi web untuk pengenalan emosi dari suara menggunakan model deep learning hybrid (CNN + LSTM).

## Fitur
- Rekaman audio langsung dari browser
- Upload file audio (.wav)
- Prediksi emosi menggunakan 3 model: Hybrid, CNN, dan LSTM
- Riwayat prediksi
- Antarmuka web yang responsif

## Teknologi
- **Backend**: Flask (Python)
- **Machine Learning**: TensorFlow, Keras
- **Audio Processing**: Librosa, SciPy
- **Frontend**: HTML, CSS, JavaScript, Bootstrap

## Model yang Digunakan
- Hybrid Model (CNN + LSTM)
- CNN Model
- LSTM Model

## Cara Menjalankan Lokal
1. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```

2. Jalankan aplikasi:
   ```bash
   python app.py
   ```

3. Buka browser dan akses: `http://localhost:5000`

## Deployment
Aplikasi ini dapat di-deploy ke platform cloud seperti Render.com, Heroku, atau PythonAnywhere.

## Dataset
Menggunakan dataset IndoWaveSentiment dengan 10 aktor dan berbagai emosi. 