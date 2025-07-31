# Speech Emotion Recognition

Aplikasi pengenalan emosi dari suara menggunakan model deep learning hybrid (CNN + LSTM).

## Demo
Coba aplikasi langsung di: [Hugging Face Spaces](https://huggingface.co/spaces)

## Fitur
- 🎙️ Rekam audio langsung dari browser
- 📁 Upload file audio (.wav)
- 🤖 3 model: Hybrid, CNN, dan LSTM
- 📊 Hasil prediksi dengan akurasi
- 🎨 Interface yang modern dan responsif

## Model yang Digunakan
- **Hybrid Model** (CNN + LSTM) - Default
- **CNN Model** 
- **LSTM Model**

## Dataset
Menggunakan dataset IndoWaveSentiment dengan 10 aktor dan berbagai emosi.

## Teknologi
- **Backend**: Gradio (Python)
- **Machine Learning**: TensorFlow, Keras
- **Audio Processing**: Librosa, SciPy
- **Frontend**: Gradio UI

## Cara Menjalankan Lokal
1. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```

2. Jalankan aplikasi:
   ```bash
   python app.py
   ```

3. Buka browser dan akses: `http://localhost:7860`

## Deployment
Aplikasi ini di-deploy menggunakan Hugging Face Spaces untuk akses publik. 