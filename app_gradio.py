import gradio as gr
import soundfile as sf
import numpy as np
import librosa
from scipy.signal import hilbert
from keras.models import load_model
from gammatone.gtgram import gtgram
import joblib
import os
from datetime import datetime

# Load model default dan scaler
scaler = joblib.load('dataset/scaler_hybrid.pkl')
label_encoder = joblib.load('dataset/onehotencoder.pkl')
emotion_labels = label_encoder.categories_[0].tolist()

# File dan path
fs = 22050

# Peta model
model_file_map = {
    'hybrid': 'dataset/model_hybrid.h5',
    'lstm': 'dataset/model_LSTM.h5',
    'cnn': 'dataset/model_cnn.h5'
}
current_model_name = None
current_model = None

def get_model(model_choice):
    global current_model_name, current_model
    if model_choice not in model_file_map:
        model_choice = 'hybrid'
    if current_model_name != model_choice or current_model is None:
        current_model = load_model(model_file_map[model_choice])
        current_model_name = model_choice
    return current_model

# Fitur hybrid
def extract_mfcc(y, sr, n_mfcc=40):
    return np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T, axis=0)

def extract_hilbert(y):
    analytic_signal = hilbert(y)
    envelope = np.abs(analytic_signal)
    spectrum = np.abs(np.fft.fft(envelope))[:len(envelope)//2]
    hilbert_features = np.array([
        np.mean(envelope), np.std(envelope), np.max(envelope), np.min(envelope), np.median(envelope),
        np.percentile(envelope, 25), np.percentile(envelope, 75),
        np.mean(spectrum), np.std(spectrum), np.max(spectrum), np.min(spectrum),
        np.median(spectrum), np.percentile(spectrum, 25), np.percentile(spectrum, 75),
        np.sum(envelope**2)
    ])
    return hilbert_features

def extract_cochleagram(y, sr):
    cochleagram = gtgram(y, sr, 0.025, 0.010, 64, 50)
    cochleagram_mean = np.mean(cochleagram, axis=1)
    return cochleagram_mean[:40]

def extract_hybrid_features(audio_data):
    try:
        # Convert audio data to numpy array
        if isinstance(audio_data, tuple):
            y, sr = audio_data
        else:
            y, sr = librosa.load(audio_data, sr=fs)
        
        if len(y) < 10000:
            return None, "Audio terlalu pendek."
        
        if len(y) > sr * 5:
            start = (len(y) - sr * 5) // 2
            y = y[start:start + sr * 5]
        
        mfcc = extract_mfcc(y, sr)
        hilbert_features = extract_hilbert(y)
        cochleagram_features = extract_cochleagram(y, sr)
        return np.concatenate((mfcc, hilbert_features, cochleagram_features), axis=0), None
    except Exception as e:
        return None, f"Ekstraksi fitur gagal: {e}"

def predict_emotion(audio, model_choice="hybrid"):
    """Fungsi prediksi untuk Gradio"""
    try:
        if audio is None:
            return "Error: Tidak ada audio yang diberikan.", 0.0
        
        # Extract features
        features, error = extract_hybrid_features(audio)
        if features is None:
            return error, 0.0
        
        # Preprocess features
        features = scaler.transform([features])[0]
        features = np.expand_dims(features, axis=0)
        features = np.expand_dims(features, axis=-1)
        
        # Predict
        model = get_model(model_choice)
        prediction = model.predict(features)
        emotion_index = np.argmax(prediction)
        
        # Decode emotion
        one_hot = np.zeros((1, len(emotion_labels)))
        one_hot[0, emotion_index] = 1
        emotion_arr = label_encoder.inverse_transform(one_hot)
        emotion = emotion_arr[0] if isinstance(emotion_arr[0], str) else emotion_arr[0][0]
        
        accuracy = float(prediction[0][emotion_index])
        
        return f"Emosi: {emotion}", accuracy * 100
    except Exception as e:
        return f"Error: {str(e)}", 0.0

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Speech Emotion Recognition", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎤 Speech Emotion Recognition")
        gr.Markdown("Aplikasi pengenalan emosi dari suara menggunakan model deep learning hybrid")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📝 Pilih Model")
                model_choice = gr.Dropdown(
                    choices=["hybrid", "cnn", "lstm"],
                    value="hybrid",
                    label="Model yang digunakan"
                )
                
                gr.Markdown("### 🎙️ Rekam Audio")
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Rekam atau upload audio (.wav)"
                )
                
                predict_btn = gr.Button("🔍 Prediksi Emosi", variant="primary")
            
            with gr.Column():
                gr.Markdown("### 📊 Hasil Prediksi")
                emotion_output = gr.Textbox(label="Hasil Emosi", interactive=False)
                accuracy_output = gr.Slider(
                    minimum=0, maximum=100, value=0, 
                    label="Akurasi (%)", interactive=False
                )
                
                gr.Markdown("### 📈 Informasi")
                info_text = gr.Markdown("""
                **Cara menggunakan:**
                1. Pilih model yang ingin digunakan
                2. Rekam audio atau upload file .wav
                3. Klik tombol "Prediksi Emosi"
                4. Lihat hasil prediksi dan akurasi
                
                **Tips:**
                - Audio minimal 2 detik
                - Format file: .wav
                - Pastikan audio jelas dan tidak berisik
                """)
        
        # Connect components
        predict_btn.click(
            fn=predict_emotion,
            inputs=[audio_input, model_choice],
            outputs=[emotion_output, accuracy_output]
        )
    
    return demo

# Launch app
if __name__ == "__main__":
    demo = create_interface()
    demo.launch() 