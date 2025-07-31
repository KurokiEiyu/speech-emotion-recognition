from flask import Flask, render_template, request, redirect, url_for, session
import soundfile as sf
import numpy as np
import librosa
from scipy.signal import hilbert
from keras.models import load_model
from gammatone.gtgram import gtgram
import joblib
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Load model default dan scaler
scaler = joblib.load('dataset/scaler_hybrid.pkl')
label_encoder = joblib.load('dataset/onehotencoder.pkl')  # ganti dengan OneHotEncoder
emotion_labels = label_encoder.categories_[0].tolist()

# File dan path
temp_audio_path = 'static/audio.wav'
temp_upload_path = 'static/uploaded_audio.wav'
last_audio_path = 'static/audio_last.wav'
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

def extract_hybrid_features(filename):
    try:
        y, sr = librosa.load(filename, sr=fs)
        if len(y) < 10000:
            print("Audio terlalu pendek.")
            return None
        if len(y) > sr * 5:
            start = (len(y) - sr * 5) // 2
            y = y[start:start + sr * 5]
        mfcc = extract_mfcc(y, sr)
        hilbert_features = extract_hilbert(y)
        cochleagram_features = extract_cochleagram(y, sr)
        return np.concatenate((mfcc, hilbert_features, cochleagram_features), axis=0)
    except Exception as e:
        print(f"Ekstraksi fitur gagal: {e}")
        return None

def add_history(emotion, accuracy, tipe, model_choice, filename=''):
    history = session.get('history', [])
    history.append({
        'emotion': emotion,
        'accuracy': round(accuracy * 100, 2),
        'tipe': tipe,
        'model': model_choice,
        'filename': filename,
        'waktu': datetime.now().strftime('%d-%m-%Y %H:%M:%S')
    })
    session['history'] = history

@app.route('/')
def index():
    return render_template('indexExperiment.html', history=session.get('history', []))

@app.route('/record', methods=['POST'])
def record():
    try:
        model_choice = request.form.get('model_choice', 'hybrid')
        if 'audio' not in request.files:
            return redirect(url_for('result', emotion='Error', accuracy=0, error='Audio tidak ditemukan.'))

        audio_file = request.files['audio']
        audio_file.save(temp_audio_path)
        with open(temp_audio_path, 'rb') as src, open(last_audio_path, 'wb') as dst:
            dst.write(src.read())

        y, sr = librosa.load(temp_audio_path, sr=fs)
        if librosa.get_duration(y=y, sr=sr) < 2.0:
            os.remove(temp_audio_path)
            return redirect(url_for('result', emotion='Error', accuracy=0, error='Audio terlalu singkat. Minimal 2 detik.'))

        features = extract_hybrid_features(temp_audio_path)
        if features is None:
            return redirect(url_for('result', emotion='Error', accuracy=0, error='Ekstraksi fitur gagal.'))

        features = scaler.transform([features])[0]
        features = np.expand_dims(features, axis=0)
        features = np.expand_dims(features, axis=-1)

        model = get_model(model_choice)
        prediction = model.predict(features)
        emotion_index = np.argmax(prediction)

        one_hot = np.zeros((1, len(emotion_labels)))
        one_hot[0, emotion_index] = 1
        emotion_arr = label_encoder.inverse_transform(one_hot)
        emotion = emotion_arr[0] if isinstance(emotion_arr[0], str) else emotion_arr[0][0]

        accuracy = float(prediction[0][emotion_index])
        add_history(emotion, accuracy, 'Rekaman', model_choice, 'Rekaman Audio')
        return redirect(url_for('result', emotion=emotion, accuracy=accuracy))
    except Exception as e:
        print(f"Error during record: {e}")
        return redirect(url_for('result', emotion='Error', accuracy=0, error='Terjadi error saat rekam.'))

@app.route('/upload', methods=['POST'])
def upload():
    try:
        model_choice = request.form.get('model_choice', 'hybrid')
        if 'file' not in request.files:
            return redirect(url_for('result', emotion='Error', accuracy=0, error='File audio tidak ditemukan.'))

        file = request.files['file']
        if file.filename == '' or not file.filename.lower().endswith('.wav'):
            return redirect(url_for('result', emotion='Error', accuracy=0, error='Format file harus .wav.'))

        file.save(temp_upload_path)
        y, sr = librosa.load(temp_upload_path, sr=fs)
        if librosa.get_duration(y=y, sr=sr) < 2.0:
            os.remove(temp_upload_path)
            return redirect(url_for('result', emotion='Error', accuracy=0, error='Audio terlalu singkat. Minimal 2 detik.'))

        with open(temp_upload_path, 'rb') as src, open(last_audio_path, 'wb') as dst:
            dst.write(src.read())

        features = extract_hybrid_features(temp_upload_path)
        if features is None:
            return redirect(url_for('result', emotion='Error', accuracy=0, error='Ekstraksi fitur gagal.'))

        features = scaler.transform([features])[0]
        features = np.expand_dims(features, axis=0)
        features = np.expand_dims(features, axis=-1)

        model = get_model(model_choice)
        prediction = model.predict(features)
        emotion_index = np.argmax(prediction)

        one_hot = np.zeros((1, len(emotion_labels)))
        one_hot[0, emotion_index] = 1
        emotion_arr = label_encoder.inverse_transform(one_hot)
        emotion = emotion_arr[0] if isinstance(emotion_arr[0], str) else emotion_arr[0][0]

        accuracy = float(prediction[0][emotion_index])
        os.remove(temp_upload_path)
        add_history(emotion, accuracy, 'Upload', model_choice, file.filename)
        return redirect(url_for('result', emotion=emotion, accuracy=accuracy))
    except Exception as e:
        print(f"Error during upload: {e}")
        return redirect(url_for('result', emotion='Error', accuracy=0, error='Terjadi error saat upload.'))

@app.route('/result/<emotion>/<float:accuracy>')
def result(emotion, accuracy):
    return render_template('indexExperiment.html',
        emotion=emotion,
        accuracy=round(accuracy * 100, 2),
        error=request.args.get('error'),
        audio_last_path=last_audio_path if os.path.exists(last_audio_path) else None,
        history=session.get('history', [])
    )

@app.route('/reset_history', methods=['POST'])
def reset_history():
    session.pop('history', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
