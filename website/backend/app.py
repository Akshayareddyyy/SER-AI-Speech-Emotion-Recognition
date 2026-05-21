from pathlib import Path
import os
import tempfile
import joblib
import numpy as np
import librosa  # type: ignore
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
from pydub import AudioSegment

# configuration
SR = 22050
N_MFCC = 40
MAX_LEN = 216

MODEL_VERSION = "Improved Conv1D MFCC Sequence"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / 'models'
MODEL_PATH = PROJECT_ROOT / 'models' / 'conv1d_improved_model.keras'
ENCODER_PATH = PROJECT_ROOT / 'models' / 'conv1d_improved_label_encoder.pkl'
SCALER_PATH = PROJECT_ROOT / 'models' / 'conv1d_improved_scaler.pkl'

app = Flask(__name__)
CORS(app)


@app.route('/model-info', methods=['GET'])
def model_info():
    return jsonify({
        "model_version": MODEL_VERSION,
        "model_path": str(MODEL_PATH),
        "encoder_path": str(ENCODER_PATH),
        "scaler_path": str(SCALER_PATH)
    })

# Load model and artifacts once
try:
    from tensorflow import keras  # type: ignore
    model = keras.models.load_model(str(MODEL_PATH))
    print("Loaded model version:", MODEL_VERSION)
    print("Using model file:", MODEL_PATH)
except Exception as e:
    model = None
    print(f"Warning: could not load model: {e}")

try:
    le = joblib.load(str(ENCODER_PATH))
    print("Using encoder file:", ENCODER_PATH)
except Exception as e:
    le = None
    print(f"Warning: could not load label encoder: {e}")

try:
    scaler = joblib.load(str(SCALER_PATH))
    print("Using scaler file:", SCALER_PATH)
except Exception as e:
    scaler = None
    print(f"Warning: could not load scaler: {e}")


def extract_mfcc_sequence(path, n_mfcc=N_MFCC, sr=SR, max_len=MAX_LEN):
    y, _ = librosa.load(path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_t = mfcc.T
    t = mfcc_t.shape[0]
    if t >= max_len:
        mfcc_t = mfcc_t[:max_len, :]
    else:
        pad_width = max_len - t
        pad = np.zeros((pad_width, n_mfcc), dtype=mfcc_t.dtype)
        mfcc_t = np.vstack([mfcc_t, pad])
    return mfcc_t


@app.route('/')
def index():
    return 'Speech Emotion Recognition API is running'




@app.route('/predict', methods=['POST'])
def predict():
    if model is None or le is None or scaler is None:
        return jsonify({'error': 'Model or artifacts not loaded on server'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # save uploaded file to a temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
    out_wav = None
    try:
        file.save(tmp.name)
        tmp_path = tmp.name

        # Convert any uploaded audio to WAV (mono, SR, 16-bit PCM) using pydub
        try:
            # load with pydub (ffmpeg must be available on PATH)
            audio = AudioSegment.from_file(tmp_path)
            # set to desired params
            audio = audio.set_frame_rate(SR).set_channels(1).set_sample_width(2)
            out_wav_f = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            out_wav = out_wav_f.name
            audio.export(out_wav, format='wav')
            out_wav_f.close()
            use_path = out_wav
        except Exception as conv_e:
            # If conversion fails, try to proceed with original file and let librosa attempt to load
            app.logger.exception('Audio conversion failed')
            use_path = tmp_path

        # extract mfcc sequence from WAV/converted file
        seq = extract_mfcc_sequence(use_path, n_mfcc=N_MFCC, sr=SR, max_len=MAX_LEN)
        X = np.expand_dims(seq, axis=0)  # (1, max_len, n_mfcc)
        # scale
        X = scaler.transform(X.reshape(-1, N_MFCC)).reshape(X.shape)
        # predict
        probs = model.predict(X)[0]
        pred = int(np.argmax(probs))
        label = le.inverse_transform([pred])[0]
        confidence = float(probs[pred])

        # top-3 predictions (emotion names and confidences)
        top_indices = np.argsort(probs)[::-1][:3]
        top_predictions = []
        for idx in top_indices:
            emotion = le.inverse_transform([int(idx)])[0]
            conf = float(probs[int(idx)])
            top_predictions.append({
                "emotion": emotion,
                "confidence": round(conf, 4)
            })

        return jsonify({
            "emotion": label,
            "confidence": round(confidence, 4),
            "model_version": MODEL_VERSION,
            "top_predictions": top_predictions
        })
    except Exception as e:
        tb = traceback.format_exc()
        app.logger.error('Prediction failed: %s', tb)
        return jsonify({'error': str(e), 'traceback': tb}), 500
    finally:
        # clean up temp files
        try:
            tmp.close()
            os.remove(tmp.name)
        except Exception:
            pass
        if out_wav:
            try:
                os.remove(out_wav)
            except Exception:
                pass


if __name__ == "__main__":
    print("Speech Emotion Recognition API running at http://127.0.0.1:8000")
    app.run(host="127.0.0.1", port=8000, debug=False)
