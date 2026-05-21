import os
import argparse
import librosa
import numpy as np
import joblib
import tensorflow as tf
import config

SR = config.SR
N_MFCC = getattr(config, 'N_MFCC', 40)
MAX_LEN = 216

MODEL_IN = "models/conv1d_improved_model.keras"
ENC_IN = "models/conv1d_improved_label_encoder.pkl"
SCALER_IN = "models/conv1d_improved_scaler.pkl"


def extract_mfcc_from_audio(y, sr=SR, n_mfcc=N_MFCC, max_len=MAX_LEN):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_t = mfcc.T
    t = mfcc_t.shape[0]
    if t >= max_len:
        mfcc_t = mfcc_t[:max_len, :]
    else:
        pad_width = max_len - t
        pad = np.zeros((pad_width, n_mfcc), dtype=mfcc_t.dtype)
        mfcc_t = np.vstack([mfcc_t, pad])
    return mfcc_t.astype(np.float32)


def predict_file(path):
    if not os.path.exists(MODEL_IN):
        raise FileNotFoundError(f"Model not found: {MODEL_IN}")
    if not os.path.exists(ENC_IN):
        raise FileNotFoundError(f"Encoder not found: {ENC_IN}")
    if not os.path.exists(SCALER_IN):
        raise FileNotFoundError(f"Scaler not found: {SCALER_IN}")

    model = tf.keras.models.load_model(MODEL_IN)
    le = joblib.load(ENC_IN)
    scaler = joblib.load(SCALER_IN)

    y, _ = librosa.load(path, sr=SR)
    seq = extract_mfcc_from_audio(y, sr=SR, n_mfcc=N_MFCC, max_len=MAX_LEN)
    # scale
    seq_scaled = scaler.transform(seq.reshape(-1, seq.shape[-1])).reshape(seq.shape)
    seq_scaled = np.expand_dims(seq_scaled, axis=0)  # (1, 216, 40)

    preds = model.predict(seq_scaled)
    idx = int(np.argmax(preds, axis=1)[0])
    label = le.inverse_transform([idx])[0]
    confidence = float(np.max(preds))

    return label, confidence


def main():
    p = argparse.ArgumentParser()
    p.add_argument('file', help='Path to audio file')
    args = p.parse_args()

    label, conf = predict_file(args.file)
    print({'emotion': label, 'confidence': conf})


if __name__ == '__main__':
    main()
