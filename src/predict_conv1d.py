import sys
import os
import joblib
import numpy as np
from tensorflow import keras
import config
from feature_extraction import extract_mfcc_sequence

MAX_LEN = 216
N_MFCC = config.N_MFCC

def predict_file(path):
    if not os.path.isfile(path):
        print(f'File not found: {path}')
        return
    models_dir = os.path.dirname(config.MODEL_PATH)
    model_path = os.path.join(models_dir, 'conv1d_emotion_model.keras')
    le_path = os.path.join(models_dir, 'conv1d_label_encoder.pkl')
    scaler_path = os.path.join(models_dir, 'conv1d_scaler.pkl')
    if not (os.path.exists(model_path) and os.path.exists(le_path) and os.path.exists(scaler_path)):
        print('Conv1D model or artifacts not found. Run src/train_conv1d.py first.')
        return
    model = keras.models.load_model(model_path)
    le = joblib.load(le_path)
    scaler = joblib.load(scaler_path)

    seq = extract_mfcc_sequence(path, n_mfcc=N_MFCC, sr=config.SR, max_len=MAX_LEN)
    X = np.expand_dims(seq, axis=0)  # (1, max_len, n_mfcc)
    # scale per feature
    X = scaler.transform(X.reshape(-1, N_MFCC)).reshape(X.shape)
    probs = model.predict(X)[0]
    pred = int(np.argmax(probs))
    label = le.inverse_transform([pred])[0]
    print(f'Predicted: {label} (prob={probs[pred]:.3f})')
    return label

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python src/predict_conv1d.py path_to_audio.wav')
    else:
        predict_file(sys.argv[1])
