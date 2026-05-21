import argparse
import os
import joblib
import numpy as np
from feature_extraction import extract_mfcc
import config
from sklearn.preprocessing import StandardScaler

def predict_file(audio_path):
    if not os.path.isfile(audio_path):
        print(f"File not found: {audio_path}")
        return
    if not (os.path.exists(config.MODEL_PATH) and os.path.exists(config.ENCODER_PATH) and os.path.exists(config.SCALER_PATH)):
        print("Model, encoder, or scaler not found. Run training first: python src/train_ml.py")
        return
    clf = joblib.load(config.MODEL_PATH)
    le = joblib.load(config.ENCODER_PATH)
    scaler = joblib.load(config.SCALER_PATH)
    feat = extract_mfcc(audio_path)
    X = np.expand_dims(feat, axis=0)
    X = scaler.transform(X)
    pred = clf.predict(X)[0]
    label = le.inverse_transform([pred])[0]
    if hasattr(clf, 'predict_proba'):
        probs = clf.predict_proba(X)[0]
        prob = probs[pred]
        print(f"Predicted: {label} (probability={prob:.3f})")
    else:
        print(f"Predicted: {label}")
    return label

def main():
    parser = argparse.ArgumentParser(description='Predict emotion for an audio file')
    parser.add_argument('audio_path', help='Path to audio file')
    args = parser.parse_args()
    predict_file(args.audio_path)

if __name__ == '__main__':
    main()
