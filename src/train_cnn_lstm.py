#!/usr/bin/env python3
"""
Train a Conv1D + LSTM model on MFCC sequence features.
Saves model and artifacts to project-level folders.
Run from project root:
    python src/train_cnn_lstm.py
"""
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

import os
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Input, Conv1D, BatchNormalization, MaxPooling1D,
                                     Dropout, Bidirectional, LSTM, Dense, GlobalAveragePooling1D)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Config
SR = 22050
N_MFCC = 40
MAX_LEN = 216  # number of time frames per sample
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 60

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


def find_audio_files(root_dirs):
    files = []
    for root in root_dirs:
        p = PROJECT_ROOT / root
        if not p.exists():
            continue
        for f in p.rglob('*'):
            if f.suffix.lower() in ('.wav', '.mp3', '.flac'):
                files.append(f)
    return files


def extract_speaker_id(path: Path):
    # Try common patterns
    s = str(path)
    # RAVDESS has Actor_01 folder
    if 'Actor_' in s:
        idx = s.find('Actor_')
        return s[idx:idx+8]  # Actor_XX
    name = path.name
    # CREMA-D often starts with speaker id digits like 1001_...
    parts = name.split('_')
    if parts and parts[0].isdigit():
        return parts[0]
    # fallback to parent folder name
    return path.parent.name


def extract_label_from_path(path: Path):
    # attempt to map filename tokens to emotion label
    name = path.stem.lower()
    # common emotion words
    for emo in ('happy', 'sad', 'angry', 'fear', 'disgust', 'neutral', 'calm', 'surprise', 'surprised'):
        if emo in name:
            return 'surprise' if emo.startswith('surpr') else ('fear' if emo=='fear' else emo)
    # Try numeric codes in RAVDESS (e.g., 03 is happy?), but leave unknown
    return 'unknown'


def mfcc_extract(file_path, sr=SR, n_mfcc=N_MFCC, max_len=MAX_LEN):
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # transpose to (time, n_mfcc)
    mfcc = mfcc.T
    if mfcc.shape[0] < max_len:
        pad_width = max_len - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:max_len, :]
    return mfcc  # shape (max_len, n_mfcc)


def build_model(input_shape, n_classes):
    model = Sequential()
    model.add(Conv1D(64, kernel_size=5, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.25))

    model.add(Conv1D(128, kernel_size=5, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.25))

    model.add(Bidirectional(LSTM(128, return_sequences=False)))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    # Use data_loader to list RAVDESS and CREMA-D files only
    import data_loader
    print('Collecting files from RAVDESS and CREMA-D...')
    files = data_loader.get_all_files()  # list of (path, label, speaker)
    if not files:
        print('No dataset files found. Ensure data/Ravdess and data/Crema-d exist.')
        sys.exit(1)

    paths = [Path(p) for p, lbl, sp in files]
    labels = np.array([lbl for p, lbl, sp in files])
    speakers = np.array([sp for p, lbl, sp in files])

    total_samples = len(paths)
    total_speakers = len(np.unique(speakers))
    from collections import Counter
    print(f'Total samples: {total_samples}')
    print(f'Total speakers: {total_speakers}')
    print('Class distribution (all):')
    print(dict(Counter(labels)))

    # Safety checks
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise ValueError('Invalid training data: only one class found')
    if total_samples < 9000:
        print('Warning: insufficient total samples (<9000). Aborting training.')
        return

    # Encode labels with LabelEncoder (fit on all labels to keep consistent mapping)
    le = LabelEncoder()
    le.fit(unique_labels)

    # Speaker-independent split (groups are speaker ids)
    y_all = le.transform(labels)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(paths, y_all, groups=speakers))

    train_paths = [paths[i] for i in train_idx]
    test_paths = [paths[i] for i in test_idx]
    y_train_labels = labels[train_idx]
    y_test_labels = labels[test_idx]
    speakers_train = speakers[train_idx]
    speakers_test = speakers[test_idx]

    train_speakers = set(speakers_train)
    test_speakers = set(speakers_test)
    overlap = train_speakers.intersection(test_speakers)

    print(f'Train samples: {len(train_paths)}')
    print(f'Test samples: {len(test_paths)}')
    print(f'Train speakers: {len(train_speakers)}')
    print(f'Test speakers: {len(test_speakers)}')
    print(f'Overlap speakers (should be 0): {len(overlap)}')
    if len(overlap) > 0:
        raise ValueError('Speaker leakage detected')

    print('Class distribution (train):')
    print(dict(Counter(y_train_labels)))
    print('Class distribution (test):')
    print(dict(Counter(y_test_labels)))

    # Extract MFCC sequences for train and test
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    for p, lbl in zip(train_paths, y_train_labels):
        try:
            seq = mfcc_extract(p, sr=SR, n_mfcc=N_MFCC, max_len=MAX_LEN)
            X_train.append(seq)
            y_train.append(lbl)
        except Exception as e:
            print(f'Error extracting train {p}: {e}')
            X_train.append(np.zeros((MAX_LEN, N_MFCC), dtype=np.float32))
            y_train.append(lbl)

    for p, lbl in zip(test_paths, y_test_labels):
        try:
            seq = mfcc_extract(p, sr=SR, n_mfcc=N_MFCC, max_len=MAX_LEN)
            X_test.append(seq)
            y_test.append(lbl)
        except Exception as e:
            print(f'Error extracting test {p}: {e}')
            X_test.append(np.zeros((MAX_LEN, N_MFCC), dtype=np.float32))
            y_test.append(lbl)

    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print(f'Train samples (after feature extraction): {len(X_train)}')
    print(f'Test samples (after feature extraction): {len(X_test)}')

    # Prepare scaler: flatten time-series per-sample to single vector for scaling, then reshape back
    X_train_flat = X_train.reshape((X_train.shape[0], -1))
    X_test_flat = X_test.reshape((X_test.shape[0], -1))

    scaler = StandardScaler()
    scaler.fit(X_train_flat)
    X_train_scaled = scaler.transform(X_train_flat).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)

    # Save scaler
    scaler_path = MODELS_DIR / 'cnn_lstm_scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f'Saved scaler to {scaler_path}')

    # Encode train/test labels to integers
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    # Class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_enc), y=y_train_enc)
    class_weights_dict = {i: float(w) for i, w in enumerate(class_weights)}
    print('Class distribution (train):', dict(Counter(y_train)))

    # Build model
    n_classes = len(le.classes_)
    input_shape = (MAX_LEN, N_MFCC)
    model = build_model(input_shape, n_classes)
    model.summary()

    # Callbacks (monitor val_accuracy)
    model_path = MODELS_DIR / 'cnn_lstm_model.keras'
    ckpt = ModelCheckpoint(str(model_path), monitor='val_accuracy', save_best_only=True, verbose=1)
    es = EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1)

    # Train
    history = model.fit(
        X_train_scaled, y_train_enc,
        validation_data=(X_test_scaled, y_test_enc),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights_dict,
        callbacks=[ckpt, es],
        verbose=2
    )

    # Evaluate
    y_pred_prob = model.predict(X_test_scaled)
    y_pred = np.argmax(y_pred_prob, axis=1)
    acc = accuracy_score(y_test_enc, y_pred)
    report = classification_report(y_test_enc, y_pred, target_names=le.classes_)
    cm = confusion_matrix(y_test_enc, y_pred)

    print(f'Accuracy: {acc:.4f}')
    print(report)
    print('Confusion Matrix:')
    print(cm)

    # Save label encoder and report
    le_path = MODELS_DIR / 'cnn_lstm_label_encoder.pkl'
    joblib.dump(le, le_path)
    print(f'Saved label encoder to {le_path}')

    report_path = OUTPUTS_DIR / 'cnn_lstm_report.txt'
    with open(report_path, 'w', encoding='utf-8') as fh:
        fh.write(f'Total samples: {total_samples}\n')
        fh.write(f'Total speakers: {total_speakers}\n')
        fh.write(f'Train samples: {len(X_train)}\n')
        fh.write(f'Test samples: {len(X_test)}\n')
        fh.write(f'Train speakers: {len(train_speakers)}\n')
        fh.write(f'Test speakers: {len(test_speakers)}\n')
        fh.write(f'Overlap speakers: {len(overlap)}\n')
        fh.write('Class distribution (train):\n')
        for lbl, cnt in dict(Counter(y_train)).items():
            fh.write(f'  {lbl}: {cnt}\n')
        fh.write('\n')
        fh.write(f'Accuracy: {acc:.4f}\n\n')
        fh.write('Classification Report:\n')
        fh.write(report)
        fh.write('\nConfusion Matrix:\n')
        fh.write(np.array2string(cm))

    print(f'Report saved to {report_path}')
    print('Done.')


if __name__ == '__main__':
    main()
