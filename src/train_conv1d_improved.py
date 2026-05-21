import os
import numpy as np
import librosa
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.metrics import classification_report, confusion_matrix
import config
import data_loader

"""
Improved Conv1D training script with simple data augmentation.

Design notes:
- Uses speaker-independent split (GroupShuffleSplit) to ensure no speaker
  overlap between train and test sets — this avoids overoptimistic results
  caused by the model memorizing speaker-specific cues.
- Augmentations (noise, time shift) are applied only to training data.
- StandardScaler is fitted only on the training set after augmentation
  to prevent data leakage from the test set.
"""

SR = config.SR
N_MFCC = getattr(config, 'N_MFCC', 40)
MAX_LEN = 216

MODEL_OUT = "models/conv1d_improved_model.keras"
ENC_OUT = "models/conv1d_improved_label_encoder.pkl"
SCALER_OUT = "models/conv1d_improved_scaler.pkl"


def extract_mfcc_from_audio(y, sr=SR, n_mfcc=N_MFCC, max_len=MAX_LEN):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_t = mfcc.T  # (t, n_mfcc)
    t = mfcc_t.shape[0]
    if t >= max_len:
        mfcc_t = mfcc_t[:max_len, :]
    else:
        pad_width = max_len - t
        pad = np.zeros((pad_width, n_mfcc), dtype=mfcc_t.dtype)
        mfcc_t = np.vstack([mfcc_t, pad])
    return mfcc_t.astype(np.float32)


def add_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    augmented = y + noise_factor * noise
    return augmented.astype(y.dtype)


def time_shift(y, shift_max=0.2):
    # shift_max is fraction of total length
    n_shift = int(len(y) * shift_max)
    shift = np.random.randint(-n_shift, n_shift)
    return np.roll(y, shift)


def build_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(256, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    files = data_loader.get_all_files()
    paths = [p for p, lbl, sp in files]
    labels = [lbl for p, lbl, sp in files]
    groups = [sp for p, lbl, sp in files]

    total_samples = len(paths)
    total_speakers = len(set(groups))

    print(f"Total samples (before augmentation): {total_samples}")
    print(f"Total speakers: {total_speakers}")

    if total_samples == 0:
        print('No files found. Check dataset paths.')
        return

    # speaker-independent split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X=paths, y=labels, groups=groups))

    train_paths = [paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    train_groups = [groups[i] for i in train_idx]

    test_paths = [paths[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    test_groups = [groups[i] for i in test_idx]

    train_speakers = set(train_groups)
    test_speakers = set(test_groups)
    overlap = train_speakers.intersection(test_speakers)

    print(f"Train samples (before augmentation): {len(train_paths)}")
    print(f"Test samples: {len(test_paths)}")
    print(f"Train speakers: {len(train_speakers)}")
    print(f"Test speakers: {len(test_speakers)}")
    print(f"Overlap speakers (should be 0): {len(overlap)}")

    # Extract MFCC sequences for test set (no augmentation)
    X_test = []
    for p in test_paths:
        try:
            y, _ = librosa.load(p, sr=SR)
            seq = extract_mfcc_from_audio(y, sr=SR, n_mfcc=N_MFCC, max_len=MAX_LEN)
            X_test.append(seq)
        except Exception as e:
            print(f"Error extracting test {p}: {e}")
            X_test.append(np.zeros((MAX_LEN, N_MFCC), dtype=np.float32))

    # For training, apply augmentations (original + noise + time shift)
    X_train = []
    y_train = []
    for p, lbl in zip(train_paths, train_labels):
        try:
            y_raw, _ = librosa.load(p, sr=SR)
            # original
            seq_orig = extract_mfcc_from_audio(y_raw, sr=SR, n_mfcc=N_MFCC, max_len=MAX_LEN)
            X_train.append(seq_orig)
            y_train.append(lbl)

            # noise
            y_noise = add_noise(y_raw, noise_factor=0.005)
            seq_noise = extract_mfcc_from_audio(y_noise, sr=SR, n_mfcc=N_MFCC, max_len=MAX_LEN)
            X_train.append(seq_noise)
            y_train.append(lbl)

            # time shift
            y_shift = time_shift(y_raw, shift_max=0.2)
            seq_shift = extract_mfcc_from_audio(y_shift, sr=SR, n_mfcc=N_MFCC, max_len=MAX_LEN)
            X_train.append(seq_shift)
            y_train.append(lbl)

        except Exception as e:
            print(f"Error extracting train {p}: {e}")
            # add zeros to keep lengths
            X_train.append(np.zeros((MAX_LEN, N_MFCC), dtype=np.float32))
            y_train.append(lbl)

    print(f"Train samples (after augmentation): {len(X_train)}")

    # Convert to arrays
    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_train = np.array(y_train)
    y_test = np.array(test_labels)

    # Show class distributions
    print('Class distribution (before augmentation):')
    print(Counter(train_labels))
    print('Class distribution (after augmentation):')
    print(Counter(y_train))

    # Fit scaler on training data only
    scaler = StandardScaler()
    # reshape to (n_samples * time, n_mfcc) for fitting per-feature
    n_samples, t, n_mfcc = X_train.shape
    X_train_flat = X_train.reshape(-1, n_mfcc)
    scaler.fit(X_train_flat)

    # transform train and test
    X_train = scaler.transform(X_train_flat).reshape(n_samples, t, n_mfcc)
    X_test = scaler.transform(X_test.reshape(-1, n_mfcc)).reshape(X_test.shape)

    # Encode labels
    le = LabelEncoder()
    le.fit(np.unique(labels))
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    # Compute class weights based on augmented training labels
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_enc), y=y_train_enc)
    class_weights_dict = {i: w for i, w in enumerate(class_weights)}

    print('Model input shape:', X_train.shape[1:])
    print('Num classes:', len(le.classes_))

    # Build model
    input_shape = (MAX_LEN, N_MFCC)
    num_classes = len(le.classes_)
    model = build_model(input_shape, num_classes)
    print('Model summary:')
    model.summary()

    # Callbacks
    es = callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    rlrop = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    # Train
    history = model.fit(
        X_train, y_train_enc,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test_enc),
        callbacks=[es, rlrop],
        class_weight=class_weights_dict,
        verbose=2
    )

    # Evaluate
    loss, acc = model.evaluate(X_test, y_test_enc, verbose=0)
    print(f'Final test accuracy: {acc * 100:.2f}%')

    preds = model.predict(X_test)
    y_pred = np.argmax(preds, axis=1)

    print('Classification report:')
    print(classification_report(y_test_enc, y_pred, target_names=le.classes_))
    print('Confusion matrix:')
    print(confusion_matrix(y_test_enc, y_pred))

    # Save artifacts
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    model.save(MODEL_OUT)
    joblib.dump(le, ENC_OUT)
    joblib.dump(scaler, SCALER_OUT)
    print('Saved improved conv1d model and artifacts.')


if __name__ == '__main__':
    main()
