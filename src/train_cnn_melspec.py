import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from collections import Counter
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    Dropout,
    GlobalAveragePooling2D,
    Dense,
)
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import get_all_files

# Training script for Mel-spectrogram + 2D CNN
# - Uses speaker-independent split (GroupShuffleSplit) to avoid speaker overlap
#   between train and test sets. This prevents the model from memorizing
#   speaker-specific characteristics and gives a realistic estimate of
#   generalization to unseen speakers.

# Constants
SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
MAX_LEN = 216  # time frames

MODEL_OUT = "models/cnn_melspec_model.keras"
ENC_OUT = "models/cnn_melspec_label_encoder.pkl"
NORM_OUT = "models/cnn_melspec_norm.pkl"


def make_mel(path):
    # Load audio at the target sample rate
    y, _ = librosa.load(path, sr=SR)
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    m_db = librosa.power_to_db(m, ref=np.max)
    # m_db shape: (n_mels, t)
    t = m_db.shape[1]
    if t >= MAX_LEN:
        m_db = m_db[:, :MAX_LEN]
    else:
        pad_width = MAX_LEN - t
        pad = np.full((N_MELS, pad_width), fill_value=np.min(m_db), dtype=m_db.dtype)
        m_db = np.concatenate([m_db, pad], axis=1)
    return m_db.astype(np.float32)


def build_model(input_shape, num_classes):
    inp = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), activation="relu", padding="same")(inp)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.4)(x)
    out = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    # gather files
    files = get_all_files()
    paths = [p for p, lbl, sp in files]
    labels = [lbl for p, lbl, sp in files]
    groups = [sp for p, lbl, sp in files]

    total_samples = len(paths)
    total_speakers = len(set(groups))

    print(f"Total samples: {total_samples}")
    print(f"Total speakers: {total_speakers}")

    if total_samples == 0:
        print("No audio files found. Check data paths.")
        return

    # speaker-independent split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X=paths, y=labels, groups=groups))

    train_speakers = set([groups[i] for i in train_idx])
    test_speakers = set([groups[i] for i in test_idx])
    overlap = train_speakers.intersection(test_speakers)

    print(f"Train speakers: {len(train_speakers)}")
    print(f"Test speakers: {len(test_speakers)}")
    print(f"Overlap speakers (should be 0): {len(overlap)}")

    # Precompute spectrograms
    X = []
    for i, p in enumerate(paths):
        try:
            spec = make_mel(p)
            X.append(spec)
        except Exception as e:
            print(f"Error processing {p}: {e}")
            X.append(np.zeros((N_MELS, MAX_LEN), dtype=np.float32))

    X = np.stack(X, axis=0)  # (N, n_mels, max_len)
    y = np.array(labels)

    # Map indices
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Why mel spectrogram?
    # Mel-spectrograms provide a compact time-frequency representation
    # that aligns with human auditory perception (mel scale). They are
    # effective inputs for 2D CNNs because nearby frequency bins and
    # time frames exhibit local patterns useful for convolutional filters.

    # Normalization: fit mean/std on training data only to avoid leakage
    # We compute global mean/std across all pixels of training set. Fitting
    # normalization using test data would leak information from the test
    # set into the training process and give optimistic performance estimates.
    mean = X_train.mean()
    std = X_train.std()
    print(f"Normalization mean: {mean:.4f}, std: {std:.4f}")

    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)

    # Add channel axis
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # Encode labels
    le = LabelEncoder()
    le.fit(y)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    # Print class distribution
    print("Class distribution:")
    print(Counter(y_train))

    input_shape = (N_MELS, MAX_LEN, 1)
    num_classes = len(le.classes_)
    model = build_model(input_shape, num_classes)
    model.summary()

    # callbacks
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    # Train
    history = model.fit(
        X_train,
        y_train_enc,
        epochs=30,
        batch_size=32,
        validation_data=(X_test, y_test_enc),
        callbacks=[es],
    )

    # Evaluate
    loss, acc = model.evaluate(X_test, y_test_enc, verbose=0)
    print(f"Final test accuracy: {acc * 100:.2f}%")

    # Predictions for classification report
    preds = model.predict(X_test)
    y_pred = np.argmax(preds, axis=1)

    print("Classification report:")
    print(classification_report(y_test_enc, y_pred, target_names=le.classes_))
    print("Confusion matrix:")
    print(confusion_matrix(y_test_enc, y_pred))

    # ensure models dir
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

    print(f"Saving model to {MODEL_OUT}")
    model.save(MODEL_OUT)
    joblib.dump(le, ENC_OUT)
    joblib.dump({"mean": float(mean), "std": float(std)}, NORM_OUT)

    print("Done.")


if __name__ == '__main__':
    main()
