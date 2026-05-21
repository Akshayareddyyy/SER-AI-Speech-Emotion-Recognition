import os
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import config
import data_loader
from feature_extraction import extract_mfcc_sequence
import collections

MAX_LEN = 216
N_MFCC = config.N_MFCC

def load_sequences(max_len=MAX_LEN):
    files = data_loader.get_all_files()
    X = []
    y = []
    groups = []
    for (path, label, speaker) in files:
        try:
            seq = extract_mfcc_sequence(path, n_mfcc=N_MFCC, sr=config.SR, max_len=max_len)
            X.append(seq)
            y.append(label)
            groups.append(speaker)
        except Exception as e:
            print(f"Error extracting {path}: {e}")
    return np.array(X, dtype=float), np.array(y), np.array(groups)

def build_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(64, kernel_size=5, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv1D(128, kernel_size=5, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    print('Loading file list and extracting MFCC sequences...')
    X, y, groups = load_sequences()
    if len(X) == 0:
        print('No data found. Check dataset paths.')
        return

    total_samples = len(X)
    total_speakers = len(np.unique(groups))
    print(f'Total samples: {total_samples}')
    print(f'Total speakers: {total_speakers}')

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # speaker-independent split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y_enc, groups=groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_enc[train_idx], y_enc[test_idx]
    groups_train, groups_test = groups[train_idx], groups[test_idx]

    train_speakers = set(groups_train)
    test_speakers = set(groups_test)
    overlap = train_speakers.intersection(test_speakers)

    print(f'Train speakers: {len(train_speakers)}')
    print(f'Test speakers: {len(test_speakers)}')
    print(f'Overlap speakers: {len(overlap)}')
    print(f'Train samples: {len(y_train)}, Test samples: {len(y_test)}')

    print('Class distribution (all):')
    for k, v in collections.Counter(y).items():
        print(f'  {k}: {v}')
    print('Class distribution (train):')
    for k, v in collections.Counter(le.inverse_transform(y_train)).items():
        print(f'  {k}: {v}')
    print('Class distribution (test):')
    for k, v in collections.Counter(le.inverse_transform(y_test)).items():
        print(f'  {k}: {v}')

    # scale features: fit scaler on training data only
    n_mfcc = X_train.shape[2]
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, n_mfcc)
    scaler.fit(X_train_flat)
    X_train = scaler.transform(X_train_flat).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, n_mfcc)).reshape(X_test.shape)

    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(le.classes_)
    model = build_model(input_shape, num_classes)
    print('Model summary:')
    model.summary()

    # training using test split as validation_data
    es = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[es]
    )

    # evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f'Final test accuracy: {acc:.4f}')
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print('Classification report:')
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print('Confusion matrix:')
    print(confusion_matrix(y_test, y_pred))

    # save model and artifacts
    models_dir = os.path.dirname(config.MODEL_PATH)
    if models_dir:
        os.makedirs(models_dir, exist_ok=True)
    model.save(os.path.join(models_dir, 'conv1d_emotion_model.keras'))
    joblib.dump(le, os.path.join(models_dir, 'conv1d_label_encoder.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'conv1d_scaler.pkl'))
    print('Saved conv1d model and artifacts.')

if __name__ == '__main__':
    main()
