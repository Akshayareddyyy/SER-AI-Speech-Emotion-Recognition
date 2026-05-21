import os
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import data_loader
import config
import collections

def main():
    print("Loading data from RAVDESS and CREMA-D...")
    X, y, groups = data_loader.load_data()
    if len(X) == 0:
        print("No audio data found. Check dataset paths in src/config.py")
        return
    X = np.array(X, dtype=float)
    y = np.array(y)
    groups = np.array(groups)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # overall counts
    total_samples = len(y_enc)
    total_speakers = len(np.unique(groups))
    class_counts = collections.Counter(y)
    print(f"Total samples: {total_samples}")
    print(f"Total speakers: {total_speakers}")
    print("Class distribution (all):")
    for k, v in class_counts.items():
        print(f"  {k}: {v}")

    # speaker-independent split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y_enc, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_enc[train_idx], y_enc[test_idx]
    groups_train, groups_test = groups[train_idx], groups[test_idx]

    train_speakers = set(groups_train)
    test_speakers = set(groups_test)
    overlap = train_speakers.intersection(test_speakers)

    print(f"Train speakers: {len(train_speakers)}")
    print(f"Test speakers: {len(test_speakers)}")
    print(f"Overlap speakers: {len(overlap)}")
    print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")

    # class distribution in train/test
    inv_classes = le.inverse_transform
    train_counts = collections.Counter(inv_classes(y_train))
    test_counts = collections.Counter(inv_classes(y_test))
    print("Class distribution (train):")
    for k, v in train_counts.items():
        print(f"  {k}: {v}")
    print("Class distribution (test):")
    for k, v in test_counts.items():
        print(f"  {k}: {v}")

    # scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Training RandomForestClassifier...")
    clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # ensure models directory
    models_dir = os.path.dirname(config.MODEL_PATH)
    if models_dir:
        os.makedirs(models_dir, exist_ok=True)

    joblib.dump(clf, config.MODEL_PATH)
    joblib.dump(le, config.ENCODER_PATH)
    joblib.dump(scaler, config.SCALER_PATH)
    print(f"Saved model to {config.MODEL_PATH}")
    print(f"Saved label encoder to {config.ENCODER_PATH}")
    print(f"Saved scaler to {config.SCALER_PATH}")

if __name__ == '__main__':
    main()
