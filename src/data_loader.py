import os
from feature_extraction import extract_mfcc
import config

def load_ravdess(path=config.RAVDESS_PATH):
    X = []
    y = []
    groups = []
    if not os.path.isdir(path):
        print(f"RAVDESS path not found: {path}")
        return X, y, groups
    for root, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith('.wav'):
                base = os.path.splitext(f)[0]
                parts = base.split('-')
                if len(parts) >= 3:
                    emo_code = parts[2]
                    label = config.RAVDESS_EMOTION_MAP.get(emo_code)
                    if label:
                        # speaker id is last field
                        speaker_id = parts[-1]
                        speaker = f"ravdess_{speaker_id}"
                        full = os.path.join(root, f)
                        try:
                            feat = extract_mfcc(full)
                            X.append(feat)
                            y.append(label)
                            groups.append(speaker)
                        except Exception as e:
                            print(f"Error processing {full}: {e}")
    return X, y, groups

def load_crema_d(path=config.CREMA_D_PATH):
    X = []
    y = []
    groups = []
    if not os.path.isdir(path):
        print(f"CREMA-D path not found: {path}")
        return X, y, groups
    for root, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith('.wav'):
                base = os.path.splitext(f)[0]
                parts = base.split('_')
                if len(parts) >= 1:
                    emo_code = parts[2].upper() if len(parts) >= 3 else None
                    label = config.CREMA_D_EMOTION_MAP.get(emo_code)
                    if label:
                        speaker_id = parts[0]
                        speaker = f"crema_{speaker_id}"
                        full = os.path.join(root, f)
                        try:
                            feat = extract_mfcc(full)
                            X.append(feat)
                            y.append(label)
                            groups.append(speaker)
                        except Exception as e:
                            print(f"Error processing {full}: {e}")
    return X, y, groups


def list_ravdess_files(path=config.RAVDESS_PATH):
    """Return list of (full_path, label, speaker) for RAVDESS files."""
    files = []
    if not os.path.isdir(path):
        return files
    for root, _, fnames in os.walk(path):
        for f in fnames:
            if f.lower().endswith('.wav'):
                base = os.path.splitext(f)[0]
                parts = base.split('-')
                if len(parts) >= 3:
                    emo_code = parts[2]
                    label = config.RAVDESS_EMOTION_MAP.get(emo_code)
                    if label:
                        speaker_id = parts[-1]
                        speaker = f"ravdess_{speaker_id}"
                        full = os.path.join(root, f)
                        files.append((full, label, speaker))
    return files


def list_crema_files(path=config.CREMA_D_PATH):
    """Return list of (full_path, label, speaker) for CREMA-D files."""
    files = []
    if not os.path.isdir(path):
        return files
    for root, _, fnames in os.walk(path):
        for f in fnames:
            if f.lower().endswith('.wav'):
                base = os.path.splitext(f)[0]
                parts = base.split('_')
                emo_code = parts[2].upper() if len(parts) >= 3 else None
                label = config.CREMA_D_EMOTION_MAP.get(emo_code)
                if label:
                    speaker_id = parts[0]
                    speaker = f"crema_{speaker_id}"
                    full = os.path.join(root, f)
                    files.append((full, label, speaker))
    return files


def get_all_files():
    """Return combined list of (full_path, label, speaker) from both datasets."""
    a = list_ravdess_files()
    b = list_crema_files()
    return a + b

def load_data():
    X1, y1, g1 = load_ravdess()
    X2, y2, g2 = load_crema_d()
    X = X1 + X2
    y = y1 + y2
    groups = g1 + g2
    return X, y, groups
