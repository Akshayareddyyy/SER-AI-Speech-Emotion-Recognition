import librosa
import numpy as np
import config

def extract_mfcc(audio_path, n_mfcc=config.N_MFCC, sr=config.SR):
    """Load an audio file and extract the mean MFCC feature vector."""
    y, _ = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean

def extract_mfcc_sequence(audio_path, n_mfcc=config.N_MFCC, sr=config.SR, max_len=216):
    """Load audio and return MFCC sequence of shape (max_len, n_mfcc).

    Pads with zeros or truncates along time axis to produce fixed-length sequences.
    """
    y, _ = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # mfcc shape is (n_mfcc, t) -> transpose to (t, n_mfcc)
    mfcc_t = mfcc.T
    t = mfcc_t.shape[0]
    if t >= max_len:
        mfcc_t = mfcc_t[:max_len, :]
    else:
        pad_width = max_len - t
        pad = np.zeros((pad_width, n_mfcc), dtype=mfcc_t.dtype)
        mfcc_t = np.vstack([mfcc_t, pad])
    return mfcc_t
