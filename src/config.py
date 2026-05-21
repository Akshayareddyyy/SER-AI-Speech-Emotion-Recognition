import os

# Make dataset paths absolute relative to the repository root. This allows
# scripts to be executed both from the repository root and from the `src`
# directory without changing configuration.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAVDESS_PATH = os.path.join(ROOT, 'data', 'Ravdess')
CREMA_D_PATH = os.path.join(ROOT, 'data', 'Crema-d')
MODEL_PATH = "models/emotion_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
SCALER_PATH = "models/scaler.pkl"

# audio / feature settings
SR = 22050
N_MFCC = 40

# RAVDESS emotion mapping (third number in filename)
RAVDESS_EMOTION_MAP = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised',
}

# CREMA-D mapping (third underscore-separated part)
CREMA_D_EMOTION_MAP = {
    'NEU': 'neutral',
    'HAP': 'happy',
    'SAD': 'sad',
    'ANG': 'angry',
    'FEA': 'fearful',
    'DIS': 'disgust',
}
