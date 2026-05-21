<<<<<<< HEAD
# SER-AI-Speech-Emotion-Recognition
A full-stack AI speech emotion recognition system with a React frontend, Flask API, and machine learning models for detecting emotions from voice audio.
=======
# Speech Emotion Recognition — Step 1

This step trains a RandomForest model using only the RAVDESS and CREMA-D datasets.

Files added:

- `src/config.py` — dataset paths and constants
- `src/data_loader.py` — load audio files and labels
- `src/feature_extraction.py` — extract 40 MFCC features using librosa
- `src/train_ml.py` — train RandomForest, print metrics, save model/encoder
- `src/predict.py` — predict emotion for a single audio file
- `requirements.txt` — Python dependencies
- `.gitignore`

Usage

1. Create a virtual environment and activate it (Windows):

```powershell
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Train the model:

```powershell
python src/train_ml.py
```

4. Predict on a single file:

```powershell
python src/predict.py "path_to_audio_file.wav"
```

Notes

- Dataset paths are defined in `src/config.py` (RAVDESS_PATH and CREMA_D_PATH).
- Model and encoder are saved to `models/emotion_model.pkl` and `models/label_encoder.pkl`.
- The code will create the `models/` folder if it does not exist.

**Model Comparison**

Below are the final comparison results for the experiments run on RAVDESS + CREMA‑D using a speaker-independent 80/20 split (no speaker overlap between train and test).

- **RandomForest + MFCC mean**: Accuracy 39.23%
- **Conv1D + MFCC sequence**: Accuracy 51.11%
- **CNN + Mel Spectrogram**: Accuracy 40.34%
- **Improved Conv1D + MFCC sequence + augmentation + class weights**: Accuracy 51.59%  ← *Best model*

- **CNN + BiLSTM MFCC sequence**: Accuracy 50.53% — trained on RAVDESS + CREMA-D with speaker-independent split. Did not outperform the Improved Conv1D.

-- New research result (wav2vec2):

Note: A research experiment using pretrained `facebook/wav2vec2-base` to extract 768-dim embeddings for each audio file followed by classical classifiers was run on the same RAVDESS + CREMA‑D data (speaker-independent 80/20 split). The new best research model achieved higher accuracy than the previous Conv1D results.

Key findings (final full-run):

- **wav2vec2 embeddings + LinearSVC**: Accuracy 54.54%  ← *New best (research)*
- **LogisticRegression**: 53.33%
- **PCA384 + MLP**: 52.56%
# SER-AI — Speech Emotion Recognition

Short project description
-------------------------
SER-AI is a full-stack Speech Emotion Recognition (SER) demo: a React frontend, a Flask backend API, and model/training scripts for experimenting with audio-based emotion classification.

Key features
------------
- Upload or record audio and get emotion predictions from the backend `/predict` endpoint.
- Local demo frontend (React + Vite) and a Flask-based prediction API.
- Example training and evaluation scripts for several models and feature pipelines.

Tech stack
----------
- Frontend: React, Vite, plain CSS
- Backend: Flask (Python)
- ML: scikit-learn, PyTorch/TensorFlow experiments (training scripts included)
- Audio tools: librosa, pydub (server-side conversion requires `ffmpeg`)

Project structure
-----------------
- `website/frontend/` — React app (source in `website/frontend/src`)
- `website/backend/` — Flask API and backend requirements
- `src/` — training and model scripts, feature extraction, prediction helpers
- `models/` — model files are NOT included in the repo (see notes below)
- `data/` — datasets (not included)

Models trained / comparison summary
----------------------------------
Experiments were run across multiple model families and feature pipelines (speaker-independent splits). Summary of notable results:

- RandomForest + MFCC mean: Accuracy 39.23%
- Conv1D + MFCC sequence: Accuracy 51.11%
- CNN + Mel Spectrogram: Accuracy 40.34%
- Improved Conv1D + MFCC sequence + augmentation + class weights: (experiment result present in repo)

Best model summary
------------------
Best experiment (reported):

- Wav2Vec2 embeddings + tuned LinearSVC
	- Best main split accuracy: 55.89%
	- Repeated speaker-independent split mean accuracy: 57.59% ± 1.92%

How to run the backend
----------------------
1. Create and activate a virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
```

2. Install backend dependencies:

```powershell
pip install -r website/backend/requirements.txt
```

3. Ensure `ffmpeg` is available on PATH for server-side conversion (used by `pydub`).

4. Run the Flask app:

```powershell
python website/backend/app.py
```

How to run the frontend
-----------------------
1. Change to frontend folder and install:

```bash
cd website/frontend
npm install
```

2. Start the dev server:

```bash
npm run dev
```

3. Open the local URL shown by Vite (usually `http://localhost:5173`).

API endpoints
-------------
- `POST /predict` — Upload audio file using form key `file` (multipart/form-data). The backend converts and processes audio, then returns JSON:

```json
{
	"emotion": "happy",
	"confidence": 0.89,
	"model_version": "wav2vec2-lsvc-v1",
	"top_predictions": [{"emotion":"happy","confidence":0.89}, ...]
}
```

Notes about model files
-----------------------
- Large model files are intentionally excluded from the repository. Place your trained model files in a local `models/` directory (the repo `.gitignore` excludes `models/`).
- The backend expects models to be available at runtime; see `website/backend/app.py` for the expected filename/location.

Future improvements
-------------------
- Add CI with tests and a build pipeline for frontend and backend.
- Provide Dockerfile and `docker-compose.yml` including `ffmpeg` for reproducible deployment.
- Add sample demo audio clips and an optional hosted tutorial video set.
- Improve accessibility, add i18n support, and add session/history for users.

Contact / License
------------------
This repository is provided as-is for research and demo purposes. See license file if included.

- 10,322 audio samples
- 115 speakers
- 8 emotion classes
- Speaker-independent 80/20 split
- No speaker overlap

Notes:
- The repeated split validation mean accuracy is 57.59% ± 1.92%.
- Current backend may still use the Improved Conv1D model unless updated.
- Best research model is `wav2vec2` + tuned `LinearSVC`.
- Deploying `wav2vec2` requires loading the wav2vec2 feature extractor and the `LinearSVC` classifier in the backend; do not update the backend yet.

>>>>>>> dbd08c8 (feat: update SER AI frontend and prediction API docs)
