# Audio Emotion Detection

End-to-end pipeline for detecting human emotions from speech recordings. The project ingests the [RAVDESS](https://zenodo.org/records/1188976) emotional speech dataset, extracts rich spectral features, trains a diverse zoo of classical and deep models, and serves real-time predictions through a Streamlit UI.

## Table of Contents

1. [Project Highlights](#project-highlights)
2. [Repository Structure](#repository-structure)
3. [Dataset & Annotation Scheme](#dataset--annotation-scheme)
4. [Feature Engineering](#feature-engineering)
5. [Model Suite](#model-suite)
6. [Environment Setup](#environment-setup)
7. [Training Workflow](#training-workflow)
8. [Verifying Saved Models](#verifying-saved-models)
9. [Interactive Inference App](#interactive-inference-app)
10. [Cached Artifacts & Outputs](#cached-artifacts--outputs)
11. [Troubleshooting & Tips](#troubleshooting--tips)

## Project Highlights

- Unified extraction pipeline (`audio_processor.AudioProcessor`) ensures identical features for training and inference.
- Ten supervised learners (5 deep learning models + 5 classical ML baselines) trained with consistent splits and preprocessing for fair comparison.
- Cached NumPy feature matrices (`X.npy`, `y.npy`) accelerate iterative experimentation.
- Streamlit dashboard (`app.py`) handles arbitrary audio uploads, waveform/spectrogram visualizations, and probability breakdowns.
- Dedicated verification script confirms that every serialized model and scaler can be loaded before deployment.

## Repository Structure

| Path                          | Description                                                                                                                                                                                  |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `app.py`                      | Streamlit UI for uploading audio, selecting a trained model, visualizing waveforms/spectrograms, and displaying per-emotion probabilities.                                                   |
| `audio_processor.py`          | Central place for feature extraction (MFCCs, chroma, spectral contrast, tonnetz, ZCR, RMS, centroid, rolloff), emotion label parsing from filenames, and helper utilities.                   |
| `model_definitions.py`        | Contains factory functions for each neural and classical model architecture used during training. Deep models are built with Keras, classical models with scikit-learn.                      |
| `train_all_models.py`         | Orchestrates the full workflow: dataset scan, caching, preprocessing (scaling + label encoding), reshaping for sequential nets, model training, evaluation, and persistence under `models/`. |
| `verify_models.py`            | Loads the shared scaler plus every saved model (Keras + sklearn) to ensure they can produce predictions on dummy data. Useful for CI smoke tests.                                            |
| `requirements.txt`            | Minimal dependency set for both the training pipeline and Streamlit app.                                                                                                                     |
| `data/RAVDESS/Actor_XX/*.wav` | Original dataset organized per RAVDESS actor. Filenames encode emotion IDs that `AudioProcessor.get_emotion_from_filename` parses.                                                           |
| `models/`                     | Stores serialized artifacts: Keras `.h5` files for DL models, `.pkl` files for sklearn models, plus the shared scaler & label encoder.                                                       |
| `X.npy`, `y.npy`              | Cached feature matrix and encoded labels; automatically regenerated if missing.                                                                                                              |
| `samples/`                    | Placeholder for any ad-hoc demo clips (not required for training).                                                                                                                           |

## Dataset & Annotation Scheme

- **Source:** Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) speech subset located at `data/RAVDESS`.
- **Label Extraction:** Filenames follow `03-01-EMOTION-...-Actor_XX.wav`. `AudioProcessor.get_emotion_from_filename` maps the two-digit `EMOTION` code to textual labels (`neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`). Files with unrecognized codes are skipped.
- **Split:** `train_test_split` with `test_size=0.2`, stratified by emotion, `random_state=42` for determinism.

## Feature Engineering

All models operate on the same 110-dimensional descriptor vector returned by `AudioProcessor.extract_features`:

- **MFCCs (13)** – per-coefficient mean & std (26 values).
- **ΔMFCCs (13)** – per-coefficient mean & std (26 values).
- **Chroma STFT (12)** – mean & std (24 values).
- **Spectral Contrast (7)** – mean & std (14 values).
- **Tonnetz (6)** – mean & std (12 values).
- **Global stats (8)** – Zero Crossing Rate (mean/std), RMS Energy (mean/std), Spectral Centroid (mean/std), Spectral Rolloff (mean/std).
- Audio is resampled to 22,050 Hz, converted to mono if necessary, and trimmed/padded to a consistent duration (via librosa defaults) before feature computation.

The feature vector is standardized with a single `StandardScaler` instance that is saved to `models/scaler.pkl` and reused during inference.

## Model Suite

Each learner has a dedicated deep dive in `docs/models/*.md`:

| Model                  | Type          | Documentation                                        |
| ---------------------- | ------------- | ---------------------------------------------------- |
| Multi-Layer Perceptron | Deep Learning | [`docs/models/mlp.md`](docs/models/mlp.md)           |
| 1D CNN                 | Deep Learning | [`docs/models/cnn1d.md`](docs/models/cnn1d.md)       |
| LSTM                   | Deep Learning | [`docs/models/lstm.md`](docs/models/lstm.md)         |
| GRU                    | Deep Learning | [`docs/models/gru.md`](docs/models/gru.md)           |
| CNN-LSTM Hybrid        | Deep Learning | [`docs/models/cnn_lstm.md`](docs/models/cnn_lstm.md) |
| Random Forest          | Classical ML  | [`docs/models/rf.md`](docs/models/rf.md)             |
| Support Vector Machine | Classical ML  | [`docs/models/svm.md`](docs/models/svm.md)           |
| Gradient Boosting      | Classical ML  | [`docs/models/gb.md`](docs/models/gb.md)             |
| K-Nearest Neighbors    | Classical ML  | [`docs/models/knn.md`](docs/models/knn.md)           |
| Extra Trees            | Classical ML  | [`docs/models/et.md`](docs/models/et.md)             |

All deep models share identical training hyper-parameters (Adam optimizer, up to 50 epochs, batch size 32, EarlyStopping on `val_accuracy`, ModelCheckpoint). Classical models use scikit-learn defaults with minor tuning (number of estimators, depth, k, etc.).

## Environment Setup

```pwsh
# 1. Create and activate a virtual environment (optional but recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Dependencies include TensorFlow (for Keras models), scikit-learn, librosa/soundfile (audio processing), and Streamlit/Matplotlib for the UI.

## Training Workflow

1. **Prepare Dataset:** Download/extract RAVDESS speech files into `data/RAVDESS`. Preserve the original actor subfolders.
2. **Run Training Script:**
   ```pwsh
   python train_all_models.py
   ```
   - If `X.npy` and `y.npy` exist, they are reused. Otherwise, the script walks `data/RAVDESS`, extracts features with `AudioProcessor`, and caches the numpy arrays.
   - Labels are encoded via `LabelEncoder` and saved to `models/label_encoder.pkl`.
   - Features are scaled with `StandardScaler` and saved to `models/scaler.pkl`.
   - Deep models receive reshaped data `(samples, features, 1)`; classical models operate on the flat scaled vectors.
3. **Results:** Test accuracies are printed per model and summarized at the end. Best-performing checkpoints are stored inside `models/` with consistent naming.

## Verifying Saved Models

After training or pulling artifacts from elsewhere, confirm they load correctly:

```pwsh
python verify_models.py
```

- Ensures the shared scaler exists and each `.h5` / `.pkl` file can produce predictions on dummy input.
- Helpful before packaging the Streamlit app or deploying to production.

## Interactive Inference App

Launch the UI to test arbitrary audio clips:

```pwsh
streamlit run app.py
```

- Choose any of the trained models from the sidebar dropdown.
- Upload audio (`wav`, `mp3`, `ogg`, `flac`, `m4a`, `aac`). Non-WAV formats are converted via `pydub`.
- The app displays the detected emotion, confidence, waveform, spectrogram, and the full probability distribution across the eight supported emotions.
- Internally reuses `AudioProcessor` for feature extraction and the persisted scaler for normalization, guaranteeing parity with training.

## Cached Artifacts & Outputs

- `models/scaler.pkl` – feature standardization parameters.
- `models/label_encoder.pkl` – mapping between integer IDs and emotion strings.
- `models/*_model.h5` – Keras models (MLP, CNN1D, LSTM, GRU, CNN-LSTM).
- `models/*_model.pkl` – scikit-learn models (RF, SVM, GB, KNN, Extra Trees).
- `X.npy`, `y.npy` – cached datasets to avoid repeated feature extraction; delete these if you need to reprocess the raw audio (e.g., after changing `AudioProcessor`).

## Troubleshooting & Tips

- **Missing DLLs / GPU errors:** TensorFlow on Windows may require Microsoft Visual C++ redistributables. Install them or run under WSL.
- **Librosa import errors:** Ensure `ffmpeg`/`pydub` dependencies are installed when handling non-WAV formats. On Windows, install `ffmpeg` and add it to `PATH` if conversions fail.
- **Out-of-date caches:** Delete `X.npy`, `y.npy`, and `models/scaler.pkl` if you modify `AudioProcessor` so the new features are recomputed.
- **Class imbalance:** RAVDESS is relatively balanced, but if you add new datasets consider weighting losses (Keras) or adjusting class weights (sklearn).
- **Extending the model zoo:** Add new factory functions to `model_definitions.py`, register them inside `models_to_train` in `train_all_models.py`, and create a new doc under `docs/models/`.

Happy experimenting! 🎵
