# Support Vector Machine (RBF)

**Artifact:** `models/svm_model.pkl`

## Purpose

Margin-based classifier that projects engineered acoustic features into an RBF kernel space to separate emotions via maximal margins. Serves as a strong classical baseline when feature dimensionality is moderate.

## Input Features

- 110 standardized descriptors from `AudioProcessor`.
- No additional reshaping; relies on dense feature vectors.

## Feature Set & Selection

- **Count:** 110 features per sample.
- **Composition:** MFCC/ΔMFCC statistics, chroma, spectral contrast, tonnetz, and global measures (ZCR, RMS, centroid, rolloff).
- **Selection process:** The SVM uses the entire engineered feature space; there is no feature elimination or ranking prior to fitting because the RBF kernel handles high-dimensional representations well.

## Configuration

- `sklearn.svm.SVC` with `kernel='rbf'` and probability estimates enabled (`probability=True`).
- Default `C` and `gamma` (can be tuned via grid search when needed).
- `random_state=42` for reproducibility of probability estimation (affects Platt scaling seed).

## Training Process

- Fitted on `X_train_scaled`, validated on `X_test_scaled` within `train_all_models.py`.
- Saved via `joblib.dump` to `models/svm_model.pkl`.

## Usage

```python
import joblib

model = joblib.load("models/svm_model.pkl")
scaler = joblib.load("models/scaler.pkl")
features_scaled = scaler.transform(features.reshape(1, -1))
probs = model.predict_proba(features_scaled)
```

## Strengths & Trade-offs

- **Pros:** Excellent performance on smaller datasets; robust margins mitigate overfitting; no need for GPU.
- **Cons:** Training time grows quadratically with dataset size; probability outputs can be slow due to internal cross-validation; model size tied to support vectors.
