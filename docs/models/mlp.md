# Multi-Layer Perceptron (MLP)

**Artifact:** `models/mlp_model.h5`

## Purpose

A dense feed-forward network that treats every engineered acoustic descriptor as an independent signal. It is lightweight, easy to deploy, and acts as a strong baseline against the more complex temporal models.

## Input Features

- Consumes the 110-dimensional feature vector produced by `AudioProcessor.extract_features` (MFCCs, delta MFCCs, chroma, spectral contrast, tonnetz, zero-crossing rate, RMS, spectral centroid, spectral rolloff).
- Features are standardized with `StandardScaler` fit on the training split and persisted at `models/scaler.pkl`.

## Architecture

- Input layer: `input_dim` neurons (one per engineered feature).
- Hidden stack:
  - Dense 512 → ReLU → BatchNorm → Dropout 0.4
  - Dense 256 → ReLU → BatchNorm → Dropout 0.3
  - Dense 128 → ReLU → BatchNorm → Dropout 0.3
- Output: Dense `num_classes` → Softmax.
- Optimizer: Adam (default learning rate).
- Loss: `sparse_categorical_crossentropy` on integer-encoded emotions.

## Training Configuration

- Dataset: full RAVDESS speech corpus under `data/RAVDESS`.
- Split: `train_test_split` with `test_size=0.2`, `random_state=42`, stratified by label.
- Epochs: up to 50 with `batch_size=32`.
- Callbacks: `EarlyStopping` (monitor `val_accuracy`, patience 5, `restore_best_weights=True`) and `ModelCheckpoint` saving the best weights to `models/mlp_model.h5`.

## Usage

```python
from tensorflow.keras.models import load_model
import joblib
import numpy as np

model = load_model("models/mlp_model.h5")
scaler = joblib.load("models/scaler.pkl")
features = np.load("sample_features.npy")  # shape (110,)
features_scaled = scaler.transform(features.reshape(1, -1))
pred = model.predict(features_scaled)
```

## Strengths & Trade-offs

- **Pros:** Fast inference, robust to small datasets, leveraged BatchNorm/Dropout for regularization.
- **Cons:** Ignores temporal locality present in sequential features; performance can saturate compared to convolutional/recurrent models on longer utterances.
