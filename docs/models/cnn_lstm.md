# Hybrid CNN-LSTM

**Artifact:** `models/cnn_lstm_model.h5`

## Purpose

Combines convolutional feature extraction with recurrent temporal modeling. The initial Conv1D layer captures short-term spectral patterns, and the downstream LSTM models longer-term emotional dynamics.

## Input Features

- 110 standardized engineered features reshaped to `(features, 1)`.
- Feature sources: MFCC & deltas, chroma, spectral contrast, tonnetz, ZCR, RMS, spectral centroid, spectral rolloff.

## Architecture

- Conv1D 64 filters (kernel 3, ReLU, same padding) → MaxPooling1D (pool=2) → Dropout 0.3.
- LSTM 64 units → Dropout 0.3.
- Dense 64 ReLU.
- Output Dense `num_classes` Softmax.
- Optimizer: Adam; Loss: sparse categorical cross-entropy.

## Training Configuration

- Uses the shared training loop in `train_all_models.py` with 50-epoch budget, batch size 32, stratified 80/20 split, and EarlyStopping/ModelCheckpoint callbacks.
- Best weights stored at `models/cnn_lstm_model.h5`.

## Usage

```python
from tensorflow.keras.models import load_model
import joblib
import numpy as np

model = load_model("models/cnn_lstm_model.h5")
scaler = joblib.load("models/scaler.pkl")
features_scaled = scaler.transform(features.reshape(1, -1))
features_seq = features_scaled.reshape(1, features_scaled.shape[1], 1)
pred = model.predict(features_seq)
```

## Strengths & Trade-offs

- **Pros:** Often the most accurate among deep models thanks to convolutional inductive bias plus temporal reasoning.
- **Cons:** Heavier compute footprint, longer training time, more sensitive to overfitting without sufficient regularization.
