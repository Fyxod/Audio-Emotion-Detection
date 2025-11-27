# Gated Recurrent Unit (GRU)

**Artifact:** `models/gru_model.h5`

## Purpose

Provides a lighter recurrent alternative to the LSTM while retaining gating mechanisms, making it attractive when latency or memory are constrained.

## Input Features

- 110 standardized features reshaped to `(features, 1)`.
- Derived from MFCC/chroma/contrast/tonnetz/ZCR/RMS/spectral stats via `AudioProcessor`.

## Architecture

- GRU 128 units with `return_sequences=True` → Dropout 0.3.
- GRU 64 units → Dropout 0.3.
- Dense 64 ReLU.
- Output Dense `num_classes` Softmax.
- Optimizer: Adam; Loss: sparse categorical cross-entropy.

## Training Configuration

- Same preprocessing, split, and callback strategy as other deep learning models (`train_all_models.py`).
- EarlyStopping + ModelCheckpoint persist the best weights to `models/gru_model.h5`.

## Usage

```python
from tensorflow.keras.models import load_model
import joblib
import numpy as np

model = load_model("models/gru_model.h5")
scaler = joblib.load("models/scaler.pkl")
features_scaled = scaler.transform(features.reshape(1, -1))
features_seq = features_scaled.reshape(1, features_scaled.shape[1], 1)
pred = model.predict(features_seq)
```

## Strengths & Trade-offs

- **Pros:** Comparable accuracy to LSTM with fewer parameters and faster convergence.
- **Cons:** Without convolutional front-ends, still depends on engineered feature ordering; may underperform on extremely long sequences compared to LSTM.
