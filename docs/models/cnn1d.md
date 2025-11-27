# 1D Convolutional Neural Network (CNN1D)

**Artifact:** `models/cnn1d_model.h5`

## Purpose

Captures local temporal patterns in the engineered feature sequences by sliding learnable kernels over the per-frame descriptors. Useful for modeling subtle prosodic variations that dense layers might miss.

## Input Features

- 110 engineered features reshaped to `(features, 1)` via `X.reshape(samples, features, 1)` before training/prediction.
- Same feature inventory as documented in `AudioProcessor`: MFCC/ΔMFCC statistics, chroma, spectral contrast, tonnetz, ZCR, RMS, centroid, rolloff.
- Standardized with the shared scaler at `models/scaler.pkl`.

## Architecture

- Input: `(input_dim, 1)` tensor.
- Block 1: Conv1D 64 filters (kernel 5, ReLU, same padding) → BatchNorm → MaxPooling1D (pool=2) → Dropout 0.3.
- Block 2: Conv1D 128 filters (kernel 5) → BatchNorm → MaxPooling1D (pool=2) → Dropout 0.3.
- Flatten → Dense 128 ReLU → BatchNorm → Dropout 0.3.
- Output Dense `num_classes` Softmax.
- Optimizer: Adam; Loss: sparse categorical cross-entropy.

## Training Configuration

- Same split, epochs (≤50), batch size (32), and callbacks as all DL models in `train_all_models.py`.
- Checkpointed automatically to `models/cnn1d_model.h5` when validation accuracy improves.

## Usage

```python
from tensorflow.keras.models import load_model
import joblib
import numpy as np

model = load_model("models/cnn1d_model.h5")
scaler = joblib.load("models/scaler.pkl")
features = np.load("sample_features.npy")  # (110,)
features_scaled = scaler.transform(features.reshape(1, -1))
features_seq = features_scaled.reshape(1, features_scaled.shape[1], 1)
pred = model.predict(features_seq)
```

## Strengths & Trade-offs

- **Pros:** Learns hierarchical local filters, resilient to noise, generally higher accuracy than pure MLP on expressive speech.
- **Cons:** Slightly heavier than dense models; still treats engineered features as pseudo-temporal sequences, so gains depend on feature ordering quality.
