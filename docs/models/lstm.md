# Long Short-Term Memory (LSTM)

**Artifact:** `models/lstm_model.h5`

## Purpose

Sequence model that captures long-range dependencies across the ordered engineered features, enabling the network to encode gradual emotional transitions.

## Input Features

- Same 110-dimensional vector reshaped to `(features, 1)` per sample.
- Features produced by `AudioProcessor.extract_features` and standardized via `models/scaler.pkl`.

## Feature Set & Selection

- **Count:** 110 ordered timesteps presented to the recurrent stack.
- **Composition:** MFCC and ΔMFCC statistics, chroma energies, spectral contrast, tonnetz, plus holistic descriptors (ZCR, RMS, centroid, rolloff).
- **Selection process:** No filter-based or model-based selection; all engineered statistics from `AudioProcessor` are preserved to let the LSTM learn its own relevance weighting.

## Architecture

- Input: `(input_dim, 1)` sequence.
- LSTM 128 units with `return_sequences=True` → Dropout 0.3.
- LSTM 64 units (final state) → Dropout 0.3.
- Dense 64 ReLU.
- Output Dense `num_classes` Softmax.
- Optimizer: Adam; Loss: sparse categorical cross-entropy.

## Training Configuration

- Same dataset split (80/20 stratified) and preprocessing as other DL models.
- Up to 50 epochs, batch size 32, early stopping on `val_accuracy` with patience 5.
- Best checkpoint saved to `models/lstm_model.h5`.

## Usage

```python
from tensorflow.keras.models import load_model
import joblib
import numpy as np

model = load_model("models/lstm_model.h5")
scaler = joblib.load("models/scaler.pkl")
features_scaled = scaler.transform(features.reshape(1, -1))
features_seq = features_scaled.reshape(1, features_scaled.shape[1], 1)
pred = model.predict(features_seq)
```

## Strengths & Trade-offs

- **Pros:** Learns contextual relationships beyond fixed receptive fields; often excels when emotional cues manifest over longer time spans.
- **Cons:** Slower to train; susceptible to overfitting if data volume is limited; lacks convolutional inductive bias for local features.
