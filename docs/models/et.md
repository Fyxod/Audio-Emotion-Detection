# Extra Trees Classifier

**Artifact:** `models/et_model.pkl`

## Purpose

Extremely randomized trees ensemble that injects more randomness than Random Forest by selecting split thresholds at random. Provides strong performance with reduced variance and faster training.

## Input Features

- 110 standardized engineered features from `AudioProcessor`.
- No additional reshaping required.

## Feature Set & Selection

- **Count:** 110 handcrafted descriptors fed to every tree.
- **Composition:** MFCC/ΔMFCC means & stds, chroma and spectral contrast stats, tonnetz descriptors, plus standalone ZCR, RMS, centroid, and rolloff metrics.
- **Selection process:** Extra Trees trains on the full engineered feature vector; randomness is introduced via split selection, not by removing features ahead of time.

## Configuration

- `ExtraTreesClassifier` with 200 estimators.
- Default maximum depth (splits until pure or minimal samples), `random_state=42`.
- Uses the entire dataset per tree (no bootstrapping) but randomizes thresholds heavily.

## Training Process

- Trained on `X_train_scaled`, validated on `X_test_scaled`.
- Saved to `models/et_model.pkl`.

## Usage

```python
import joblib

model = joblib.load("models/et_model.pkl")
scaler = joblib.load("models/scaler.pkl")
features_scaled = scaler.transform(features.reshape(1, -1))
probs = model.predict_proba(features_scaled)
```

## Strengths & Trade-offs

- **Pros:** Fast to train, typically slightly less overfit than Random Forest, intrinsically parallelizable.
- **Cons:** Interpretability of single trees is low; may require more estimators to stabilize predictions.
