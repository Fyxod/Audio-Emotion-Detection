# Gradient Boosting Classifier

**Artifact:** `models/gb_model.pkl`

## Purpose

Sequential ensemble of shallow decision trees optimized via gradient boosting. Captures complex feature interactions and provides calibrated probability estimates without deep learning infrastructure.

## Input Features

- 110 standardized descriptors (MFCC statistics, chroma, spectral features, etc.).
- Consumes flattened vectors; no reshaping.

## Feature Set & Selection

- **Count:** 110 handcrafted attributes per utterance.
- **Composition:** Means/stds of MFCCs and ΔMFCCs, chroma, spectral contrast, tonnetz, supported by scalar ZCR, RMS, centroid, and rolloff features.
- **Selection process:** Gradient Boosting ingests the full engineered set; no pre-selection is applied because boosting inherently learns feature importance through split gains.

## Configuration

- `GradientBoostingClassifier` with:
  - `n_estimators=100`
  - `learning_rate=0.1`
  - `max_depth=5`
  - `random_state=42`

## Training Process

- Shares the same train/test split as other models.
- Learned on `X_train_scaled`; validation accuracy computed via `.score(X_test_scaled, y_test)`.
- Persisted to `models/gb_model.pkl` through `joblib.dump`.

## Usage

```python
import joblib

model = joblib.load("models/gb_model.pkl")
scaler = joblib.load("models/scaler.pkl")
features_scaled = scaler.transform(features.reshape(1, -1))
probs = model.predict_proba(features_scaled)
```

## Strengths & Trade-offs

- **Pros:** High accuracy with small datasets, interpretable feature importances, solid probability calibration.
- **Cons:** Training is sequential (no parallelization across trees), can overfit without tuning learning rate/depth, requires careful hyper-parameter tuning for peak results.
