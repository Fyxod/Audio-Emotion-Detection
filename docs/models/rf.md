# Random Forest Classifier

**Artifact:** `models/rf_model.pkl`

## Purpose

Classical ensemble baseline that aggregates predictions from multiple decision trees trained on bootstrapped samples. Provides interpretability and fast inference without GPU requirements.

## Input Features

- Directly consumes the 110-dimensional standardized feature vectors output by `AudioProcessor`.
- No reshaping required; scaling handled via `models/scaler.pkl`.

## Configuration

- `RandomForestClassifier` with 200 estimators.
- Maximum tree depth: 20.
- `random_state=42` to ensure reproducibility.
- Uses Gini impurity and square-root feature selection per split (sklearn defaults).

## Training Process

- Features split with the same `train_test_split` (80/20) used for deep models.
- Fitted on `X_train_scaled`, evaluated with `.score` on `X_test_scaled`.
- Serialized via `joblib.dump` to `models/rf_model.pkl`.

## Usage

```python
import joblib
import numpy as np

model = joblib.load("models/rf_model.pkl")
scaler = joblib.load("models/scaler.pkl")
features_scaled = scaler.transform(features.reshape(1, -1))
probs = model.predict_proba(features_scaled)
```

## Strengths & Trade-offs

- **Pros:** Handles non-linear interactions, robust to outliers, requires minimal tuning.
- **Cons:** Memory intensive with many trees; lacks temporal reasoning compared to sequence models; probability calibration may be coarse for rare classes.
