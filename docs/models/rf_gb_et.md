# RF + GB + ET Ensemble

**Artifact:** `models/rf_gb_et_model.pkl`

## Purpose

Combines Random Forest, Gradient Boosting, and Extra Trees through soft voting to blend bagging- and boosting-based decision tree ensembles. The goal is to balance variance reduction (RF/ET) with strong bias correction (GB).

## Input Features

- Shared 110-dimensional standardized vectors produced by `AudioProcessor.extract_features`.
- No reshaping; operates on flat feature arrays scaled via `models/scaler.pkl`.

## Feature Set & Selection

- **Count:** 110 handcrafted descriptors per utterance.
- **Composition:** MFCC and ΔMFCC statistics, chroma energies, spectral contrast bands, tonnetz, plus ZCR, RMS, spectral centroid, and rolloff metrics.
- **Selection process:** The ensemble ingests the complete engineered feature set without pruning. Diversity is achieved via the differing learning biases of RF, GB, and ET rather than upfront feature ranking.

## Architecture & Configuration

- Base estimators mirror the standalone implementations:
  - RandomForestClassifier (`n_estimators=200`, `max_depth=20`).
  - GradientBoostingClassifier (`n_estimators=100`, `learning_rate=0.1`, `max_depth=5`).
  - ExtraTreesClassifier (`n_estimators=200`).
- Wrapped by `VotingClassifier` with `voting='soft'` to average class probabilities.

## Training Configuration

- Same preprocessing and 80/20 stratified split as other ML models in `train_all_models.py`.
- Each base estimator is fit on the scaled training set within the VotingClassifier; evaluation uses `.score` on the test split.
- Serialized via `joblib.dump` to `models/rf_gb_et_model.pkl`.

## Usage

```python
import joblib
import numpy as np

model = joblib.load("models/rf_gb_et_model.pkl")
scaler = joblib.load("models/scaler.pkl")
features_scaled = scaler.transform(features.reshape(1, -1))
probs = model.predict_proba(features_scaled)
```

## Strengths & Trade-offs

- **Pros:** Mixes complementary tree ensembles, delivering robust accuracy with modest extra cost; soft voting smooths probability estimates.
- **Cons:** Larger memory footprint; training time accumulates across three full estimators; interpretability is reduced versus single models.
