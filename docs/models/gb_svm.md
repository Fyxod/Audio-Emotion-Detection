# GB + SVM Ensemble

**Artifact:** `models/gb_svm_model.pkl`

## Purpose

Pairs Gradient Boosting's strong handling of complex feature interactions with SVM's maximal-margin decision surface. Soft voting balances the two to improve generalization on ambiguous utterances.

## Input Features

- Uses the same 110 standardized descriptors from `AudioProcessor`.
- Flat vectors; no reshaping needed.

## Feature Set & Selection

- **Count:** 110 features per clip.
- **Composition:** MFCC/ΔMFCC mean & std values, chroma energies, spectral contrast, tonnetz, and holistic descriptors (ZCR, RMS, spectral centroid, spectral rolloff).
- **Selection process:** Retains every engineered feature. Complementarity relies on learner diversity rather than feature filtering or ranking.

## Architecture & Configuration

- Base estimators:
  - GradientBoostingClassifier (`n_estimators=100`, `learning_rate=0.1`, `max_depth=5`).
  - SVC (`kernel='rbf'`, `probability=True`).
- Combined via `VotingClassifier` with `voting='soft'` for probability averaging.

## Training Configuration

- Follows the same preprocessing, stratified split, and scaling as other classical models.
- Fit on `X_train_scaled`, evaluated on `X_test_scaled` inside `train_all_models.py`.
- Persisted with `joblib.dump` to `models/gb_svm_model.pkl`.

## Usage

```python
import joblib

model = joblib.load("models/gb_svm_model.pkl")
scaler = joblib.load("models/scaler.pkl")
features_scaled = scaler.transform(features.reshape(1, -1))
probs = model.predict_proba(features_scaled)
```

## Strengths & Trade-offs

- **Pros:** Compact yet expressive ensemble; SVM mitigates boosting's overconfidence, boosting adds non-linear feature interactions beyond SVM kernels alone.
- **Cons:** Requires fitting two relatively heavy estimators; sensitive to SVM hyperparameters; inference slower than single-model approaches.
