# RF + SVM + KNN Ensemble

**Artifact:** `models/rf_svm_knn_model.pkl`

## Purpose

Blends a tree ensemble, a margin-based classifier, and an instance-based learner to capture complementary decision boundaries. Soft voting allows smooth probability fusion across modeling paradigms.

## Input Features

- Accepts the 110-length standardized vectors from `AudioProcessor`.
- No reshaping necessary; relies on `StandardScaler` for distance/margin stability.

## Feature Set & Selection

- **Count:** 110 engineered descriptors per sample.
- **Composition:** MFCC and ΔMFCC mean/std summaries, chroma, spectral contrast, tonnetz, along with ZCR, RMS, spectral centroid, and rolloff statistics.
- **Selection process:** Uses the entire engineered feature bank. Diversity stems from heterogenous learners (RF, SVM, KNN) rather than explicit feature subset selection.

## Architecture & Configuration

- Base estimators:
  - RandomForestClassifier (`n_estimators=200`, `max_depth=20`).
  - SVC (`kernel='rbf'`, `probability=True`).
  - KNeighborsClassifier (`n_neighbors=5`).
- Combined with `VotingClassifier(voting='soft')` so predicted probabilities are averaged before argmax.

## Training Configuration

- Shares the standard stratified 80/20 split and scaling pipeline inside `train_all_models.py`.
- The ensemble is fit on `X_train_scaled`; accuracy computed via `.score(X_test_scaled, y_test)`.
- Saved with `joblib.dump` to `models/rf_svm_knn_model.pkl`.

## Usage

```python
import joblib

model = joblib.load("models/rf_svm_knn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
features_scaled = scaler.transform(features.reshape(1, -1))
probs = model.predict_proba(features_scaled)
```

## Strengths & Trade-offs

- **Pros:** Captures global, local, and margin-based patterns simultaneously; resilient to different error modes.
- **Cons:** Training slower due to SVM fitting; inference latency increases because all three estimators must run; probability calibration depends on SVM's Platt scaling.
