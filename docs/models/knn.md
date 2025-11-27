# K-Nearest Neighbors (KNN)

**Artifact:** `models/knn_model.pkl`

## Purpose

Instance-based learner that classifies an utterance by looking at the dominant emotion among its closest neighbors in the engineered feature space. Acts as a simple sanity check for feature quality.

## Input Features

- Uses the 110-dimensional standardized vectors directly.
- Distance metric implicitly assumes comparable feature scales, hence the reliance on `StandardScaler`.

## Feature Set & Selection

- **Count:** 110 numerical attributes per sample.
- **Composition:** MFCC/ΔMFCC mean and std summaries, chroma energies, spectral contrast, tonnetz, along with ZCR, RMS, centroid, and rolloff statistics.
- **Selection process:** KNN stores and compares the entire engineered feature vector; no feature pruning occurs because distance-based voting benefits from the full handcrafted space.

## Configuration

- `KNeighborsClassifier` with `n_neighbors=5` (Euclidean distance).
- No training phase beyond storing the scaled training set inside the estimator.

## Training Process

- Fitted on `X_train_scaled` and labels `y_train`.
- Accuracy evaluated on `X_test_scaled`.
- Serialized with `joblib.dump` to `models/knn_model.pkl`.

## Usage

```python
import joblib

model = joblib.load("models/knn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
features_scaled = scaler.transform(features.reshape(1, -1))
probs = model.predict_proba(features_scaled)
```

## Strengths & Trade-offs

- **Pros:** Zero training time, interpretable influence of neighbors, good baseline for feature relevance.
- **Cons:** Memory-heavy, slow at inference on large datasets, sensitive to noisy features and class imbalance.
