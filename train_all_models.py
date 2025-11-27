import os
import numpy as np
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from audio_processor import AudioProcessor
from model_definitions import *

# CONFIGURATION
DATASET_PATH = "data/RAVDESS"
MODELS_DIR = "models"
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
TEST_SIZE = 0.2
RANDOM_SEED = 42
EPOCHS = 50
BATCH_SIZE = 32


def extract_dataset_features(dataset_path: str):
    processor = AudioProcessor()
    X, y = [], []
    count = 0

    print("Scanning dataset...")
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                emotion = processor.get_emotion_from_filename(file)
                if emotion == "unknown":
                    continue

                _, features = processor.process_audio_file(file_path)
                if features is not None:
                    X.append(features)
                    y.append(emotion)
                    count += 1

                if count % 500 == 0:
                    print(f"Processed {count} files...")

    return np.array(X), np.array(y)


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Data Extraction
    if os.path.exists("X.npy") and os.path.exists("y.npy"):
        print("Loading cached features...")
        X = np.load("X.npy")
        y = np.load("y.npy")
    else:
        print("Extracting features from RAVDESS dataset...")
        X, y = extract_dataset_features(DATASET_PATH)
        np.save("X.npy", X)
        np.save("y.npy", y)

    print(f"Features: {X.shape}, Labels: {len(y)}")

    # 2. Preprocessing
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    num_classes = len(np.unique(y_encoded))
    joblib.dump(encoder, ENCODER_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)

    # Reshaped data for DL models (samples, features, 1)
    X_train_reshaped = X_train_scaled.reshape(
        X_train_scaled.shape[0], X_train_scaled.shape[1], 1
    )
    X_test_reshaped = X_test_scaled.reshape(
        X_test_scaled.shape[0], X_test_scaled.shape[1], 1
    )

    # 3. Model Training
    models_to_train = {
        # Deep Learning
        "mlp": (create_mlp_model, X_train_scaled, X_test_scaled, True),
        "cnn1d": (create_cnn1d_model, X_train_reshaped, X_test_reshaped, True),
        "lstm": (create_lstm_model, X_train_reshaped, X_test_reshaped, True),
        "gru": (create_gru_model, X_train_reshaped, X_test_reshaped, True),
        "cnn_lstm": (create_cnn_lstm_model, X_train_reshaped, X_test_reshaped, True),
        # Machine Learning
        "rf": (create_rf_model, X_train_scaled, X_test_scaled, False),
        "svm": (create_svm_model, X_train_scaled, X_test_scaled, False),
        "gb": (create_gb_model, X_train_scaled, X_test_scaled, False),
        "knn": (create_knn_model, X_train_scaled, X_test_scaled, False),
        "et": (create_et_model, X_train_scaled, X_test_scaled, False),
        "rf_gb_et": (create_rf_gb_et_ensemble, X_train_scaled, X_test_scaled, False),
        "rf_svm_knn": (
            create_rf_svm_knn_ensemble,
            X_train_scaled,
            X_test_scaled,
            False,
        ),
        "gb_svm": (create_gb_svm_ensemble, X_train_scaled, X_test_scaled, False),
    }

    results = {}

    for name, (model_func, X_tr, X_te, is_dl) in models_to_train.items():
        print(f"\n{'='*20} Training {name.upper()} {'='*20}")

        if is_dl:
            # Deep Learning Training
            model = model_func(X_tr.shape[1], num_classes)
            save_path = os.path.join(MODELS_DIR, f"{name}_model.h5")

            callbacks = [
                EarlyStopping(
                    monitor="val_accuracy", patience=5, restore_best_weights=True
                ),
                ModelCheckpoint(
                    save_path, monitor="val_accuracy", save_best_only=True, verbose=0
                ),
            ]

            history = model.fit(
                X_tr,
                y_train,
                validation_data=(X_te, y_test),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=callbacks,
                verbose=1,
            )

            loss, acc = model.evaluate(X_te, y_test, verbose=0)
            model.save(save_path)  # Ensure final save

        else:
            # Sklearn Training
            model = model_func()
            model.fit(X_tr, y_train)
            acc = model.score(X_te, y_test)
            save_path = os.path.join(MODELS_DIR, f"{name}_model.pkl")
            joblib.dump(model, save_path)

        print(f"{name.upper()} Test Accuracy: {acc:.4f}")
        results[name] = acc

    print("\n" + "=" * 40)
    print("FINAL RESULTS")
    print("=" * 40)
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name.upper()}: {acc:.4f}")


if __name__ == "__main__":
    main()
