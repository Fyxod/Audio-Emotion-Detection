import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from audio_processor import AudioProcessor

# CONFIGURATION

DATASET_PATH = "data/RAVDESS"
MODEL_PATH = "models/emotion_model.h5"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
TEST_SIZE = 0.2
RANDOM_SEED = 42
EPOCHS = 60
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# FEATURE EXTRACTIO
def extract_dataset_features(dataset_path: str):
    processor = AudioProcessor()
    X, y = [], []
    count = 0

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

                if count % 100 == 0:
                    print(f"Processed {count} files...")

    return np.array(X), np.array(y)

# MODEL CREATION
def create_model(input_dim, num_classes):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.4),

        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(num_classes, activation='softmax')
    ])

    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# MAIN TRAINING PIPELINE
def main():
    print("🎧 Extracting features from RAVDESS dataset...")
    X, y = extract_dataset_features(DATASET_PATH)
    print(f"Features extracted: {X.shape}, Labels: {len(y)}")

    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    num_classes = len(np.unique(y_encoded))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_encoded
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create model
    model = create_model(input_dim=X_train.shape[1], num_classes=num_classes)
    model.summary()

    # Callbacks
    os.makedirs("models", exist_ok=True)
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True)

    # Train
    print("🚀 Training deep model...")
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )

    # Evaluate
    loss, acc = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test Accuracy: {acc:.3f}")

    # Save scaler & encoder
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    print(f"Saved model to '{MODEL_PATH}', scaler to '{SCALER_PATH}', and encoder to '{ENCODER_PATH}'")

if __name__ == "__main__":
    main()
