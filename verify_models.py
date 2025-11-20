import os
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from model_definitions import *

MODELS_DIR = "models"
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

MODELS = {
    "mlp": {"file": "mlp_model.h5", "type": "dl"},
    "cnn1d": {"file": "cnn1d_model.h5", "type": "dl"},
    "lstm": {"file": "lstm_model.h5", "type": "dl"},
    "gru": {"file": "gru_model.h5", "type": "dl"},
    "cnn_lstm": {"file": "cnn_lstm_model.h5", "type": "dl"},
    "rf": {"file": "rf_model.pkl", "type": "ml"},
    "svm": {"file": "svm_model.pkl", "type": "ml"},
    "gb": {"file": "gb_model.pkl", "type": "ml"},
    "knn": {"file": "knn_model.pkl", "type": "ml"},
    "et": {"file": "et_model.pkl", "type": "ml"},
}

def verify_models():
    print("Verifying models...")
    
    if not os.path.exists(SCALER_PATH):
        print("❌ Scaler not found!")
        return

    scaler = joblib.load(SCALER_PATH)
    print("✅ Scaler loaded.")

    # Create dummy features (assuming 110 features as per AudioProcessor)
    dummy_features = np.random.rand(1, 110)
    dummy_features_scaled = scaler.transform(dummy_features)

    all_passed = True

    for name, info in MODELS.items():
        path = os.path.join(MODELS_DIR, info["file"])
        if not os.path.exists(path):
            print(f"❌ {name}: File not found at {path}")
            all_passed = False
            continue

        try:
            if info["type"] == "dl":
                model = load_model(path)
                
                # Handle reshaping for DL models
                try:
                    input_shape = model.input_shape
                    if len(input_shape) == 3:
                        input_data = dummy_features_scaled.reshape(1, 110, 1)
                    else:
                        input_data = dummy_features_scaled
                except AttributeError:
                    input_data = dummy_features_scaled
                
                pred = model.predict(input_data, verbose=0)
            else:
                model = joblib.load(path)
                pred = model.predict_proba(dummy_features_scaled)
            
            print(f"✅ {name}: Loaded and predicted successfully. Output shape: {pred.shape}")
        except Exception as e:
            print(f"❌ {name}: Error - {e}")
            all_passed = False

    if all_passed:
        print("\n🎉 All models verified successfully!")
    else:
        print("\n⚠️ Some models failed verification.")

if __name__ == "__main__":
    verify_models()
