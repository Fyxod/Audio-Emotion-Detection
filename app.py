import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import tempfile
from audio_processor import AudioProcessor
import soundfile as sf
import pickle
from tensorflow.keras.models import load_model
from pydub import AudioSegment
import joblib

# Configuration
MODELS_DIR = "models"
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

# Model Definitions
MODELS = {
    "Multi-Layer Perceptron (MLP)": {"file": "mlp_model.h5", "type": "dl"},
    "1D CNN": {"file": "cnn1d_model.h5", "type": "dl"},
    "LSTM": {"file": "lstm_model.h5", "type": "dl"},
    "GRU": {"file": "gru_model.h5", "type": "dl"},
    "CNN-LSTM": {"file": "cnn_lstm_model.h5", "type": "dl"},
    "Random Forest": {"file": "rf_model.pkl", "type": "ml"},
    "Support Vector Machine (SVM)": {"file": "svm_model.pkl", "type": "ml"},
    "Gradient Boosting": {"file": "gb_model.pkl", "type": "ml"},
    "K-Nearest Neighbors (KNN)": {"file": "knn_model.pkl", "type": "ml"},
    "Extra Trees": {"file": "et_model.pkl", "type": "ml"},
    "RF + GB + ET (Ensemble)": {"file": "rf_gb_et_model.pkl", "type": "ml"},
    "RF + SVM + KNN (Ensemble)": {"file": "rf_svm_knn_model.pkl", "type": "ml"},
    "GB + SVM (Ensemble)": {"file": "gb_svm_model.pkl", "type": "ml"},
}

# Define the emotion labels (must match training)
EMOTIONS = [
    "angry",
    "calm",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
]

# Initialize audio processor
audio_processor = AudioProcessor()


def load_selected_model(model_name):
    """Load the selected model and scaler."""
    model_info = MODELS[model_name]
    model_path = os.path.join(MODELS_DIR, model_info["file"])

    if not os.path.exists(model_path):
        return None, None, None

    try:
        if model_info["type"] == "dl":
            model = load_model(model_path)
        else:
            model = joblib.load(model_path)

        scaler = joblib.load(SCALER_PATH)
        return model, scaler, model_info["type"]
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


def predict_emotion(model, scaler, model_type, features):
    if model is None or scaler is None:
        raise ValueError("Model or scaler not loaded.")

    # Scale features
    features_scaled = scaler.transform(features.reshape(1, -1))

    # Reshape if needed
    if model_type == "dl":
        try:
            input_shape = model.input_shape
            if len(input_shape) == 3:
                features_scaled = features_scaled.reshape(
                    1, features_scaled.shape[1], 1
                )
        except AttributeError:
            pass  # Should not happen for Keras models

    if model_type == "dl":
        prediction = model.predict(features_scaled)[0]
    else:
        prediction = model.predict_proba(features_scaled)[0]

    emotion_idx = prediction.argmax()
    confidence = prediction[emotion_idx]
    return EMOTIONS[emotion_idx], confidence, prediction


def plot_waveform(audio, sr):
    plt.figure(figsize=(8, 3))
    librosa.display.waveshow(audio, sr=sr)
    plt.title("Audio Waveform")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    return plt


def plot_spectrogram(audio, sr):
    plt.figure(figsize=(8, 3))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    return plt


def convert_to_wav(input_file, output_path):
    """Convert any audio file to WAV format."""
    try:
        audio = AudioSegment.from_file(input_file)
        audio.export(output_path, format="wav")
        return True
    except Exception as e:
        st.error(f"Error converting audio file: {e}")
        return False


def main():
    st.set_page_config(
        page_title="Audio Emotion Recognition", page_icon="🎵", layout="wide"
    )

    st.title("🎵 Audio Emotion Recognition")
    st.write("Analyze emotions from speech using AI")

    # Sidebar for model selection
    st.sidebar.header("Model Configuration")
    selected_model_name = st.sidebar.selectbox("Select Model", list(MODELS.keys()))

    # Load model
    model, scaler, model_type = load_selected_model(selected_model_name)

    if model is None:
        st.sidebar.warning(
            f"Model '{selected_model_name}' not found. Please train the models first."
        )
    else:
        st.sidebar.success(f"Loaded: {selected_model_name}")

    st.header("Upload Audio")
    uploaded_file = st.file_uploader(
        "Choose an audio file", type=["wav", "mp3", "ogg", "flac", "m4a", "aac"]
    )

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
        ) as temp_input:
            temp_input.write(uploaded_file.getvalue())
            temp_input_path = temp_input.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            temp_wav_path = temp_wav.name

        if uploaded_file.name.lower().endswith(".wav"):
            temp_wav_path = temp_input_path
        else:
            if not convert_to_wav(temp_input_path, temp_wav_path):
                st.error("Failed to convert audio file to WAV format.")
                os.unlink(temp_input_path)
                return

        st.audio(uploaded_file, format="audio/wav")

        if model is not None:
            display_results(temp_wav_path, model, scaler, model_type)
        else:
            st.error("Please select a valid model to proceed.")

        os.unlink(temp_input_path)
        if temp_wav_path != temp_input_path:
            os.unlink(temp_wav_path)
    else:
        st.info("Please upload an audio file to analyze.")


def display_results(audio_path, model, scaler, model_type):
    with st.spinner("Processing audio and analyzing emotions..."):
        audio, features = audio_processor.process_audio_file(audio_path)

        if audio is not None and features is not None:
            try:
                with st.spinner("Making predictions..."):
                    emotion, confidence, probs = predict_emotion(
                        model, scaler, model_type, features
                    )

                st.markdown("---")
                st.markdown(f"## Detected Emotion: {emotion.upper()}")
                st.markdown(f"### Confidence: {confidence:.2%}")
                st.markdown("---")

                col1, col2 = st.columns(2)
                with col1:
                    st.pyplot(plot_waveform(audio, audio_processor.sample_rate))
                with col2:
                    st.pyplot(plot_spectrogram(audio, audio_processor.sample_rate))

                st.subheader("Emotion Probability Distribution")
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.bar(EMOTIONS, probs)
                ax.set_title("Emotion Probability Distribution")
                ax.set_ylim(0, 1)
                plt.xticks(rotation=45)
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            st.error("Error processing audio file. Please try again.")


if __name__ == "__main__":
    main()
