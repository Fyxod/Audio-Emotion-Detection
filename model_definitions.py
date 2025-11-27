import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    Conv1D,
    MaxPooling1D,
    Flatten,
    LSTM,
    GRU,
    Input,
)
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Deep Learning Models (Keras)


def create_mlp_model(input_dim, num_classes):
    """Multi-Layer Perceptron"""
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(512, activation="relu"),
            BatchNormalization(),
            Dropout(0.4),
            Dense(256, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def create_cnn1d_model(input_dim, num_classes):
    """1D Convolutional Neural Network"""
    model = Sequential(
        [
            Input(shape=(input_dim, 1)),
            Conv1D(64, kernel_size=5, activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            Conv1D(128, kernel_size=5, activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            Flatten(),
            Dense(128, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def create_lstm_model(input_dim, num_classes):
    """Long Short-Term Memory Network"""
    model = Sequential(
        [
            Input(shape=(input_dim, 1)),
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            LSTM(64),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def create_gru_model(input_dim, num_classes):
    """Gated Recurrent Unit Network"""
    model = Sequential(
        [
            Input(shape=(input_dim, 1)),
            GRU(128, return_sequences=True),
            Dropout(0.3),
            GRU(64),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def create_cnn_lstm_model(input_dim, num_classes):
    """Hybrid CNN-LSTM Network"""
    model = Sequential(
        [
            Input(shape=(input_dim, 1)),
            Conv1D(64, kernel_size=3, activation="relu", padding="same"),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            LSTM(64),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


# Machine Learning Models (Sklearn)


def create_rf_model():
    """Random Forest Classifier"""
    return RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)


def create_svm_model():
    """Support Vector Machine"""
    return SVC(kernel="rbf", probability=True, random_state=42)


def create_gb_model():
    """Gradient Boosting Classifier"""
    return GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
    )


def create_knn_model():
    """K-Nearest Neighbors"""
    return KNeighborsClassifier(n_neighbors=5)


def create_et_model():
    """Extra Trees Classifier"""
    return ExtraTreesClassifier(n_estimators=200, random_state=42)


# Ensemble Models (Sklearn)


def create_rf_gb_et_ensemble():
    """Soft-voting ensemble of RF, GB, and Extra Trees."""
    estimators = [
        ("rf", create_rf_model()),
        ("gb", create_gb_model()),
        ("et", create_et_model()),
    ]
    return VotingClassifier(estimators=estimators, voting="soft")


def create_rf_svm_knn_ensemble():
    """Soft-voting ensemble of RF, SVM, and KNN."""
    estimators = [
        ("rf", create_rf_model()),
        ("svm", create_svm_model()),
        ("knn", create_knn_model()),
    ]
    return VotingClassifier(estimators=estimators, voting="soft")


def create_gb_svm_ensemble():
    """Soft-voting ensemble of GB and SVM."""
    estimators = [
        ("gb", create_gb_model()),
        ("svm", create_svm_model()),
    ]
    return VotingClassifier(estimators=estimators, voting="soft")
