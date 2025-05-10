# config.py
"""
Configuration file for the Emotion Detection project.
"""
import os
import torch

# --- Directory Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Root directory of the project
DATA_DIR = os.path.join(BASE_DIR, "data")  # Fixed path to use relative path
RAW_RAVDESS_DIR = os.path.join(DATA_DIR, "Audio_Speech_Actors_01-24")
FILTERED_DATA_DIR = os.path.join(DATA_DIR, "filtered_ravdess") # Processed audio files
FEATURES_DIR = os.path.join(DATA_DIR, "features") # Saved features (optional)
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "trained_models")
RESULTS_DIR = os.path.join(BASE_DIR, "results") # For saving metrics, plots

# --- Emotion Mapping ---
EMOTION_MAP = {
    "01": "neutral", "03": "happy", "04": "sad",
    "05": "angry", "08": "excited"
}
EMOTION_LABELS = sorted(list(EMOTION_MAP.values()))
NUM_CLASSES = len(EMOTION_LABELS)

# --- Audio Parameters ---
SAMPLE_RATE = 16000  # Hz
DURATION = 3         # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# --- Feature Extraction Parameters ---
N_MFCC = 40
N_MELS = 40
# The shape of the extracted features (before batching) will be (TIME_STEPS, FEATURE_DIM)
# Based on notebook: X.shape (num_samples, 94, 92) -> (samples, time_steps, features)
TIME_STEPS = 94    # Corresponds to CNN Height if using 2D Conv
FEATURE_DIM = 93   # Corresponds to CNN Width if using 2D Conv, or input_size for RNN/Transformer
CNN_INPUT_CHANNELS = 1 # For spectrogram-like features

# --- Data Handling ---
TEST_SPLIT_SIZE = 0.20
VALIDATION_SPLIT_SIZE = 0.20 # of the original training set (0.25 * (1-0.2) = 0.2 of total)
RANDOM_SEED = 42
BATCH_SIZE = 32 # Default batch size

# --- Training & Model Parameters ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 10
SCHEDULER_STEP_SIZE = 7
SCHEDULER_GAMMA = 0.5

# --- Optuna Hyperparameter Optimization ---
OPTUNA_N_TRIALS = 20       # Number of trials for HPO
OPTUNA_TRIAL_EPOCHS = 15   # Reduced epochs for faster HPO trials
OPTUNA_PATIENCE = 5        # Early stopping for HPO trials

# --- Augmentation ---
# Parameters for audiomentations (if used)
AUGMENT_PROBABILITY = 0.5
TIME_STRETCH_MIN_MAX = (0.9, 1.1)
PITCH_SHIFT_MIN_MAX_SEMITONES = (-2, 2)
SHIFT_MIN_MAX_FRACTION = (-0.2, 0.2)

# --- File Names for Saved Data/Objects ---
PROCESSED_DATA_FILE_X = os.path.join(FEATURES_DIR, "X_data.npy")
PROCESSED_DATA_FILE_Y = os.path.join(FEATURES_DIR, "y_data.npy")
LABEL_ENCODER_FILE = os.path.join(MODEL_SAVE_DIR, "label_encoder.joblib")
SCALER_FILE = os.path.join(MODEL_SAVE_DIR, "scaler.joblib")

# --- Create directories if they don't exist ---
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FILTERED_DATA_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

if __name__ == '__main__':
    print(f"Project Base Directory: {BASE_DIR}")
    print(f"Device: {DEVICE}")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Emotion labels: {EMOTION_LABELS}")
    print(f"Default Batch Size: {BATCH_SIZE}")
    print(f"Data will be filtered to: {FILTERED_DATA_DIR}")
    print(f"Features will be saved in: {FEATURES_DIR}")
    print(f"Models will be saved in: {MODEL_SAVE_DIR}")
