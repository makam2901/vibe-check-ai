# data_utils.py
"""
Utilities for data filtering, feature extraction, preprocessing, and splitting.
"""
import os
import shutil
import librosa
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split # For stratified splitting
from tqdm import tqdm
import joblib

import config

def filter_ravdess_files():
    """
    Filters RAVDESS dataset by copying relevant audio files to FILTERED_DATA_DIR
    based on emotions in config.EMOTION_MAP.
    Skips if FILTERED_DATA_DIR already contains files.
    """
    if any(Path(config.FILTERED_DATA_DIR).rglob('*.wav')):
        print(f"Filtered data already exists in {config.FILTERED_DATA_DIR}. Skipping filtering.")
        return

    print(f"Filtering RAVDESS data from: {config.RAW_RAVDESS_DIR} to {config.FILTERED_DATA_DIR}")
    os.makedirs(config.FILTERED_DATA_DIR, exist_ok=True)
    
    copied_files_count = 0
    if not os.path.exists(config.RAW_RAVDESS_DIR):
        print(f"ERROR: Raw RAVDESS directory not found at {config.RAW_RAVDESS_DIR}")
        return

    for actor_dir_path in Path(config.RAW_RAVDESS_DIR).glob("Actor_*"):
        if not actor_dir_path.is_dir():
            continue
        for audio_file_path in actor_dir_path.glob("*.wav"):
            try:
                parts = audio_file_path.name.split("-")
                if len(parts) > 2:
                    emotion_id = parts[2]
                    if emotion_id in config.EMOTION_MAP:
                        label = config.EMOTION_MAP[emotion_id]
                        dest_dir = os.path.join(config.FILTERED_DATA_DIR, label)
                        os.makedirs(dest_dir, exist_ok=True)
                        shutil.copy(audio_file_path, os.path.join(dest_dir, audio_file_path.name))
                        copied_files_count +=1
            except Exception as e:
                print(f"Error processing file {audio_file_path.name}: {e}")
    
    if copied_files_count > 0:
        print(f"Finished filtering. Copied {copied_files_count} files to {config.FILTERED_DATA_DIR}.")
    else:
        print(f"Warning: No files were copied. Check RAW_RAVDESS_DIR ({config.RAW_RAVDESS_DIR}) and EMOTION_MAP.")

def extract_single_file_features(file_path):
    """
    Extracts features from a single audio file.
    Output shape: (TIME_STEPS, FEATURE_DIM)
    """
    try:
        y, sr = librosa.load(file_path, sr=config.SAMPLE_RATE, mono=True)
        y = librosa.util.fix_length(data=y, size=config.SAMPLES_PER_TRACK)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.N_MFCC)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=config.N_MELS)
        hop_length = 512
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
        target_frames = mfcc.shape[1]
        if chroma.shape[1] != target_frames: chroma = librosa.util.fix_length(chroma, size=target_frames, axis=1)
        if zcr.shape[1] != target_frames: zcr = librosa.util.fix_length(zcr, size=target_frames, axis=1)
        features = np.vstack([mfcc, mel, chroma, zcr])
        return features.T 
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

def process_and_save_all_data():
    """
    Processes all audio files: extracts features, encodes labels, scales features globally.
    Saves X_data, y_data, label_encoder, and scaler. Skips if files exist.
    """
    if os.path.exists(config.PROCESSED_DATA_FILE_X) and \
       os.path.exists(config.PROCESSED_DATA_FILE_Y) and \
       os.path.exists(config.LABEL_ENCODER_FILE) and \
       os.path.exists(config.SCALER_FILE):
        print("Processed data files, label encoder, and scaler already exist. Skipping processing.")
        return

    print(f"Processing all audio files from: {config.FILTERED_DATA_DIR}")
    all_features_list, all_labels_list = [], []
    emotion_folders = [ef for ef in Path(config.FILTERED_DATA_DIR).iterdir() if ef.is_dir()]
    if not emotion_folders:
        print(f"No emotion subdirectories in {config.FILTERED_DATA_DIR}. Run filter_ravdess_files().")
        return

    for label_dir in tqdm(emotion_folders, desc="Extracting features"):
        emotion_label = label_dir.name
        if emotion_label not in config.EMOTION_LABELS: continue
        for audio_file in label_dir.glob("*.wav"):
            features = extract_single_file_features(str(audio_file))
            if features is not None and features.shape == (config.TIME_STEPS, config.FEATURE_DIM):
                all_features_list.append(features)
                all_labels_list.append(emotion_label)
            elif features is not None:
                print(f"Warning: Feature shape mismatch for {audio_file.name}. Expected {(config.TIME_STEPS, config.FEATURE_DIM)}, got {features.shape}. Skipping.")

    if not all_features_list:
        print("No features extracted. Check audio files/parameters.")
        return

    X_data = np.array(all_features_list)
    y_strings = np.array(all_labels_list)

    label_encoder = LabelEncoder()
    label_encoder.fit(config.EMOTION_LABELS)
    y_encoded = label_encoder.transform(y_strings)

    num_samples, time_steps, feature_dim = X_data.shape
    X_data_reshaped = X_data.reshape(-1, feature_dim)
    scaler = StandardScaler()
    X_scaled_reshaped = scaler.fit_transform(X_data_reshaped)
    X_scaled = X_scaled_reshaped.reshape(num_samples, time_steps, feature_dim)

    np.save(config.PROCESSED_DATA_FILE_X, X_scaled)
    np.save(config.PROCESSED_DATA_FILE_Y, y_encoded)
    joblib.dump(label_encoder, config.LABEL_ENCODER_FILE)
    joblib.dump(scaler, config.SCALER_FILE)
    print(f"Processed data saved: X {X_scaled.shape}, y {y_encoded.shape}. Encoder/Scaler saved.")

def load_full_processed_data():
    """Loads full preprocessed X_data, y_data, label_encoder, and scaler."""
    if not all(os.path.exists(f) for f in [config.PROCESSED_DATA_FILE_X, config.PROCESSED_DATA_FILE_Y,
                                           config.LABEL_ENCODER_FILE, config.SCALER_FILE]):
        print("Processed data files not found. Please run process_and_save_all_data() first.")
        return None, None, None, None
    
    X_scaled = np.load(config.PROCESSED_DATA_FILE_X)
    y_encoded = np.load(config.PROCESSED_DATA_FILE_Y)
    label_encoder = joblib.load(config.LABEL_ENCODER_FILE)
    scaler = joblib.load(config.SCALER_FILE)
    print("Full processed data and transformers loaded.")
    return X_scaled, y_encoded, label_encoder, scaler

def get_stratified_data_splits():
    """
    Loads the full processed data and performs a stratified train-validation-test split.
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, scaler
    """
    X_full, y_full, label_encoder, scaler = load_full_processed_data()
    if X_full is None:
        return None, None, None, None, None, None, None, None

    # First split: separate out the test set
    # (1 - TEST_SPLIT_SIZE) for train_val, TEST_SPLIT_SIZE for test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_full, y_full,
        test_size=config.TEST_SPLIT_SIZE,
        stratify=y_full, # Stratify by full labels
        random_state=config.RANDOM_SEED
    )

    # Second split: split train_val into train and validation
    # VALIDATION_SPLIT_SIZE is relative to the original full dataset.
    # So, the size for validation set from train_val_set should be:
    # val_proportion_of_train_val = TEST_SPLIT_SIZE / (1 - VALIDATION_SPLIT_SIZE) -> This is incorrect.
    # It should be: val_size / (train_val_size)
    # If VALIDATION_SPLIT_SIZE is, e.g., 0.2 of original, and TEST_SPLIT_SIZE is 0.2 of original,
    # then train_val is 0.8 of original. Validation is 0.2 / 0.8 = 0.25 of train_val.
    
    # Adjust validation split size to be a proportion of the X_train_val set
    # Example: if original TEST_SPLIT_SIZE=0.2, VALIDATION_SPLIT_SIZE=0.2
    # Then X_train_val is 80% of data. We want validation to be 20% of original.
    # So, validation should be 0.2 / 0.8 = 0.25 of X_train_val.
    relative_val_size = config.VALIDATION_SPLIT_SIZE / (1.0 - config.TEST_SPLIT_SIZE)
    if relative_val_size >= 1.0 or relative_val_size <= 0:
        print(f"Warning: Calculated relative_val_size ({relative_val_size:.2f}) is invalid. Check TEST/VALIDATION_SPLIT_SIZE in config.")
        print("Falling back to using a small validation set or no validation set if sizes are problematic.")
        # Handle edge case: if test split is too large, train_val might be too small for further splitting.
        if len(X_train_val) < 2 or relative_val_size >=1.0 : # Cannot split further meaningfully
            X_train, X_val, y_train, y_val = X_train_val, np.array([]), y_train_val, np.array([])
            print("Using all of train_val for training, validation set is empty.")
        else: # Proceed with split if possible
             X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=relative_val_size,
                stratify=y_train_val, # Stratify by train_val labels
                random_state=config.RANDOM_SEED
            )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=relative_val_size,
            stratify=y_train_val, # Stratify by train_val labels
            random_state=config.RANDOM_SEED
        )


    print(f"Data split (stratified):")
    print(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    if X_val.size > 0:
        print(f"  X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    else:
        print(f"  X_val is empty.")
    print(f"  X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, scaler

if __name__ == '__main__':
    # Step 1: Filter raw audio files (if not already done)
    filter_ravdess_files()
    
    # Step 2: Extract features, preprocess, and save all data (if not already done)
    process_and_save_all_data()
    
    # Step 3: Get stratified splits
    print("\nAttempting to get stratified data splits...")
    X_tr, X_v, X_te, y_tr, y_v, y_te, le, sc = get_stratified_data_splits()
    
    if X_tr is not None:
        print("\n--- Data Splitting Summary ---")
        print(f"Train set size: {len(y_tr)}")
        if y_v.size > 0 : print(f"Validation set size: {len(y_v)}")
        else: print("Validation set is empty.")
        print(f"Test set size: {len(y_te)}")
        print(f"Label Encoder Classes: {le.classes_}")
        
        # Verify class distribution in splits (optional detailed check)
        print("\nClass distribution in y_train:", np.unique(y_tr, return_counts=True))
        if y_v.size > 0: print("Class distribution in y_val:", np.unique(y_v, return_counts=True))
        print("Class distribution in y_test:", np.unique(y_te, return_counts=True))
