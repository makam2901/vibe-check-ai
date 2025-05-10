# evaluate.py
"""
Script for evaluating a trained emotion detection model on the test set.
Handles model-specific hyperparameters for instantiation.
"""
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

import config
import data_utils 
import models     
from dataset import EmotionDataset
from train_utils import load_model

def plot_confusion_matrix(y_true, y_pred, class_names, model_name="", report_labels=None):
    """Plots and saves the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=report_labels)
    print("\n--- Confusion Matrix (Numerical Data) ---")
    print(f"Shape of confusion matrix array (cm): {cm.shape}")
    print("Confusion matrix (cm):\n", cm)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 10})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14)
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    cm_filename = os.path.join(config.RESULTS_DIR, f"{model_name.replace('.pt', '')}_confusion_matrix.png")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plt.savefig(cm_filename)
    print(f"Confusion matrix plot saved to {cm_filename}")

def evaluate_model(model_class_name, 
                   model_filename, 
                   model_specific_params=None, # Dictionary for model's architectural HPs
                   test_batch_size=config.BATCH_SIZE):
    """
    Evaluates a trained model on the predefined test set.
    Args:
        model_specific_params (dict, optional): Architectural hyperparameters for model instantiation.
    """
    print(f"Evaluating model: {model_filename} (Class: {model_class_name}) on device: {config.DEVICE}")
    if model_specific_params:
        print(f"  Using model-specific architectural HPs for instantiation: {model_specific_params}")

    # 1. Load Data Splits
    _X_train_discard, _X_val_discard, X_test, \
    _y_train_discard, _y_val_discard, y_test, \
    label_encoder, _scaler_discard = data_utils.get_stratified_data_splits()

    if X_test is None or y_test is None or label_encoder is None:
        print("Failed to load or split data for evaluation. Exiting.")
        return
    if X_test.size == 0:
        print("Test set (X_test) is empty. Cannot evaluate.")
        return

    test_dataset = EmotionDataset(features=X_test, labels=y_test)
    if len(test_dataset) == 0:
        print("Test dataset is empty after EmotionDataset creation. Cannot evaluate.")
        return

    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_dataset)}")

    # 2. Instantiate Model with correct architectural parameters
    try:
        ModelClass = getattr(models, model_class_name)
        init_params = {"num_classes": config.NUM_CLASSES}
        if model_specific_params:
            init_params.update(model_specific_params)
        
        # Ensure dropout_rate is passed if not in model_specific_params and model expects it
        # For evaluation, dropout is off via model.eval(), but instantiation needs to match.
        if 'dropout_rate' not in init_params and model_class_name == "CNNEmotion":
             init_params['dropout_rate'] = 0.1 # Default, actual value less critical for eval instantiation

        model_instance = ModelClass(**init_params) 
        print(f"  Instantiated {model_class_name} for loading with params: {init_params}")

    except AttributeError:
        print(f"Error: Model class '{model_class_name}' not found in models.py.")
        return
    except TypeError as e:
        print(f"Error instantiating model {model_class_name}. Check its __init__ parameters and provided model_specific_params: {e}")
        return
        
    # Load saved weights
    model = load_model(model_instance, model_filename, device=config.DEVICE)
    if not hasattr(model, 'state_dict') or model is model_instance and not os.path.exists(os.path.join(config.MODEL_SAVE_DIR, model_filename)): 
        # Second condition checks if load_model returned original instance due to file not found
        print(f"Failed to load model weights for {model_filename}. Exiting.")
        return

    # 3. Evaluation Loop (remains the same)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels_batch in tqdm(test_loader, desc="Evaluating Test Set", leave=False): 
            inputs = inputs.to(config.DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

    if not all_labels or not all_preds:
        print("No predictions or labels collected from the test set.")
        return

    # 4. Metrics (remains the same)
    class_names = label_encoder.classes_ 
    report_labels_indices = np.arange(config.NUM_CLASSES) 
    print("\n--- Classification Report (Test Set) ---")
    report = classification_report(all_labels, all_preds, labels=report_labels_indices, target_names=class_names, zero_division=0)
    print(report)
    report_filename = os.path.join(config.RESULTS_DIR, f"{model_filename.replace('.pt', '')}_TEST_classification_report.txt")
    with open(report_filename, 'w') as f: f.write(report)
    print(f"Test classification report saved to {report_filename}")
    plot_confusion_matrix(all_labels, all_preds, class_names, f"{model_filename.replace('.pt', '')}_TEST", report_labels=report_labels_indices)

if __name__ == '__main__':
    if not all(os.path.exists(f) for f in [config.PROCESSED_DATA_FILE_X, config.PROCESSED_DATA_FILE_Y, config.LABEL_ENCODER_FILE, config.SCALER_FILE]):
        print("Processed data/encoder/scaler not found. Running data preprocessing steps first...")
        data_utils.filter_ravdess_files()
        data_utils.process_and_save_all_data()
        print("Data preprocessing complete. You can now run evaluate.py.")
    
    # --- Configuration for this evaluation run ---
    MODEL_CLASS_TO_EVALUATE = "SequentialCRNN" # Or "CNNEmotion", "CnnLstmFusion"
    # This should be the filename of the model trained with the HPs below
    SAVED_MODEL_FILENAME = "sequentialcrnn_final_trained_v2.pt" 

    # **IMPORTANT**: These architectural HPs MUST match those used to train SAVED_MODEL_FILENAME
    model_arch_hps = {}
    if MODEL_CLASS_TO_EVALUATE == "CNNEmotion":
        model_arch_hps = {"dropout_rate": 0.25} # Example
    elif MODEL_CLASS_TO_EVALUATE == "CnnLstmFusion":
        model_arch_hps = { # Example values, use actual trained values
            "cnn_dropout_rate": 0.15, "lstm_hidden_size": 128, "lstm_num_layers": 2,
            "lstm_dropout_rate": 0.25, "fusion_dropout_rate": 0.4, "lstm_bidirectional": True
        }
    elif MODEL_CLASS_TO_EVALUATE == "SequentialCRNN":
        model_arch_hps = { # Example values, use actual trained values
            "cnn_out_channels": 64, "lstm_hidden_size": 128, "lstm_num_layers": 1,
            "cnn_dropout_rate": 0.1, "lstm_dropout_rate": 0.2, "classifier_dropout_rate": 0.3,
            "lstm_bidirectional": True
        }
    # Add more 'elif' blocks for other models

    model_path_to_check = os.path.join(config.MODEL_SAVE_DIR, SAVED_MODEL_FILENAME)
    if not os.path.exists(model_path_to_check):
        print(f"ERROR: Model file '{SAVED_MODEL_FILENAME}' not found in '{config.MODEL_SAVE_DIR}'.")
        print("Please train the model or check the filename and its architectural HPs.")
    else:
        evaluate_model(
            model_class_name=MODEL_CLASS_TO_EVALUATE,
            model_filename=SAVED_MODEL_FILENAME,
            model_specific_params=model_arch_hps # Pass the architectural HPs
        )
# evaluate.py
"""
Script for evaluating a trained emotion detection model on the test set.
Handles model-specific hyperparameters for instantiation.
"""
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

import config
import data_utils 
import models     
from dataset import EmotionDataset
from train_utils import load_model

def plot_confusion_matrix(y_true, y_pred, class_names, model_name="", report_labels=None):
    """Plots and saves the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=report_labels)
    print("\n--- Confusion Matrix (Numerical Data) ---")
    print(f"Shape of confusion matrix array (cm): {cm.shape}")
    print("Confusion matrix (cm):\n", cm)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 10})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14)
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    cm_filename = os.path.join(config.RESULTS_DIR, f"{model_name.replace('.pt', '')}_confusion_matrix.png")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plt.savefig(cm_filename)
    print(f"Confusion matrix plot saved to {cm_filename}")

def evaluate_model(model_class_name, 
                   model_filename, 
                   model_specific_params=None, # Dictionary for model's architectural HPs
                   test_batch_size=config.BATCH_SIZE):
    """
    Evaluates a trained model on the predefined test set.
    Args:
        model_specific_params (dict, optional): Architectural hyperparameters for model instantiation.
    """
    print(f"Evaluating model: {model_filename} (Class: {model_class_name}) on device: {config.DEVICE}")
    if model_specific_params:
        print(f"  Using model-specific architectural HPs for instantiation: {model_specific_params}")

    # 1. Load Data Splits
    _X_train_discard, _X_val_discard, X_test, \
    _y_train_discard, _y_val_discard, y_test, \
    label_encoder, _scaler_discard = data_utils.get_stratified_data_splits()

    if X_test is None or y_test is None or label_encoder is None:
        print("Failed to load or split data for evaluation. Exiting.")
        return
    if X_test.size == 0:
        print("Test set (X_test) is empty. Cannot evaluate.")
        return

    test_dataset = EmotionDataset(features=X_test, labels=y_test)
    if len(test_dataset) == 0:
        print("Test dataset is empty after EmotionDataset creation. Cannot evaluate.")
        return

    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_dataset)}")

    # 2. Instantiate Model with correct architectural parameters
    try:
        ModelClass = getattr(models, model_class_name)
        init_params = {"num_classes": config.NUM_CLASSES}
        if model_specific_params:
            init_params.update(model_specific_params)
        
        # Ensure dropout_rate is passed if not in model_specific_params and model expects it
        # For evaluation, dropout is off via model.eval(), but instantiation needs to match.
        if 'dropout_rate' not in init_params and model_class_name == "CNNEmotion":
             init_params['dropout_rate'] = 0.1 # Default, actual value less critical for eval instantiation

        model_instance = ModelClass(**init_params) 
        print(f"  Instantiated {model_class_name} for loading with params: {init_params}")

    except AttributeError:
        print(f"Error: Model class '{model_class_name}' not found in models.py.")
        return
    except TypeError as e:
        print(f"Error instantiating model {model_class_name}. Check its __init__ parameters and provided model_specific_params: {e}")
        return
        
    # Load saved weights
    model = load_model(model_instance, model_filename, device=config.DEVICE)
    if not hasattr(model, 'state_dict') or model is model_instance and not os.path.exists(os.path.join(config.MODEL_SAVE_DIR, model_filename)): 
        # Second condition checks if load_model returned original instance due to file not found
        print(f"Failed to load model weights for {model_filename}. Exiting.")
        return

    # 3. Evaluation Loop (remains the same)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels_batch in tqdm(test_loader, desc="Evaluating Test Set", leave=False): 
            inputs = inputs.to(config.DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

    if not all_labels or not all_preds:
        print("No predictions or labels collected from the test set.")
        return

    # 4. Metrics (remains the same)
    class_names = label_encoder.classes_ 
    report_labels_indices = np.arange(config.NUM_CLASSES) 
    print("\n--- Classification Report (Test Set) ---")
    report = classification_report(all_labels, all_preds, labels=report_labels_indices, target_names=class_names, zero_division=0)
    print(report)
    report_filename = os.path.join(config.RESULTS_DIR, f"{model_filename.replace('.pt', '')}_TEST_classification_report.txt")
    with open(report_filename, 'w') as f: f.write(report)
    print(f"Test classification report saved to {report_filename}")
    plot_confusion_matrix(all_labels, all_preds, class_names, f"{model_filename.replace('.pt', '')}_TEST", report_labels=report_labels_indices)

if __name__ == '__main__':
    # Ensure data is preprocessed
    if not all(os.path.exists(f) for f in [config.PROCESSED_DATA_FILE_X, 
                                           config.PROCESSED_DATA_FILE_Y,
                                           config.LABEL_ENCODER_FILE,
                                           config.SCALER_FILE]):
        print("Processed data/encoder/scaler not found. Running data preprocessing steps first...")
        data_utils.filter_ravdess_files()
        data_utils.process_and_save_all_data()
        print("Data preprocessing complete. You can now run evaluate.py.")
    else:
        print("Processed data, encoder, and scaler found.")
    
    # --- Configuration for this evaluation run ---
    MODEL_CLASS_TO_EVALUATE = "SequentialCRNN" 
    # This should be the filename of the model trained with the HPs below
    SAVED_MODEL_FILENAME = "sequentialcrnn_final_trained_v2.pt" 

    # **IMPORTANT**: These architectural HPs MUST match those used to train SAVED_MODEL_FILENAME
    # Using the HPO results you provided for SequentialCRNN:
    model_arch_hps = {}
    if MODEL_CLASS_TO_EVALUATE == "SequentialCRNN":
        model_arch_hps = {
            "cnn_out_channels": 32,
            "lstm_hidden_size": 128,
            "lstm_num_layers": 2,
            "cnn_dropout_rate": 0.05422795470218011,
            "lstm_dropout_rate": 0.39974605818052883,
            "classifier_dropout_rate": 0.2985000185938243,
            "lstm_bidirectional": True # Assuming this was True during training (default in model)
        }
    elif MODEL_CLASS_TO_EVALUATE == "CNNEmotion":
        # Example if you were evaluating CNNEmotion, adjust with its HPO results
        model_arch_hps = { 
            "dropout_rate": 0.25 # Replace with actual HPO dropout for CNNEmotion
        }
    elif MODEL_CLASS_TO_EVALUATE == "CnnLstmFusion":
        # Example if you were evaluating CnnLstmFusion
        model_arch_hps = { 
            "cnn_dropout_rate": 0.1699953273472689,
            "lstm_hidden_size": 128,
            "lstm_num_layers": 1,
            "lstm_dropout_rate": 0.39812532016294355,
            "fusion_dropout_rate": 0.3023622942508569,
            "lstm_bidirectional": True
        }
    # Add more 'elif' blocks for other models if you evaluate them

    model_path_to_check = os.path.join(config.MODEL_SAVE_DIR, SAVED_MODEL_FILENAME)
    if not os.path.exists(model_path_to_check):
        print(f"ERROR: Model file '{SAVED_MODEL_FILENAME}' not found in '{config.MODEL_SAVE_DIR}'.")
        print("Please train the model or check the filename and its architectural HPs.")
    else:
        print(f"Attempting to evaluate model: {SAVED_MODEL_FILENAME}")
        print(f"Using architectural HPs for instantiation: {model_arch_hps}")
        evaluate_model(
            model_class_name=MODEL_CLASS_TO_EVALUATE,
            model_filename=SAVED_MODEL_FILENAME,
            model_specific_params=model_arch_hps # Pass the architectural HPs
        )
