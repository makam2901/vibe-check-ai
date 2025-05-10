# train.py
"""
Main script for training an emotion detection model.
Allows selection of different model architectures and uses stratified data splits.
Handles model-specific hyperparameters via a dictionary.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt

import config
import data_utils 
import models     
from dataset import EmotionDataset
from train_utils import train_epoch, validate_epoch, save_model

def plot_training_history(train_losses, val_losses, train_accs, val_accs, model_name_plot):
    """Plots and saves training and validation loss and accuracy."""
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    if val_losses and not all(np.isnan(val_losses)):
        plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.title(f'Training and Validation Loss ({model_name_plot})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'bo-', label='Training Accuracy')
    if val_accs and not all(np.isnan(val_accs)):
        plt.plot(epochs, val_accs, 'ro-', label='Validation Accuracy')
    plt.title(f'Training and Validation Accuracy ({model_name_plot})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plot_filename = os.path.join(config.RESULTS_DIR, f"{model_name_plot.replace('.pt', '')}_training_history.png")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plt.savefig(plot_filename)
    print(f"Training history plot saved to {plot_filename}")

def main_train(model_class_name="CNNEmotion",
               learning_rate=config.DEFAULT_LEARNING_RATE,
               num_epochs=config.DEFAULT_EPOCHS,
               batch_size=config.BATCH_SIZE,
               model_save_name="default_best_model.pt",
               model_specific_params=None, # Dictionary for model-specific HPs
               scheduler_gamma_param=config.SCHEDULER_GAMMA,
               scheduler_step_size_param=config.SCHEDULER_STEP_SIZE
               ):
    """
    Main training function.
    Args:
        model_specific_params (dict, optional): Dictionary containing hyperparameters
                                                specific to the model's __init__ method.
    """
    print(f"Starting training for model: {model_class_name}")
    print(f"  Learning Rate: {learning_rate}, Batch Size: {batch_size}")
    print(f"  Epochs: {num_epochs}, Device: {config.DEVICE}")
    print(f"  Model will be saved as: {model_save_name}")
    if model_specific_params:
        print(f"  Model-specific HPs: {model_specific_params}")


    # 1. Load Data Splits
    X_train, X_val, _, \
    y_train, y_val, _, \
    label_encoder, _ = data_utils.get_stratified_data_splits()

    if X_train is None:
        print("Failed to get stratified data splits. Exiting.")
        return
    
    train_dataset = EmotionDataset(features=X_train, labels=y_train)
    val_dataset = None
    if X_val.size > 0:
        val_dataset = EmotionDataset(features=X_val, labels=y_val)
    else:
        print("Warning: Validation set is empty.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True if config.DEVICE.type == 'cuda' else False)
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True if config.DEVICE.type == 'cuda' else False)

    print(f"Training samples: {len(train_dataset)}")
    if val_dataset: print(f"Validation samples: {len(val_dataset)}")

    # 3. Initialize Model, Criterion, Optimizer
    try:
        ModelClass = getattr(models, model_class_name)
        
        # Prepare parameters for model instantiation
        init_params = {"num_classes": config.NUM_CLASSES}
        if model_specific_params:
            init_params.update(model_specific_params)
        
        # Ensure dropout_rate is passed if not in model_specific_params and model expects it
        # (e.g., for a generic CNNEmotion if not all its params are in model_specific_params)
        if 'dropout_rate' not in init_params and model_class_name == "CNNEmotion":
             init_params['dropout_rate'] = 0.1 # Default or from a general HP

        model = ModelClass(**init_params).to(config.DEVICE)
        print(f"  Instantiated {model_class_name} with params: {init_params}")

    except AttributeError:
        print(f"Error: Model class '{model_class_name}' not found in models.py.")
        available_models = [name for name, obj in models.__dict__.items() if isinstance(obj, type) and issubclass(obj, nn.Module) and obj is not nn.Module]
        print(f"Available models in models.py: {available_models}")
        return
    except TypeError as e:
        print(f"Error instantiating model {model_class_name}. Check its __init__ parameters and provided model_specific_params: {e}")
        return

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=scheduler_step_size_param,
                                          gamma=scheduler_gamma_param)

    # 4. Training Loop (remains largely the same)
    best_val_accuracy = 0.0
    epochs_no_improve = 0
    train_losses_history, val_losses_history = [], []
    train_accs_history, val_accs_history = [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        train_losses_history.append(train_loss); train_accs_history.append(train_acc)
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        current_val_acc = 0.0
        if val_loader:
            val_loss, val_acc_epoch = validate_epoch(model, val_loader, criterion, config.DEVICE)
            val_losses_history.append(val_loss); val_accs_history.append(val_acc_epoch)
            current_val_acc = val_acc_epoch
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {current_val_acc:.4f}")
        else:
            val_losses_history.append(np.nan); val_accs_history.append(np.nan)

        if scheduler: scheduler.step(); print(f"  Current LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_loader:
            if current_val_acc > best_val_accuracy:
                print(f"  Validation accuracy improved ({best_val_accuracy:.4f} --> {current_val_acc:.4f}). Saving model...")
                best_val_accuracy = current_val_acc
                save_model(model, model_save_name)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"  Validation accuracy did not improve. Early stopping counter: {epochs_no_improve}/{config.EARLY_STOPPING_PATIENCE}")
        else:
            print(f"  No validation set. Saving model for epoch {epoch+1}.")
            save_model(model, model_save_name)

        if val_loader and epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered based on validation accuracy.")
            break
            
    print("\nTraining complete.")
    if val_loader: print(f"Best validation accuracy achieved: {best_val_accuracy:.4f}")
    plot_training_history(train_losses_history, val_losses_history, train_accs_history, val_accs_history, model_save_name)

if __name__ == '__main__':
    if not all(os.path.exists(f) for f in [config.PROCESSED_DATA_FILE_X, 
                                           config.PROCESSED_DATA_FILE_Y,
                                           config.LABEL_ENCODER_FILE,
                                           config.SCALER_FILE]):
        print("Processed data/encoder/scaler not found. Running data preprocessing steps...")
        data_utils.filter_ravdess_files() 
        data_utils.process_and_save_all_data() 
        print("Data preprocessing complete. You can now run train.py again.")
    else:
        print("Processed data, encoder, and scaler found.")

    # --- Configuration for this training run ---
    # SELECTED_MODEL_CLASS = "CNNEmotion"
    SELECTED_MODEL_CLASS = "SequentialCRNN"
    # SELECTED_MODEL_CLASS = "SequentialCRNN"
    
    # General training parameters (can be from HPO or set manually)
    LEARNING_RATE = 0.0003594104449862055
    BATCH_SIZE = 32
    NUM_EPOCHS = config.DEFAULT_EPOCHS 
    SCHEDULER_GAMMA = 0.8486511262477342
    SCHEDULER_STEP = 4
    
    # Dictionary for model-specific hyperparameters
    model_hps = {}

    if SELECTED_MODEL_CLASS == "CNNEmotion":
        # HPs for CNNEmotion (primarily dropout_rate)
        model_hps = {
            "dropout_rate": 0.25 # Example from a hypothetical HPO
        }
    elif SELECTED_MODEL_CLASS == "CnnLstmFusion":
        # HPs for CnnLstmFusion (examples)
        model_hps = {
            "cnn_dropout_rate": 0.1699953273472689,
            "lstm_hidden_size": 128,
            "lstm_num_layers": 1,
            "lstm_dropout_rate": 0.39812532016294355,
            "fusion_dropout_rate": 0.3023622942508569,
            "lstm_bidirectional": True 
        }
    elif SELECTED_MODEL_CLASS == "SequentialCRNN":
        # HPs for SequentialCRNN (examples)
        model_hps = {
            "cnn_out_channels": 32,
            "lstm_hidden_size": 128,
            "lstm_num_layers": 2,
            "cnn_dropout_rate": 0.05422795470218011,
            "lstm_dropout_rate": 0.39974605818052883,
            "classifier_dropout_rate": 0.2985000185938243,
            "lstm_bidirectional": True
        }
    # Add more 'elif' blocks for other models and their specific HPs

    MODEL_FILENAME = f"{SELECTED_MODEL_CLASS.lower()}_final_trained_v2.pt" # Updated filename

    print(f"\n--- Preparing to Train: {SELECTED_MODEL_CLASS} ---")
    print(f"  LR: {LEARNING_RATE}, Batch: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}")
    print(f"  Scheduler Gamma: {SCHEDULER_GAMMA}, Scheduler Step: {SCHEDULER_STEP}")
    if model_hps:
        print(f"  Model Specific HPs: {model_hps}")
    
    main_train(
        model_class_name=SELECTED_MODEL_CLASS,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        model_save_name=MODEL_FILENAME,
        model_specific_params=model_hps, # Pass the dictionary
        scheduler_gamma_param=SCHEDULER_GAMMA,
        scheduler_step_size_param=SCHEDULER_STEP
    )
