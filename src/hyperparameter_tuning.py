# hyperparameter_tuning.py
"""
Script for hyperparameter optimization using Optuna.
Supports multiple model architectures.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna
import os
import numpy as np

import config
import data_utils 
import models     
from dataset import EmotionDataset
from train_utils import train_epoch, validate_epoch

# Global cache for loaded data splits
_DATA_SPLITS_CACHE = {}

def get_hpo_data_splits():
    """
    Loads and returns stratified train and validation data splits using data_utils.
    Caches the splits for subsequent calls within the same HPO run.
    """
    if "train_dataset" not in _DATA_SPLITS_CACHE:
        X_train, X_val, _, \
        y_train, y_val, _, \
        label_encoder, _ = data_utils.get_stratified_data_splits()

        if X_train is None:
            raise RuntimeError("Failed to get stratified data splits for HPO. Run data_utils.process_and_save_all_data() first.")

        train_dataset = EmotionDataset(features=X_train, labels=y_train)
        val_dataset = None
        if X_val.size > 0:
            val_dataset = EmotionDataset(features=X_val, labels=y_val)
        else:
            print("Warning: Validation set (X_val) is empty for HPO. This might affect HPO performance if not handled in objective.")
        
        _DATA_SPLITS_CACHE["train_dataset"] = train_dataset
        _DATA_SPLITS_CACHE["val_dataset"] = val_dataset
        # _DATA_SPLITS_CACHE["label_encoder"] = label_encoder # Not directly used in objective but good to have if needed
        
        print(f"Data splits loaded for HPO: Train {len(train_dataset) if train_dataset else 0}, Val {len(val_dataset) if val_dataset else 0}")

    return _DATA_SPLITS_CACHE["train_dataset"], _DATA_SPLITS_CACHE["val_dataset"]


def _run_trial_epoch_loop(trial, model, train_loader, val_loader, criterion, optimizer, scheduler):
    """Helper function for the epoch loop within an Optuna trial."""
    best_val_accuracy_trial = 0.0
    epochs_no_improve = 0

    for epoch in range(config.OPTUNA_TRIAL_EPOCHS):
        _, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        
        current_val_acc = -1.0 # Default if no validation
        if val_loader and len(val_loader.dataset) > 0: # Check if val_loader is not None and has data
            val_loss, val_acc_epoch = validate_epoch(model, val_loader, criterion, config.DEVICE)
            current_val_acc = val_acc_epoch
            trial.set_user_attr(f"epoch_{epoch+1}_val_loss", val_loss)
        else: 
            print(f"Warning: No validation data for trial {trial.number}, epoch {epoch+1}. Optuna will use train_acc or a default bad value.")
            # Optuna needs a metric. If no val_acc, using train_acc is an option, but less ideal.
            # Or return a fixed bad score if validation is essential.
            # For now, if val_loader is None, current_val_acc remains -1.0 which is a bad score.
            # If you want to use train_acc: current_val_acc = train_acc
            pass


        trial.set_user_attr(f"epoch_{epoch+1}_train_acc", train_acc)
        trial.set_user_attr(f"epoch_{epoch+1}_val_acc", current_val_acc if current_val_acc != -1.0 else train_acc)


        if scheduler:
            scheduler.step()

        # Use current_val_acc for improvement check if available, else fallback (though HPO should focus on val_acc)
        metric_to_check = current_val_acc if val_loader and len(val_loader.dataset) > 0 else train_acc

        if metric_to_check > best_val_accuracy_trial: # Note: if no val, this becomes best_train_accuracy_trial
            best_val_accuracy_trial = metric_to_check
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        trial.report(metric_to_check, epoch) 
        if trial.should_prune():
            print(f"Trial {trial.number} pruned at epoch {epoch+1}.")
            raise optuna.exceptions.TrialPruned()

        if epochs_no_improve >= config.OPTUNA_PATIENCE:
            print(f"Trial {trial.number} early stopped at epoch {epoch+1}.")
            break
    return best_val_accuracy_trial


def objective_cnn(trial: optuna.trial.Trial):
    """Optuna objective function for CNNEmotion model."""
    train_dataset, val_dataset = get_hpo_data_splits()

    if val_dataset is None or len(val_dataset) == 0 :
        print("Error: Validation dataset is empty for CNN HPO trial. Returning -1.0 (bad score).")
        return -1.0 

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5) # General dropout for CNNEmotion
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = models.CNNEmotion(
        num_classes=config.NUM_CLASSES,
        dropout_rate=dropout_rate 
    ).to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    scheduler_gamma = trial.suggest_float("scheduler_gamma", 0.7, 0.99)
    scheduler_step_size = trial.suggest_int("scheduler_step_size", 3, 10)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    
    return _run_trial_epoch_loop(trial, model, train_loader, val_loader, criterion, optimizer, scheduler)


def objective_cnnlstmfusion(trial: optuna.trial.Trial):
    """Optuna objective function for CnnLstmFusion model."""
    train_dataset, val_dataset = get_hpo_data_splits()

    if val_dataset is None or len(val_dataset) == 0:
        print("Error: Validation dataset is empty for CnnLstmFusion HPO trial. Returning -1.0.")
        return -1.0

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    
    cnn_dropout_rate = trial.suggest_float("cnn_dropout_rate", 0.05, 0.4)
    lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size", [64, 128]) 
    lstm_num_layers = trial.suggest_int("lstm_num_layers", 1, 2) 
    lstm_dropout_rate = trial.suggest_float("lstm_dropout_rate", 0.05, 0.4)
    fusion_dropout_rate = trial.suggest_float("fusion_dropout_rate", 0.1, 0.5)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = models.CnnLstmFusion(
        num_classes=config.NUM_CLASSES,
        cnn_dropout_rate=cnn_dropout_rate,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
        lstm_dropout_rate=lstm_dropout_rate,
        lstm_bidirectional=True, 
        fusion_dropout_rate=fusion_dropout_rate
    ).to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler_gamma = trial.suggest_float("scheduler_gamma", 0.7, 0.99)
    scheduler_step_size = trial.suggest_int("scheduler_step_size", 3, 10)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    return _run_trial_epoch_loop(trial, model, train_loader, val_loader, criterion, optimizer, scheduler)

def objective_sequentialcrnn(trial: optuna.trial.Trial):
    """Optuna objective function for SequentialCRNN model."""
    train_dataset, val_dataset = get_hpo_data_splits()
    if val_dataset is None or len(val_dataset) == 0: 
        print("Error: Validation dataset is empty for SequentialCRNN HPO trial. Returning -1.0.")
        return -1.0

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    
    cnn_out_channels = trial.suggest_categorical("cnn_out_channels", [32, 64])
    lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size", [64, 128, 256])
    lstm_num_layers = trial.suggest_int("lstm_num_layers", 1, 2)
    cnn_dropout_rate = trial.suggest_float("cnn_dropout_rate", 0.05, 0.3) # For Dropout2d
    lstm_dropout_rate = trial.suggest_float("lstm_dropout_rate", 0.1, 0.4)
    classifier_dropout_rate = trial.suggest_float("classifier_dropout_rate", 0.2, 0.5)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = models.SequentialCRNN(
        num_classes=config.NUM_CLASSES,
        cnn_out_channels=cnn_out_channels,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
        cnn_dropout_rate=cnn_dropout_rate,
        lstm_dropout_rate=lstm_dropout_rate,
        classifier_dropout_rate=classifier_dropout_rate,
        lstm_bidirectional=True 
    ).to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler_gamma = trial.suggest_float("scheduler_gamma", 0.7, 0.99)
    scheduler_step_size = trial.suggest_int("scheduler_step_size", 3, 10)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    return _run_trial_epoch_loop(trial, model, train_loader, val_loader, criterion, optimizer, scheduler)


OBJECTIVE_FUNCTIONS = {
    "CNNEmotion": objective_cnn,
    "CnnLstmFusion": objective_cnnlstmfusion,
    "SequentialCRNN": objective_sequentialcrnn,
}

if __name__ == "__main__":
    # Ensure data is preprocessed
    if not all(os.path.exists(f) for f in [config.PROCESSED_DATA_FILE_X, 
                                           config.PROCESSED_DATA_FILE_Y,
                                           config.LABEL_ENCODER_FILE,
                                           config.SCALER_FILE]):
        print("Processed data/encoder/scaler not found. Running data preprocessing steps...")
        data_utils.filter_ravdess_files()
        data_utils.process_and_save_all_data()
        print("Data preprocessing complete. You can now run HPO.")
    else:
        print("Processed data, encoder, and scaler found.")
    
    MODEL_TO_TUNE = "CnnLstmFusion" # Change this to tune other models

    if MODEL_TO_TUNE not in OBJECTIVE_FUNCTIONS:
        print(f"Error: Objective function for '{MODEL_TO_TUNE}' not defined.")
        print(f"Available objective functions for: {list(OBJECTIVE_FUNCTIONS.keys())}")
    else:
        objective_func = OBJECTIVE_FUNCTIONS[MODEL_TO_TUNE]
        study_name = f"{MODEL_TO_TUNE}_hpo_study_v4" # Updated study name
        
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5, 
            n_warmup_steps=max(1, config.OPTUNA_TRIAL_EPOCHS // 3), 
            interval_steps=1
        )

        # Ensure results directory exists for SQLite DB
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        db_path = os.path.join(config.RESULTS_DIR, f"{study_name}.db")
        storage_uri = f"sqlite:///{db_path}"


        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            pruner=pruner,
            storage=storage_uri,
            load_if_exists=True
        )

        print(f"\nStarting Optuna study for {MODEL_TO_TUNE} (Study: {study_name})...")
        print(f"  Number of trials: {config.OPTUNA_N_TRIALS}")
        print(f"  Epochs per trial: {config.OPTUNA_TRIAL_EPOCHS}")
        print(f"  Early stopping patience per trial: {config.OPTUNA_PATIENCE}")
        print(f"  Study results DB: {storage_uri}")

        try:
            study.optimize(objective_func, n_trials=config.OPTUNA_N_TRIALS, timeout=None)
        except KeyboardInterrupt:
            print("HPO study interrupted by user.")
        except Exception as e:
            print(f"An error occurred during HPO: {e}")

        print("\nHyperparameter optimization finished.")
        print(f"Number of finished trials: {len(study.trials)}")

        if study.trials: # Check if there are any trials
            try:
                best_trial = study.best_trial
                print(f"Best trial for {MODEL_TO_TUNE}:")
                print(f"  Value (Max Validation Accuracy): {best_trial.value:.4f}")
                print("  Best Parameters:")
                for key, value in best_trial.params.items():
                    print(f"    {key}: {value}")
            except ValueError: # Handles case where no trials completed successfully
                 print("No successful trials completed to determine the best one.")
        else:
            print("No trials were run or completed.")
        
        try:
            study_df = study.trials_dataframe()
            if not study_df.empty:
                csv_path = os.path.join(config.RESULTS_DIR, f"{study_name}_results.csv")
                study_df.to_csv(csv_path, index=False)
                print(f"Study results DataFrame saved to {csv_path}")
            else:
                print("Study DataFrame is empty, not saving CSV.")
        except Exception as e:
            print(f"Could not save study results to CSV: {e}")
