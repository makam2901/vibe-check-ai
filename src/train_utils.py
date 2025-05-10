# train_utils.py
"""
Utilities for model training, validation, saving, and loading.
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import config

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Performs a single training epoch.
    Args:
        model (torch.nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): The device to train on (e.g., 'cuda' or 'cpu').
    Returns:
        tuple: (epoch_loss, epoch_accuracy)
    """
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Zero the parameter gradients

        outputs = model(inputs) # Forward pass
        loss = criterion(outputs, labels) # Calculate loss

        loss.backward()  # Backward pass
        optimizer.step() # Optimize

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions.double() / total_samples
    return epoch_loss, epoch_accuracy.item()

def validate_epoch(model, dataloader, criterion, device):
    """
    Performs a single validation epoch.
    Args:
        model (torch.nn.Module): The model to validate.
        dataloader (torch.utils.data.DataLoader): DataLoader for validation data.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): The device to validate on.
    Returns:
        tuple: (epoch_loss, epoch_accuracy)
    """
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient calculations
        for inputs, labels in tqdm(dataloader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions.double() / total_samples
    return epoch_loss, epoch_accuracy.item()

def save_model(model, model_name, save_dir=config.MODEL_SAVE_DIR):
    """
    Saves the model's state dictionary.
    Args:
        model (torch.nn.Module): The model to save.
        model_name (str): Name of the model file (e.g., "cnn_best.pt").
        save_dir (str): Directory to save the model.
    """
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, model_name)
    try:
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

def load_model(model, model_name, load_dir=config.MODEL_SAVE_DIR, device=config.DEVICE):
    """
    Loads the model's state dictionary.
    Args:
        model (torch.nn.Module): The model instance to load weights into.
        model_name (str): Name of the model file.
        load_dir (str): Directory from where to load the model.
        device (torch.device): Device to map the loaded model to.
    Returns:
        torch.nn.Module: The model with loaded weights.
                         Returns the original model if file not found.
    """
    model_path = os.path.join(load_dir, model_name)
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device) # Ensure model is on the correct device
            model.eval() # Set to eval mode after loading
            print(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            return model # Return original model on error
    else:
        print(f"Model file not found at {model_path}. Returning an untrained model.")
        return model

if __name__ == '__main__':
    print("Testing train_utils functions...")
    # This requires dummy model, dataloaders, criterion, optimizer.
    # For a quick test, we'll just check if functions are defined.

    # --- Dummy components for testing ---
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(config.FEATURE_DIM, config.NUM_CLASSES)
        def forward(self, x):
            # Assuming x is (batch, time_steps, features), take mean over time
            x = torch.mean(x, dim=1)
            return self.linear(x)

    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np

    # Create dummy data
    bs = 4 # batch_size
    num_samples = 20
    X_dummy = torch.randn(num_samples, config.TIME_STEPS, config.FEATURE_DIM)
    y_dummy = torch.randint(0, config.NUM_CLASSES, (num_samples,))
    dummy_dataset = TensorDataset(X_dummy, y_dummy)
    dummy_dataloader = DataLoader(dummy_dataset, batch_size=bs)

    device = config.DEVICE
    model = DummyModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"\nUsing device: {device}")

    # Test train_epoch
    print("\nTesting train_epoch...")
    try:
        train_loss, train_acc = train_epoch(model, dummy_dataloader, criterion, optimizer, device)
        print(f"Dummy train_epoch completed. Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
    except Exception as e:
        print(f"Error in train_epoch test: {e}")

    # Test validate_epoch
    print("\nTesting validate_epoch...")
    try:
        val_loss, val_acc = validate_epoch(model, dummy_dataloader, criterion, device)
        print(f"Dummy validate_epoch completed. Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
    except Exception as e:
        print(f"Error in validate_epoch test: {e}")

    # Test save_model and load_model
    print("\nTesting save_model and load_model...")
    test_model_name = "dummy_model_test.pt"
    save_model(model, test_model_name)
    
    new_model_instance = DummyModel() # Create a new instance
    loaded_model = load_model(new_model_instance, test_model_name, device=device)
    
    # Basic check if loading worked (e.g., by comparing a parameter)
    if loaded_model and hasattr(model, 'linear') and hasattr(loaded_model, 'linear'):
        if torch.equal(model.linear.weight, loaded_model.linear.weight):
            print("Model save and load test PASSED (parameter check).")
        else:
            print("Model save and load test FAILED (parameter mismatch).")
    else:
        print("Model save and load test FAILED (model not loaded or attributes missing).")
    
    # Clean up dummy model file
    if os.path.exists(os.path.join(config.MODEL_SAVE_DIR, test_model_name)):
        os.remove(os.path.join(config.MODEL_SAVE_DIR, test_model_name))
        print(f"Cleaned up {test_model_name}")

    print("\ntrain_utils tests complete.")
