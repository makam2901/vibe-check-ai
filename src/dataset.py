# dataset.py
"""
Defines the PyTorch Dataset class for loading emotion speech data.
"""
import torch
from torch.utils.data import Dataset
import numpy as np # For the example usage
import config

class EmotionDataset(Dataset):
    """
    Custom PyTorch Dataset for emotion speech recognition.
    Args:
        features (np.array or torch.Tensor): Preprocessed audio features.
        labels (np.array or torch.Tensor): Encoded integer labels.
        transform (callable, optional): Optional transform for augmentation.
    """
    def __init__(self, features, labels, transform=None):
        if not isinstance(features, torch.Tensor):
            self.features = torch.tensor(features, dtype=torch.float32)
        else:
            self.features = features.float()
            
        if not isinstance(labels, torch.Tensor):
            self.labels = torch.tensor(labels, dtype=torch.long)
        else:
            self.labels = labels.long()
            
        self.transform = transform

        if self.features.shape[0] != self.labels.shape[0]:
            raise ValueError("Mismatch in number of samples between features and labels.")
        
        if self.features.ndim > 1: # If not an empty dataset
            if self.features.shape[1] != config.TIME_STEPS or \
               self.features.shape[2] != config.FEATURE_DIM:
                print(f"Warning: Dataset shape ({self.features.shape[1]}, {self.features.shape[2]}) "
                      f"vs config ({config.TIME_STEPS}, {config.FEATURE_DIM}).")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature_sample = self.features[idx]
        label_sample = self.labels[idx]

        if self.transform:
            # Placeholder for augmentation logic
            # feature_sample = self.transform(feature_sample)
            pass

        return feature_sample, label_sample

if __name__ == '__main__':
    print("Running example usage of EmotionDataset...")
    num_dummy_samples = 100
    X_dummy = np.random.rand(num_dummy_samples, config.TIME_STEPS, config.FEATURE_DIM).astype(np.float32)
    y_dummy = np.random.randint(0, config.NUM_CLASSES, size=num_dummy_samples).astype(np.int64)

    print(f"Dummy features: {X_dummy.shape}, Dummy labels: {y_dummy.shape}")

    try:
        emotion_dataset = EmotionDataset(features=X_dummy, labels=y_dummy)
        print(f"Dataset created with {len(emotion_dataset)} samples.")

        if len(emotion_dataset) > 0:
            sample_feature, sample_label = emotion_dataset[0]
            print(f"Sample feature: {sample_feature.shape}, dtype: {sample_feature.dtype}")
            print(f"Sample label: {sample_label.item()}, dtype: {sample_label.dtype}")

            from torch.utils.data import DataLoader
            dummy_dataloader = DataLoader(emotion_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
            
            print(f"\nIterating DataLoader (batch size {config.BATCH_SIZE})...")
            for i, (batch_features, batch_labels) in enumerate(dummy_dataloader):
                if i == 0:
                    print(f"  Batch 1: Features: {batch_features.shape}, Labels: {batch_labels.shape}")
            print(f"Successfully iterated through {len(dummy_dataloader)} batches.")
        else:
            print("Dataset empty.")
        
        print("\nEmotionDataset example usage complete.")
    except Exception as e:
        print(f"Error in EmotionDataset example: {e}")
