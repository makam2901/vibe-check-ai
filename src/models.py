# models.py
"""
Defines the neural network architectures for emotion detection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import config

# --- 1. CNN Model (CNNEmotion) ---
class CNNEmotion(nn.Module):
    """Simple CNN for emotion classification. Input: (B, 1, TIME_STEPS, FEATURE_DIM)"""
    def __init__(self, num_classes=config.NUM_CLASSES, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(config.CNN_INPUT_CHANNELS, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flattened_size = 32 * (config.TIME_STEPS // 2) * (config.FEATURE_DIM // 2)
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(128, num_classes)

    def forward(self, x):
        if x.ndim == 3: x = x.unsqueeze(1)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.out(x)
        return x

# --- 2. CnnLstmFusion Model ---
class CnnLstmFusion(nn.Module):
    """Hybrid CNN-LSTM model. Fuses global CNN and LSTM features."""
    def __init__(self, num_classes=config.NUM_CLASSES,
                 cnn_dropout_rate=0.1, lstm_hidden_size=128,
                 lstm_num_layers=2, lstm_dropout_rate=0.3,
                 fusion_dropout_rate=0.4, lstm_bidirectional=True):
        super().__init__()
        # CNN Branch
        self.conv1 = nn.Conv2d(config.CNN_INPUT_CHANNELS, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32); self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64); self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.cnn_flattened_size = 64 * 4 * 4
        self.cnn_fc = nn.Linear(self.cnn_flattened_size, 128)
        self.cnn_dropout = nn.Dropout(cnn_dropout_rate)
        # LSTM Branch
        self.lstm = nn.LSTM(
            input_size=config.FEATURE_DIM, hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers, batch_first=True,
            dropout=lstm_dropout_rate if lstm_num_layers > 1 else 0,
            bidirectional=lstm_bidirectional
        )
        self.lstm_fc_input_size = lstm_hidden_size * 2 if lstm_bidirectional else lstm_hidden_size
        self.lstm_fc = nn.Linear(self.lstm_fc_input_size, 128)
        self.lstm_dropout = nn.Dropout(lstm_dropout_rate)
        # Fusion and Classifier
        self.fusion_fc1 = nn.Linear(128 + 128, 128)
        self.fusion_dropout = nn.Dropout(fusion_dropout_rate)
        self.output_fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # CNN Branch
        x_cnn = x.unsqueeze(1) if x.ndim == 3 else x
        x_cnn = self.pool1(F.relu(self.bn1(self.conv1(x_cnn))))
        x_cnn = self.pool2(F.relu(self.bn2(self.conv2(x_cnn))))
        x_cnn = self.adaptive_pool(x_cnn)
        x_cnn = x_cnn.view(x_cnn.size(0), -1)
        x_cnn = F.relu(self.cnn_fc(x_cnn)); x_cnn = self.cnn_dropout(x_cnn)
        # LSTM Branch
        x_lstm, _ = self.lstm(x)
        x_lstm = x_lstm[:, -1, :]
        x_lstm = F.relu(self.lstm_fc(x_lstm)); x_lstm = self.lstm_dropout(x_lstm)
        # Fusion
        fused_features = torch.cat((x_cnn, x_lstm), dim=1)
        output = F.relu(self.fusion_fc1(fused_features))
        output = self.fusion_dropout(output); output = self.output_fc(output)
        return output

# --- ADD NEW SEQUENTIAL CRNN MODEL HERE ---
# --- 3. SequentialCRNN Model ---
class SequentialCRNN(nn.Module):
    """
    Sequential Convolutional Recurrent Neural Network.
    CNN extracts features per time step, then LSTM processes the sequence.
    Input shape: (batch_size, TIME_STEPS, FEATURE_DIM)
    """
    def __init__(self, num_classes=config.NUM_CLASSES,
                 cnn_out_channels=64,  # Output channels from the last CNN layer
                 lstm_hidden_size=128, lstm_num_layers=2,
                 cnn_dropout_rate=0.1, lstm_dropout_rate=0.3,
                 classifier_dropout_rate=0.4, lstm_bidirectional=True):
        super().__init__()

        # CNN Backbone
        # Input: (B, 1, TIME_STEPS, FEATURE_DIM) e.g., (B, 1, 94, 92)
        self.conv1 = nn.Conv2d(config.CNN_INPUT_CHANNELS, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # Pool along feature dim to reduce it, keep time steps mostly intact
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)) # (B, 32, 94, 46)

        self.conv2 = nn.Conv2d(32, cnn_out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(cnn_out_channels)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)) # (B, cnn_out, 94, 23)
        
        self.cnn_dropout = nn.Dropout2d(cnn_dropout_rate) # Spatial dropout

        # Calculate the feature dimension after CNN processing (before LSTM)
        # After pool2, feature_dim becomes config.FEATURE_DIM // 4
        # The number of time steps (TIME_STEPS) remains largely the same (94)
        # So, input to LSTM will be (B, TIME_STEPS, cnn_out_channels * (FEATURE_DIM // 4))
        self.lstm_input_dim = cnn_out_channels * (config.FEATURE_DIM // 4)

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout_rate if lstm_num_layers > 1 else 0,
            bidirectional=lstm_bidirectional
        )

        # Classifier
        self.classifier_dropout = nn.Dropout(classifier_dropout_rate)
        lstm_output_dim = lstm_hidden_size * 2 if lstm_bidirectional else lstm_hidden_size
        self.fc_out = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, x):
        # x input shape: (B, TIME_STEPS, FEATURE_DIM)
        
        # CNN Part
        # Reshape for Conv2D: (B, Channels, Time, Freq)
        x_cnn = x.unsqueeze(1) # (B, 1, TIME_STEPS, FEATURE_DIM)
        
        x_cnn = self.pool1(F.relu(self.bn1(self.conv1(x_cnn))))
        x_cnn = self.cnn_dropout(x_cnn) # Apply dropout after activation/pooling
        x_cnn = self.pool2(F.relu(self.bn2(self.conv2(x_cnn))))
        x_cnn = self.cnn_dropout(x_cnn)
        # Output of CNN: (B, cnn_out_channels, TIME_STEPS, FEATURE_DIM_reduced)
        # e.g., (B, 64, 94, 23)

        # Prepare for LSTM: (B, TIME_STEPS, new_feature_dim)
        # We want TIME_STEPS to be the sequence length for LSTM.
        # So, permute and reshape:
        batch_size, channels, time_steps_cnn, features_cnn = x_cnn.shape
        x_lstm_in = x_cnn.permute(0, 2, 1, 3) # (B, TIME_STEPS, cnn_out_channels, FEATURE_DIM_reduced)
        x_lstm_in = x_lstm_in.reshape(batch_size, time_steps_cnn, channels * features_cnn)
        # Now x_lstm_in is (B, 94, 64 * 23) which matches (B, TIME_STEPS, self.lstm_input_dim)

        # LSTM Part
        x_lstm_out, _ = self.lstm(x_lstm_in)
        
        # Use the output of the last time step for classification
        x_lstm_out = x_lstm_out[:, -1, :] 
        
        # Classifier Part
        output = self.classifier_dropout(x_lstm_out)
        output = self.fc_out(output)
        
        return output

if __name__ == '__main__':
    print("Testing model instantiations and forward passes...")
    device = config.DEVICE
    bs = 4 # batch_size

    # Test CNNEmotion
    print("\nTesting CNNEmotion...")
    # ... (CNNEmotion test code as before) ...
    cnn_in = torch.randn(bs, config.TIME_STEPS, config.FEATURE_DIM).to(device); cnn_model = models.CNNEmotion().to(device)
    try: cnn_out = cnn_model(cnn_in); assert cnn_out.shape == (bs, config.NUM_CLASSES); print("CNN Test PASSED.")
    except Exception as e: print(f"CNN Test FAILED: {e}")

    # Test CnnLstmFusion
    print("\nTesting CnnLstmFusion...")
    # ... (CnnLstmFusion test code as before) ...
    fusion_in = torch.randn(bs, config.TIME_STEPS, config.FEATURE_DIM).to(device); fusion_model = models.CnnLstmFusion().to(device)
    try: fusion_out = fusion_model(fusion_in); assert fusion_out.shape == (bs, config.NUM_CLASSES); print("CnnLstmFusion Test PASSED.")
    except Exception as e: print(f"CnnLstmFusion Test FAILED: {e}")

    # --- Test SequentialCRNN ---
    print("\nTesting SequentialCRNN...")
    crnn_in = torch.randn(bs, config.TIME_STEPS, config.FEATURE_DIM).to(device)
    crnn_model = SequentialCRNN(
        num_classes=config.NUM_CLASSES,
        cnn_out_channels=32, # Example
        lstm_hidden_size=64, # Example
        lstm_num_layers=1,   # Example
        cnn_dropout_rate=0.1,
        lstm_dropout_rate=0.2,
        classifier_dropout_rate=0.3
    ).to(device)
    try:
        crnn_out = crnn_model(crnn_in)
        print(f"SequentialCRNN Input shape: {crnn_in.shape}")
        print(f"SequentialCRNN Output shape: {crnn_out.shape} (Expected: ({bs}, {config.NUM_CLASSES}))")
        assert crnn_out.shape == (bs, config.NUM_CLASSES), "SequentialCRNN output shape mismatch"
        print("SequentialCRNN test PASSED.")
    except Exception as e:
        print(f"SequentialCRNN test FAILED: {e}")

