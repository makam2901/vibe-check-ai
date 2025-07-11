# VibeCheck AI: Emotion Detection from Speech

VibeCheck AI is a deep learning-powered system designed to detect and classify human emotions directly from the sound of a voice. This project explores the nuances of human-computer interaction by building a robust model capable of understanding the emotional layer in speech.

[Medium Article]([url](https://medium.com/@manikeshmakam/can-ai-feel-your-vibe-introducing-vibecheck-ai-your-emotion-detection-system-2e248ac5cf7c))

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Methodology](#methodology)
  - [1. Data Source & Preparation](#1-data-source--preparation)
  - [2. Feature Extraction](#2-feature-extraction)
  - [3. Data Augmentation](#3-data-augmentation)
  - [4. Model Architecture](#4-model-architecture)
  - [5. Hyperparameter Tuning & Optimization](#5-hyperparameter-tuning--optimization)
- [Results](#results)
- [How to Use](#how-to-use)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
- [Dependencies](#dependencies)
- [Contributors](#contributors)

## Project Overview

Human emotions are incredibly complex and are often woven into the subtle nuances of our speech—the tone, pitch, and rhythm. Capturing these vocal cues is key to building more empathetic and intelligent AI. VibeCheck AI was developed to tackle this challenge by creating a system that can accurately identify a range of emotions from audio recordings.

Potential applications include:
- **Mental Health Monitoring:** Tools that can help identify emotional distress.
- **Smarter Customer Service:** Automated systems that respond with more understanding.
- **Engaging Entertainment:** Games and interactive experiences that adapt to a user's emotional state.

## Features

- Classifies speech into five core emotions: **happy, sad, angry, neutral, and excited**.
- Processes raw audio data into a machine-learning-ready format.
- Utilizes a Convolutional Neural Network (CNN) built with PyTorch.
- Employs automated hyperparameter tuning with Optuna for performance optimization.
- Includes data augmentation to enhance model robustness.

## Methodology

The project follows a standard machine learning workflow from data collection to model evaluation.

### 1. Data Source & Preparation

- **Dataset:** The project uses the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset.
- **Filtering:** The dataset is filtered to include only speech files corresponding to the five target emotions. The code includes a script to automatically perform this filtering.
- **Standardization:** All audio files are converted to a uniform format (WAV, 16 kHz sampling rate) and padded or truncated to a consistent length of 3 seconds.

### 2. Feature Extraction

To enable the model to learn from the audio, raw waveforms are transformed into a visual representation.
- **Log-Mel Spectrograms:** We use the `Librosa` library to generate Log-Mel Spectrograms from the audio files. This feature acts as a "visual fingerprint" of the sound, highlighting different frequencies over time, which is highly effective for capturing emotional cues.
- **Normalization:** Each spectrogram is standardized on an instance-wise basis (mean of 0, standard deviation of 1) to ensure consistency across the dataset.

### 3. Data Augmentation

To prevent the model from overfitting and to handle potential class imbalances, we augment the training data using the `audiomentations` library. The following transformations are applied randomly:
- **Time Stretching:** Speeds up or slows down the audio.
- **Pitch Shifting:** Raises or lowers the pitch.
- **Shifting:** Shifts the audio in time.

### 4. Model Architecture

After experimenting with both CNN and CRNN (Convolutional Recurrent Neural Network) architectures, the CNN was selected as the top performer.

The `CNNEmotion` model architecture consists of:
- **Two Convolutional Blocks:** Each block contains a `Conv2d` layer, `BatchNorm2d`, `ReLU` activation, and `MaxPool2d`. These blocks are responsible for detecting local patterns (like edges and textures) in the spectrograms.
- **A Flattening Layer:** Converts the 2D feature maps into a 1D vector.
- **A Fully Connected Block:** Contains `Linear` layers, `ReLU` activation, and `Dropout` for regularization before producing the final classification.

### 5. Hyperparameter Tuning & Optimization

- **Optuna:** We used the Optuna framework to automate the process of finding the best hyperparameters for our model. This involved systematically exploring various combinations of learning rates, dropout probabilities, and CNN filter counts.
- **Median Pruner:** To make the optimization process more efficient, a `MedianPruner` was used to halt unpromising trials early.

## Results

The final optimized CNN model was trained on a combination of the training and validation sets and then evaluated on a hold-out test set.
- **Test Accuracy:** **72.83%**
- **Test Loss:** 0.7134

The classification report shows strong performance, especially for certain emotions:
- **High Precision:** The model is very accurate when predicting 'angry' (97%) and 'neutral' (100%).
- **Good Recall:** The model is effective at identifying most instances of 'sad' (84%) and 'angry' (74%).

![Classification Report](https://i.imgur.com/W2h5h3f.png)

## How to Use

Follow these steps to set up and run the project locally.

### Prerequisites
- Python 3.7+
- `pip` package manager

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/VibeCheckAI.git](https://github.com/your-username/VibeCheckAI.git)
    cd VibeCheckAI
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(You will need to create a `requirements.txt` file. See the [Dependencies](#dependencies) section below)*

4.  **Download the Dataset:**
    - Download the RAVDESS dataset from [here](https://zenodo.org/record/1188976).
    - Extract the `Audio_Speech_Actors_01-24` folder and place it inside a `data/` directory in the project root.

### Usage

1.  **Prepare the Data:**
    Run the data preparation script from the notebook to filter the RAVDESS dataset and create the structured `filtered_ravdess` directory.

2.  **Train the Model:**
    You can run the entire Jupyter Notebook (`Vibecheckai.html`) to train the model from scratch, which includes the hyperparameter tuning process.

3.  **Predict with the Pre-trained Model:**
    The repository includes a pre-trained model (`best_model/emotion_model.pth`). You can use the following code snippet to load the model and predict the emotion of a new audio file.

    ```python
    import torch
    from model_architecture import CNNEmotion # Assuming you save the model class in this file
    from feature_extraction import preprocess_live_audio # Assuming you save the function here

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNEmotion(num_classes=5, input_shape=(1, 64, 94)) # Use the correct input_shape
    model.load_state_dict(torch.load("best_model/emotion_model.pth", map_location=device))
    model.to(device).eval()

    # Preprocess a new audio file and predict
    audio_path = "path/to/your/audio.wav"
    input_tensor = preprocess_live_audio(audio_path, device)

    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = torch.argmax(output, dim=1).item()

    emotions = ['angry', 'excited', 'happy', 'neutral', 'sad'] # Ensure this order matches your LabelEncoder
    print(f"Predicted Emotion: {emotions[pred_idx]}")
    ```

## Dependencies
Create a `requirements.txt` file with the following contents:
```
numpy
pandas
librosa
matplotlib
seaborn
torch
audiomentations
scikit-learn
optuna
tqdm
```

## Contributors
- **Makam, Manikesh**
- **Sikha, Sucheth**
- **Khot, Nirant**
