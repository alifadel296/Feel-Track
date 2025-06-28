# Emotion Recognition System

A real-time emotion recognition pipeline using MediaPipe for landmark extraction and a hybrid CNN–BiLSTM–Attention model built with TensorFlow/Keras.

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [How It Words](#how-it-works)
* [Data Collection](#data-collection)
* [Training](#training)
* [Real-Time Detection](#real-time-detection)
* [Usage Examples](#usage-examples)


## Overview

This project demonstrates an end-to-end emotion recognition system:

1. **Data Collection**: Capture face and pose landmarks via MediaPipe and save to CSV.
2. **Preprocessing & Augmentation**: Normalize and augment landmark data.
3. **Model Training**: Train a hybrid CNN–BiLSTM–Attention network to classify emotions.
4. **Real-Time Inference**: Load the trained model to perform emotion detection from live webcam feed.

## Features

* Flexible CSV-based data logging of multiple emotions
* Augmentation of training data for robustness
* Hybrid CNN + BiLSTM + Attention architecture
* Real-time webcam inference with optional holistic landmark overlay
* Class-weighted training to handle imbalanced datasets

## How It works

### Model:

- `model.py`: Defines a hybrid CNN + BiLSTM + Attention model. CNN layers extract spatial features from sequences of landmarks, BiLSTM captures temporal dependencies, and the attention layer focuses on salient features. Outputs emotion classes using softmax activation.

### Data Processing:

- `data_collection.py`: Captures pose and facial landmarks using MediaPipe and saves them to a CSV file, each labeled with the target emotion.

- `preprocess_data.py`: Handles label encoding and augmentation. Ensures balanced data and compatible shapes for the model.

- `utils.py`: Contains utility function.

### Training & Evaluation: 

- `train.py`: Compiles and trains the model using categorical cross-entropy and class weights to handle imbalances. Includes callbacks for early stopping and learning rate reduction.

### Real‑Time Inferencing:

- `detect.py`: Loads the trained model and uses live webcam input to perform real-time emotion classification. ***Optionally draws holistic landmarks on the frame***.



## Data Collection


Use `data_collection.py` to record landmark data for your emotions of interest:

```bash
python data_collection.py \
  --emotions_list EMOTION_LIST \
  --path_of_csv DATA_PATH
  --pause NUM_OF_SECONDS
````

Each session captures pose and face landmarks and appends rows labeled by emotion to the CSV.
### Note:
- Press `q` to stop capturing for the current emotion.
- After completing the current emotion's capture session, the webcam will pause for X seconds (default 2 seconds) before proceeding to the next.  


## Training

Train the model via `train.py`:

```bash
python train.py \
  --root DATA_PATH
  --epochs NUM_EPOCHS \
  --batch_size BATCH_SIZE \
  --checkpoint_dir CHECKPOINT_DIR
  --debug
```

* `--debug` displays loss/accuracy plots and evaluation metrics after training.

## Real-Time Detection

Run `detect.py` script to see live predictions:

```bash
python detect.py --model_path TRAINED_MODEL_PATH
 --with_holistic
```

* `--with_holistic` overlays facial, pose, and hand landmarks on the video feed.


## Usage Examples

1. **Collect data**:

   ```bash
   python data_collection.py -el happy sad angry -ps data/emotions.csv
   ```

2. **Train model**:

   ```bash
   python train.py --root data/emotions.csv --epochs 30 --batch_size 8
   ```

3. **Run detection**:

   ```bash
   python detect.py -mp ./emotion_model.h5 -wh
   ```


