# Surgical Instrumentation Segmentation Challenge using U-Net

This repository is designed for the Surgical Instrumentation Segmentation Challenge, utilizing the U-Net architecture.

## Features
- **Modifications**:
  - Replaced RMSprop with **AdamW** optimizer.
  - Improved prediction output with **semi-transparent masks** for better visualization.
  - Simultaneously generates **true masks** during prediction.
- **Validation Improvements**:
  - Added `validate.py` to compute and display **class-wise Dice scores**.
---

## Repository Structure
- **`unzipping.ipynb`**: Notebook for downloading and preparing the dataset.
- **`train.py`**: Script to train the U-Net model.
- **`predict.py`**: Script to generate predictions and masks.
- **`validate.py`**: Script to evaluate the model and compute Dice scores.
---

## Usage Instructions

### 1. Prepare the Dataset
- Use the `unzipping.ipynb` notebook to:
  - Download the dataset.
  - Sample the video frames at **1 FPS**.

### 2. Training the Model
- Train the U-Net model using `train.py`.
- Check the available hyperparameters:
  ```bash
  python train.py -h
  ```

### 3. Validate the Model
- Use `validate.py` to validate the trained model and compute class-wise Dice scores:
  ```bash
  python validate.py
  ```

### 4. Predict Masks
- Generate predictions with `predict.py`:
  ```bash
  python predict.py -h
  ```
- The predictions include:
  - **Better semi-transparent masks** for easier visualization.
  - **True masks** to compare with ground truth.