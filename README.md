# Aorta Bifurcation Localization in CT Scans

# Overview
This repository contains code for localizing the aorta bifurcation point in CT scans using a 2.5D UNet architecture. The aorta bifurcation point is a crucial landmark in medical imaging. This project aims to automate the localization process to assist medical professionals in their diagnostic and treatment procedures.

# Requirements 
* Python 3.x
* TensorFlow
* Keras
* numpy
* pandas
* matplotlib
* scikit-image
* scipy
* ITK-SNAP (Optional for data annotation)
* MATLAB (Optional for additional analysis or visualization)

# Dataset
The dataset used for training and evaluation should consist of CT scans with corresponding annotations marking the aorta bifurcation point. Ensure that the dataset is appropriately preprocessed and formatted before training.

# Training
To train the model, follow these steps:
* Prepare Dataset: Organize your dataset with CT scans and corresponding annotations.
* Preprocessing: Preprocess the data as required, including resizing, normalization, and augmentation.
* Configuration: Set the necessary parameters in the configuration file, such as batch size, learning rate, and number of epochs.
* Training: Run the training script, providing the path to your dataset and configuration file.

# Evaluation
After training, evaluate the model's performance using the evaluation script. This involves running inference on a separate test set and calculating relevant metrics such as Dice score and distances between centroids.

# Inference
Use the trained model for inference on new CT scans to localize the aorta bifurcation point. Provide the path to the input CT scan, and the model will output the predicted coordinates of the bifurcation point.

# Demo
Here are screenshots of demo training and inference:

<img width="477" alt="image" src="https://github.com/toan-ly/Aorta-Bifurcation-Localizer/assets/104543062/bac6b5b2-2c9d-4966-8347-492678b9b70b">

<img width="455" alt="image" src="https://github.com/toan-ly/Aorta-Bifurcation-Localizer/assets/104543062/de165901-cfb3-4c74-969e-26be9b09ad7f">

<img width="951" alt="image" src="https://github.com/toan-ly/Aorta-Bifurcation-Localizer/assets/104543062/df290374-8b80-4e29-a0c3-5c3c9f25ed08">
