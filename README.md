♟ Chess Piece Recognition — Deep Learning Project
Using Convolutional Neural Networks (CNNs) to Classify Chess Pieces from Images

Authors:

El Moussaoui Noura

Es-Selmy Niama

Year: 2025
Course: AI / Machine Learning TP

📌 Overview

This project develops a Convolutional Neural Network (CNN) capable of recognizing 12 chess piece categories, distinguishing both piece type (King, Queen, Rook, Bishop, Knight, Pawn) and piece color (White/Black).

The notebook includes:

Full data loading and preprocessing

Grayscale conversion and resizing

CNN architecture development

Training, validation, and testing

Visualization of predictions

Error analysis (misclassified samples)

Final model evaluation

🗂 Dataset

The dataset contains 12 classes, structured in three folders:

Chess_pieces/
 ├── train/
 ├── valid/
 └── test/


Each class corresponds to one chess piece+color, for example:

wr – White Rook

bq – Black Queen

wp – White Pawn

... (total 12 classes)

Images were standardized to 75×75 grayscale.

⚙ Project Pipeline
1. Data Setup

Connect Google Drive

Load dataset from folder structure

Explore dataset: number of samples per class

Visualize images before preprocessing

2. Preprocessing

Convert RGB → Grayscale

Normalize pixel values to [0, 1]

Resize to 75×75

Batch & shuffle using tf.data

3. Model Architecture

CNN built with TensorFlow/Keras:

Rescaling(1/255)

3× Convolution + ReLU layers

MaxPooling layers

Dropout for regularization

Fully connected Dense layers

Softmax output (12 classes)

The model reaches ≈ 99.5% accuracy on the test set.

📊 Results & Evaluation
✔ Training & Validation Curves

The notebook includes plots for:

Accuracy vs Epochs

Loss vs Epochs

✔ Test Set Evaluation

Final performance on unseen images:

Test Accuracy: ~99.5%

Test Loss: extremely low

✔ Prediction Visualization

The notebook displays:

Correct predictions (in blue)

Incorrect predictions (in red)

Probability distribution bars for all 12 classes

A grid of many test images with predicted labels

✔ Error Analysis

We automatically extract and plot all misclassified images, helping identify possible weaknesses or ambiguous samples.

📥 Using the Trained Model

You can load a single image from the test set and perform:

Preprocessing

Converting to batch format

Predicting class probabilities

Visualizing prediction confidence

The notebook includes a reusable function for single-image inference.

💾 Saving & Loading the Model

The model is saved in .keras format:

model.save("chess_cnn_model.keras")


To load later:

from tensorflow.keras.models import load_model
model = load_model("chess_cnn_model.keras")


This avoids re-training when re-opening the notebook.

📖 Notebook Contents

The full project notebook is available here:
👉 Chess_Piece_Recognition.ipynb

You can open it directly in Google Colab:

(Replace USERNAME/REPO accordingly)

🚀 How to Run the Project

Open the notebook in Google Colab

Connect to Google Drive

Run all steps (or load the saved model for faster inference)

Test predictions or upload your own chess piece images

📌 Future Improvements

Possible extensions:

Add a lightweight mobile-friendly model (TensorFlow Lite)

Support for color (RGB) images instead of grayscale

Object detection on full chessboard images

Augment dataset for robustness

📜 License

This project is for academic use (2025 AI/ML TP).
All code is free to study or reuse for educational purposes.
