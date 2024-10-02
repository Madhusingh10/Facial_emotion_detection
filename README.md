This project is a Facial Emotion Recognition System that leverages Convolutional Neural Networks (CNNs) for detecting emotions in real time. Below is a detailed explanation of the project:

Tech Stack:
Frontend:

OpenCV: Captures real-time video from a webcam and detects faces using a Haar Cascade Classifier.
Matplotlib: Used for visualizing images during the model evaluation process.

Backend:

Keras (with TensorFlow backend): The deep learning framework for building and training the CNN model.
Pandas & NumPy: For data handling and manipulation, especially managing image paths and labels.
Application Workflow:

Data Handling:

The training and testing images are organized into directories based on emotion labels (e.g., 'angry', 'happy', etc.).
Image data is loaded using keras_preprocessing.image.load_img, converted to grayscale, and reshaped to (48, 48, 1) for the CNN input.

Model Architecture:

Built using Keras' Sequential API.
Multiple Conv2D layers with ReLU activations and MaxPooling2D for downsampling.
Dropout layers are added to prevent overfitting.
The final fully connected layers include Dense layers with ReLU activations, followed by a softmax output layer to predict one of the seven emotions (angry, disgust, fear, happy, neutral, sad, surprise).

Training:

The model is trained using the FER2013 dataset (converted into images) for 100 epochs.
Categorical cross-entropy is the loss function, and the Adam optimizer is used for training.

Real-time Emotion Detection:

The trained model is deployed with OpenCV to capture real-time video input.
The model detects faces from the webcam feed, preprocesses each face, and predicts the emotion using the trained CNN model.
The predicted emotion is displayed on the video feed above the detected face using OpenCV.

Model Used:
Convolutional Neural Network (CNN): The core of the system, trained to recognize seven distinct emotions from facial expressions. It includes layers of convolutions, pooling, dropout for regularization, and dense layers for classification.

Potential Applications:
Mental health assessments: Using emotion detection to monitor emotional states.
User experience analysis: Real-time emotion detection during software or product usage.
This system demonstrates a robust and real-time approach to facial emotion detection and can be expanded further for various applications.
![WhatsApp Image 2024-10-02 at 11 24 37 AM](https://github.com/user-attachments/assets/f2603eab-e152-46a0-aabf-3cc8f40f18a8)


