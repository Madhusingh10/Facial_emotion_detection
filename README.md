This project is a Facial Emotion Recognition System that uses Convolutional Neural Networks (CNNs) for real-time emotion detection.
Tech Stack:
Frontend:
OpenCV: For capturing real-time video from a webcam and detecting faces using a Haar Cascade Classifier.
Matplotlib: Used for visualizing images in some steps of the model evaluation process.
Backend:
Keras (with TensorFlow backend): Deep learning framework used for building and training the CNN model.
Pandas & NumPy: For data handling and manipulation, particularly managing image paths and labels.
Application Workflow:
Data Handling:
The training and testing images are organized into directories based on the emotion labels (e.g., 'angry', 'happy', etc.).
The image data is loaded using keras_preprocessing.image.load_img, converted to grayscale, and reshaped to (48, 48, 1) for the CNN input.
Model Architecture:
Built using Keras' Sequential API.
Multiple Conv2D layers with ReLU activations and MaxPooling2D for downsampling.
Dropout layers are added to prevent overfitting.
The final fully connected layers consist of Dense layers with ReLU, followed by the output layer with a softmax activation to predict one of the seven emotions (angry, disgust, fear, happy, neutral, sad, surprise).
Training:
The model is trained using the FER2013 dataset (https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset), now stored as images, for 100 epochs.
The categorical cross-entropy loss function is used, and the optimizer is Adam.
Real-time Emotion Detection:

The trained model is deployed with OpenCV to capture real-time video input.
The model detects faces from the webcam feed, preprocesses each face, and predicts the emotion using the trained CNN model.
The predicted emotion is displayed on the video feed above the detected face using OpenCV.
Model Used:
Convolutional Neural Network (CNN): The core of the system, trained to recognize seven distinct emotions from facial expressions. It includes layers of convolutions, pooling, dropout for regularization, and dense layers for classification.

This system can be applied in various real-world scenarios, such as:

Mental health assessments using emotion detection for monitoring emotional states.
User experience analysis by detecting emotions in real-time during software or product usage.
