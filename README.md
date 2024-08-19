

# CIFAR-10 Classification using Linear Models, ANN, and CNN

This project showcases the classification of the CIFAR-10 dataset using three different neural network architectures: a Linear Model, an Artificial Neural Network (ANN), and a Convolutional Neural Network (CNN). The primary objective is to demonstrate the effectiveness of these models on a standard image classification task and to understand the underlying scientific concepts.

## Table of Contents
1. [Introduction](#introduction)
2. [Loading the CIFAR-10 Dataset](#loading-the-cifar-10-dataset)
3. [Scientific Concepts and Models](#scientific-concepts-and-models)
   - Linear Model
   - Artificial Neural Network (ANN)
   - Convolutional Neural Network (CNN)
4. [Model Performance-Accuracy](#model-performance)
5. [Conclusion](#conclusion)

## Introduction

The CIFAR-10 dataset is a well-known benchmark in machine learning and computer vision. It consists of 60,000 32x32 color images across 10 different classes, with 50,000 images designated for training and 10,000 for testing. The classes cover a broad range of objects such as airplanes, cars, birds, and more. The classification task involves assigning one of the 10 classes to each image.

## Loading the CIFAR-10 Dataset

The CIFAR-10 dataset was loaded and preprocessed, which included normalizing the image data (scaling pixel values to a range between 0 and 1) and converting class labels into a one-hot encoded format. This preprocessing step is crucial for preparing the data for neural network models, ensuring that the models can efficiently learn from the data.

## Scientific Concepts and Models

### 1. Linear Model
- **Concept**: A linear model is the simplest form of a neural network, consisting of a single layer with no hidden units. It makes predictions based on a weighted sum of the input features, followed by a softmax activation function that converts the output into probabilities across the 10 classes.
- **Use Case**: In this project, the linear model serves as a baseline to understand the complexity of the CIFAR-10 dataset. Due to its simplicity, it is expected to perform poorly on this task, providing a benchmark for evaluating more complex models.

### 2. Artificial Neural Network (ANN)
- **Concept**: An Artificial Neural Network (ANN) consists of multiple layers of neurons, including one or more hidden layers. Each neuron in the hidden layers applies a non-linear activation function (e.g., ReLU) to the input data, enabling the model to learn complex patterns and representations.
- **Architecture**: The ANN used in this project includes a few dense (fully connected) layers, with each layer applying a ReLU activation. The final layer uses a softmax activation function to output probabilities for each class.
- **Use Case**: The ANN is designed to capture more complex relationships in the CIFAR-10 dataset compared to the linear model, leading to improved performance.

### 3. Convolutional Neural Network (CNN)
- **Concept**: Convolutional Neural Networks (CNNs) are specifically designed for processing grid-like data such as images. CNNs use convolutional layers that apply filters to local receptive fields of the input image, capturing spatial hierarchies and patterns (e.g., edges, textures). This makes them highly effective for image classification tasks.
- **Architecture**: The CNN implemented in this project consists of multiple convolutional layers, each followed by a pooling layer to reduce spatial dimensions. Dropout layers are also used to prevent overfitting. The final layers are fully connected, leading to a softmax output.
- **Use Case**: The CNN is expected to outperform both the linear model and the ANN due to its ability to effectively capture and utilize the spatial structure of image data.

## Model Performance

- **Linear Model**: As expected, the linear model performed the worst among the three, with limited accuracy. This is due to its inability to capture non-linear relationships and spatial hierarchies in the image data. 


    $Accuracy: 48.00$%
- **Artificial Neural Network (ANN)**: The ANN showed improved performance over the linear model, demonstrating the benefits of using multiple layers and non-linear activations. However, its performance was still not optimal for a complex image classification task like CIFAR-10.

    $Accuracy: 52.78$%
- **Convolutional Neural Network (CNN)**: The CNN significantly outperformed both the linear model and the ANN. Its ability to learn and utilize spatial features of the images made it the most effective model for this task, achieving the highest accuracy on the test set.

    $Accuracy: 77.79$%

    $Top-3 Accuracy: 94.99$%
## Conclusion

This project illustrates the progression from simple to more complex models for image classification. The results highlight the importance of model architecture in machine learning, particularly the effectiveness of CNNs for tasks involving image data. The CNN's superior performance underscores its suitability for handling the spatial nature of images, making it a powerful tool for similar classification tasks.
