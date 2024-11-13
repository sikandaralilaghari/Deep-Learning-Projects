# Simple Convolutional Neural Network (CNN) for Fashion MNIST
This projects implements a simple Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images from the Fashion MNIST dataset contains grayscale images of 10 different clothing items, and this CNN model aims to classify each image correctly.

# Dataset
The Fashion MNIST dataset consists of 70,000 grayscale images in 10 classes. Each image is 28 x 28 pixels. The dataset split into:
60000 training images
10000 test images 

The classes include items such as T-shirts, trousers, pullovers, dresses, and coats. 
#Model Architecture
1. Input Layer: 28 X 28 grayscale images
2. Convolutional Layer 1: 32 filters of size 3 X 3, Relu Activation
3. Max pooling layer 1: 2 X 2 pool size
4. Convolutional Layer 2: 64 filters of size 3 X 3, RelU activation
5. Max pooling layer 2: 2 X 2 pool size.
6. Convolutional layer 3: 64 filters of size 3X3, RelU activation
7. Flatten Layer: Converts the 2D matrix into a 1D vector.
8. Fully connected Layer: 64 units, ReLU activation
9. Output Layer: 10 units with softmax activation (one for each class)


# Requirements
Python 3.x 
TensorFlow 2.x
Keras
Matplotlib (for plotting accuracy and loss)

You can install dependencies by running:
pip install tensorflow keras matplotlib


Code Explanation:
The code in SimpleCNN.ipynb

1. Data Loading and Preprocessing
   1. Loads the Fashion MNIST dataset from tensorflow.keras.datasets
   2. Normalize the image pixel values to a range of [0,1]
   3. Reshapes the images to add a channel dimension
2. Model definition:
   1. A Sequential model is created with 3 convolutional layers followed by max pooling.
   2. The output from the convolutional layer is flattened and fed into a fully connected layer with RelU activation.
   3. The final layer uses softmax activation for multi-class classification
3. Compilation: The model is compiled using the Adam optimizer and sparse categorical cross-entropy loss.
4. Training: The model is trained for 10 epochs, with accuracy calculated on both training and test sets.
5. Evaluation and Visualization:
   1. The test accuracy and loss are displayed.
   2. A plot of accuracy over epochs is generated to visualize training progress.


# Results
The CNN model should achieve an accuracy of approximately 90% on the test set, demonstrating its ability to learn and generalize from the Fashion MNIST dataset.

![image](https://github.com/user-attachments/assets/dc51efa5-528e-45b4-b390-3f4c1b7715b5)


