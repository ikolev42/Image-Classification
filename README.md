# Image-Classification
CIFAR-10 Image Classification with Convolutional Neural Networks (CNN)

This project showcases the implementation of a Convolutional Neural Network (CNN) for classifying images from the CIFAR-10 dataset using TensorFlow and Keras. The CIFAR-10 dataset is a popular benchmark dataset in the field of computer vision, containing 60,000 32x32 color images categorized into 10 classes, including airplanes, cars, birds, cats, and more.

Features
Data Loading and Preprocessing:

Automatically loads the CIFAR-10 dataset using TensorFlow/Keras.
Normalizes image pixel values to the range [0, 1] for faster training.
Converts class labels into one-hot encoded format for categorical classification.
Model Architecture:

Built using a Sequential API in Keras.
Includes three convolutional layers (Conv2D) for feature extraction, each followed by a pooling layer (MaxPooling2D) to reduce dimensionality.
Flatten layer to convert 2D feature maps into a 1D vector.
Fully connected dense layers for classification, with a final softmax activation to output probabilities for 10 classes.
Model Training:

Compiles the model using the Adam optimizer and categorical cross-entropy loss function.
Trains the model on the CIFAR-10 training dataset with validation on the test set.
Visualizes training and validation accuracy over epochs.
Model Evaluation:

Evaluates the trained model's accuracy on the test dataset.
Predicts categories for sample images from the test set.
Visualization:

Plots training and validation accuracy over time.
Displays test images with predicted and actual labels for visual inspection.

Installation and Usage
Clone the repository:

bash
git clone https://github.com/your-username/cifar10-cnn-classification.git
cd cifar10-cnn-classification
Install the required Python libraries:

bash
pip install tensorflow matplotlib
Run the script:

bash
python cifar10_cnn.py
Results
The CNN achieves decent accuracy on the CIFAR-10 dataset, depending on the number of epochs and hyperparameters used.
The training and validation accuracy trends are visualized using Matplotlib.
Sample test images are displayed with their predicted and actual labels for easy evaluation.
Dataset
The CIFAR-10 dataset contains:

60,000 images in total: 50,000 for training and 10,000 for testing.
10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.
For more information about the CIFAR-10 dataset, visit CIFAR-10 Dataset.

Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.
