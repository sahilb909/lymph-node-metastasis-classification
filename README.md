# Lymph Node Metastasis Classification using Deep Learning
This project focuses on classifying histopathological images from the PatchCamelyon (PCam) dataset into two categories: positive and negative. Two deep learning models, VGG16 (Transfer Learning) and ResNet, are employed for this task.

# Histopathological Image Classification using Deep Learning


## Project Structure

- `training.py`: Contains code for training the models.
- `evaluation.py`: Handles evaluation metrics like ROC-AUC curve.
- `data_exploration.py`: Provides functionality to test the model on a selected record.
- `helper_functions.py`: Contains utility functions required throughout the project.

## Dataset

The PatchCamelyon (PCam) dataset consists of histopathological images that are used to determine whether lymph nodes are positive or negative for metastatic cancer. The dataset is divided into training, testing, and validation sets.

## Requirements
- Python 3.x
- TensorFlow 2.x
- Keras
- h5py
- numpy
- matplotlib

## Data Preparation

The dataset is initially stored in HDF5 format and then converted to TFRecord format for efficient training. The `create_tfrecord` and `load_data_from_hdf5` functions in `helper_functions.py` handle this conversion.

## Training

The models are trained using the training dataset and the best weights are saved using callbacks. Early stopping is used to prevent overfitting.

## Evaluation

The `evaluation.py` script calculates the ROC-AUC score to evaluate the performance of the trained models on the test dataset.

## Data Exploration

The `data_exploration.py` script allows for testing the model on a selected record from the test dataset.
