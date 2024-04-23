import tensorflow as tf
from matplotlib import pyplot as plt
import h5py
import numpy as np

# Function to display record from test dataset
def display_record(dataset, record_number):
    for i, (x, y) in enumerate(dataset):
        if i == record_number:
            print(f"Record {record_number}:")
            plt.imshow(x)
            print(f"Label: {decode_output(y.numpy())}")

            x = tf.expand_dims(x, axis=0)

            print("VGG16 Model Prediction: ", decode_output(predict("VGG16_Model_Best.h5", x)))
            print("ResNet Model Prediction: ", decode_output(predict("ResNet_Model_Best.h5", x)))
            
            break
    else:
        print("Record number exceeds dataset size.")

def predict(model_name, x):
            
            # Load the model
            model = tf.keras.models.load_model(model_name)
            
            # Predict using the model
            return model.predict(x)

def decode_output(x):
    x = np.argmax(x)
    
    if x == 0:
        return "Benign"
    else:
        return "Malignant"
    

def create_tfrecord(x_data, y_data, tfrecord_filename):
    assert len(x_data) == len(y_data), "Input and output data must have the same length."
    
    # Define feature description for TFRecord
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()]))
    
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    
    # Open TFRecord file writer
    with tf.io.TFRecordWriter(tfrecord_filename) as writer:
        for x, y in zip(x_data, y_data):
            # Ensure x has the correct shape
            assert x.shape == (96, 96, 3), "Input data must have shape (96, 96, 3)."
            # Convert y to one-hot encoding
            y_encoded = np.eye(2)[y]
            y_encoded = y_encoded.reshape(-1)  # Reshape y_encoded to a 1D array
            
            # Create example
            example = tf.train.Example(features=tf.train.Features(feature={
                'x': _bytes_feature(x),    # Convert image to bytes
                'y': _int64_feature(y_encoded.astype(np.int32).tolist())   # Store one-hot encoded label as int64 list
            }))
            # Serialize example and write to TFRecord
            writer.write(example.SerializeToString())

def parse_tfrecord_fn(example):
    feature_description = {
        'x': tf.io.FixedLenFeature([], tf.string),
        'y': tf.io.FixedLenFeature([2], tf.int64)  # Assuming y is one-hot encoded with shape (None, 2)
    }
    example = tf.io.parse_single_example(example, feature_description)
    
    # Decode x
    x = tf.io.decode_jpeg(example['x'], channels=3)
    x = tf.cast(x, tf.float32) / 255.  # Normalize pixel values
    
    # Decode y (assuming one-hot encoding)
    y = example['y']
    
    return x, y

def load_tfrecord_dataset(file_pattern):
    files = tf.data.Dataset.list_files(file_pattern)
    dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=4, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset
