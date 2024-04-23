from helper_functions import create_tfrecord
import h5py

# Define file paths
train_x_path = 'train/camelyonpatch_level_2_split_train_x.h5'
train_y_path = 'train/camelyonpatch_level_2_split_train_y.h5'

test_x_path = 'test/camelyonpatch_level_2_split_test_x.h5'
test_y_path = 'test/camelyonpatch_level_2_split_test_y.h5'

val_x_path = 'val/camelyonpatch_level_2_split_valid_x.h5'
val_y_path = 'val/camelyonpatch_level_2_split_valid_y.h5'

# Function to load data samples from HDF5 files
def load_data_from_hdf5(x_path, y_path):
    with h5py.File(x_path, 'r') as x_file, h5py.File(y_path, 'r') as y_file:
        x_data = x_file['x'][:]
        y_data = y_file['y'][:]
    return x_data, y_data

# Load data samples
x_train, y_train = load_data_from_hdf5(train_x_path, train_y_path)
x_test, y_test = load_data_from_hdf5(test_x_path, test_y_path)
x_val, y_val = load_data_from_hdf5(val_x_path, val_y_path)

# Create TFRecord files
create_tfrecord(x_train, y_train, 'tfrecords/train.tfrecord')
create_tfrecord(x_test, y_test, 'tfrecords/test.tfrecord')
create_tfrecord(x_val, y_val, 'tfrecords/val.tfrecord')
