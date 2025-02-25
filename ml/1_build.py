# Description:
# Our build phase simply gathers the required data and prepares it, saving it for use in the next steps
# We do this separately to training, and save the data before calling training separately, to break the ML process down into steps
# Each step will be run individually by the CircleCI pipeline so that any problems during a run can be identified and investigated
# The further you break down your process, the more insight you will get from your CI/CD pipeline if something goes wrong
# These stages may not map directly to your own stages or methodologies, but are demonstrative
# In real-world usage, your data would come from your own warehouse, and be regularly consumed, most likely via a separate ETL process

# Dependencies:
# You must have set up your environment by running abilities/install.sh to install the required Python dependencies

# Import dependencies
import sys
import os
import pickle
import subprocess
from tensorflow import keras

# Step 1: Install the requirements for the project
def install_requirements():
    try:
        print("Installing requirements from requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Requirements installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while installing requirements: {e}")
        sys.exit(1)

# Step 2: Prepare the data
def prepare_data():
    # Get the path to this script
    script_dir = sys.path[0]

    # Import the Fashion MNIST dataset - in real world usage, this would be your own data
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Scale the values to 0.0 to 1.0
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Reshape for feeding into the model
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

    print('\ntrain_images.shape: {}, of {}'.format(train_images.shape, train_images.dtype))
    print('test_images.shape: {}, of {}'.format(test_images.shape, test_images.dtype))

    # Ensure the training_data directory exists
    training_data_dir = os.path.join(script_dir, 'training_data')
    os.makedirs(training_data_dir, exist_ok=True)

    # Pickle is used to serialize numpy arrays for use in subsequent steps
    # https://docs.python.org/3/library/pickle.html
    with open(os.path.join(training_data_dir, 'train_images.pkl'), 'wb') as f:
        pickle.dump(train_images, f)

    with open(os.path.join(training_data_dir, 'train_labels.pkl'), 'wb') as f:
        pickle.dump(train_labels, f)

    with open(os.path.join(training_data_dir, 'test_images.pkl'), 'wb') as f:
        pickle.dump(test_images, f)

    with open(os.path.join(training_data_dir, 'test_labels.pkl'), 'wb') as f:
        pickle.dump(test_labels, f)

# Main function
if __name__ == "__main__":
    # Step 1: Install requirements
    install_requirements()

    # Step 2: Prepare the data
    prepare_data()
