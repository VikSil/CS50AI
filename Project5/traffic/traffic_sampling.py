import cv2
import numpy as np
import os
import pandas as pd
import sys
import tensorflow as tf

import random
from tensorflow.python.keras.utils.layer_utils import count_params
from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4

HIDDEN_NODES_RANGE = (14, 129)
NUM_OF_FILTERS_RANGE = (1, 100)
SIZE_OF_FILTERS = [(2,2), (3,3), (4,4), (5,5)]
SIZE_OF_POOLING = [(2,2), (3,3), (4,4), (5,5)]
TYPE_OF_POOLING = ['max','average']
DROPOUT_RANGE = (0, 70)

OUTPUT_FILE = 'sample.csv'


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    df = pd.DataFrame(columns = ['accuracy', 'loss', 'acc_convergence','loss_convergence','acc_valid','loss_valid', 'parameters','hidden_nodes', 'filters', 'filter_size', 'pooling_size', 'pooling_type', 'dropout', ])

    df.to_csv(OUTPUT_FILE, index = False)

    for _ in range(3000):
        hidden_nodes = random.randint(HIDDEN_NODES_RANGE[0], HIDDEN_NODES_RANGE[1])
        filters = random.randint(NUM_OF_FILTERS_RANGE[0], NUM_OF_FILTERS_RANGE[1])
        filter_size = random.choice(SIZE_OF_FILTERS)
        pooling_size = random.choice(SIZE_OF_POOLING)
        pooling_type = random.choice(TYPE_OF_POOLING)
        dropout = random.randint(DROPOUT_RANGE[0], DROPOUT_RANGE[1])

        # Get a compiled neural network
        model = get_model(hidden_nodes,filters, filter_size, pooling_size,pooling_type, dropout/100 )

        trainable_count = count_params(model.trainable_weights)


        # Fit model on training data
        history = model.fit(x_train, y_train, epochs=EPOCHS, verbose = None)

        # Evaluate neural network performance
        output = model.evaluate(x_test,  y_test, verbose = None)

        acc_convergence_epochs = loss_convergence_epochs = 0

        for i in range(EPOCHS):
            if output[1] - history.history['accuracy'][i] < 0.01:
                acc_convergence_epochs = i+1
                break

        for i in range(EPOCHS):
            if output[0] - history.history['loss'][i] < 0.01:
                loss_convergence_epochs = i+1
                break

        acc_valid = output[1] > history.history['accuracy'][EPOCHS-1]
        loss_valid = output[1] > history.history['loss'][EPOCHS-1]


        row = [round(output[1],4), round(output[0],4), acc_convergence_epochs, loss_convergence_epochs, acc_valid, loss_valid, trainable_count, hidden_nodes, filters, filter_size,pooling_size, pooling_type, dropout]
        df.loc[-1] = row
        df.to_csv(OUTPUT_FILE, index = False, header = False, mode = 'a')
        df.drop(df.tail(1).index,inplace=True)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []

    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            
            # convert to class 'numpy.ndarray'
            image = cv2.imread(f'{subdir}{os.sep}{file}')
            # normalise colorspace
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image / 255.0
            # normalise size
            image = cv2.resize(image, dsize=(IMG_WIDTH, IMG_HEIGHT))
            images.append(image)
            
            labels.append(subdir.split(os.sep)[-1])

    return (images, labels)

def get_model(hidden_nodes=1,filters=1, filter_size= (2,2), pooling_size=(2,2),pooling_type = 'max', dropout= 0):
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    if pooling_type == 'max':
        pooling_layer = tf.keras.layers.MaxPooling2D(pool_size=pooling_size)
    else:
        pooling_layer = tf.keras.layers.AveragePooling2D(pool_size=pooling_size)


    model = tf.keras.models.Sequential(
        [
        # TODO - get a better grip on how to form these layers
        # and argument in the README.md
           tf.keras.layers.Conv2D(
               # 32 is the number of filters in the convolutional layer  
               # (3,3) means that each filter will have a 3 x 3 kernel
               filters, filter_size, activation = 'relu', input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)
               ), 
           
           # pooling layer reduces 2 x 2 regions to single max value
           pooling_layer,
           
           tf.keras.layers.Flatten(),
           
           # hidden layer with 128 nodes
           tf.keras.layers.Dense(hidden_nodes,activation = 'relu'),
           # 50% dropout
           tf.keras.layers.Dropout(dropout),
           
           tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
        ]        
    )

    
    model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['accuracy'] )
    
    return model


if __name__ == "__main__":
    main()
