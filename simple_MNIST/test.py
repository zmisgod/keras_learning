import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model

def main():
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255

    x_test = np.expand_dims(x_test, -1)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = load_model("final.model")

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

main()