# from tensorflow import keras
# print(keras.__version__)
# import tensorflow as tf
# physical_device = tf.config.experimental.list_physical_devices('GPU')
# print(f'Device found : {physical_device}')
import tensorflow as tf
from tensorflow.python.client import device_lib
print(tf.__version__)
print(device_lib.list_local_devices())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# print(tf.__version__)
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# tf.test.is_gpu_available()