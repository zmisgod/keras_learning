import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.models import load_model

def doRec(path, model):
    image_size = (180, 180)
    img = keras.utils.load_img(path, target_size=image_size)
    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    predictions = model.predict(img_array)
    score = float(predictions[0])
    print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")

def test():
    model = load_model("finial.model")
    arr = ["./test/1.jpeg", "./test/2.jpeg","./test/3.jpeg", "./test/4.jpeg"]
    for index in range(len(arr)):
        doRec(arr[index], model)

test()
'''
1/1 [==============================] - 3s 3s/step
This image is 99.98% cat and 0.02% dog.
1/1 [==============================] - 0s 16ms/step
This image is 93.77% cat and 6.23% dog.
1/1 [==============================] - 0s 0s/step
This image is 10.44% cat and 89.56% dog.
1/1 [==============================] - 0s 16ms/step
This image is 0.00% cat and 100.00% dog.
'''