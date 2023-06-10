import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model

def doRec(path, model,index):
    #因为训练集的图片都是黑背景，白色字体，所以需要将输入的图片做相似处理
    image = cv2.imread(path, 0)
    image = cv2.resize(image,(28, 28))
    height, width = image.shape
    dst = np.zeros((height, width), np.uint8)
    for i in range(height):
        for j in range(width):
            dst[i,j] = 255-image[i,j]
    
    image = dst
    name = './test/new_%d.jpg' % index
    cv2.imwrite(name, image)#保存图片

    image = np.array(image).astype(np.float32)
    img_array = keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    predictions = model.predict(img_array)
    predictions = np.argmax(predictions)
    print("result: ", path, predictions)


def main():
    model = load_model("final.model")
    arr = ["./test/0.jpg","./test/1.jpg","./test/2.jpg",
           "./test/3.jpg","./test/4.jpg","./test/5.jpg",
           "./test/6.jpg","./test/7.jpg","./test/8.jpg",
           "./test/9.jpg","./test/13.jpg"
        ]
    for index in range(len(arr)):
        doRec(arr[index],model,index)

main()
'''
1/1 [==============================] - 3s 3s/step
result:  ./test/0.jpg 0
1/1 [==============================] - 0s 16ms/step
result:  ./test/1.jpg 1
1/1 [==============================] - 0s 16ms/step
result:  ./test/2.jpg 2
1/1 [==============================] - 0s 16ms/step
result:  ./test/3.jpg 3
1/1 [==============================] - 0s 16ms/step
result:  ./test/4.jpg 4
1/1 [==============================] - 0s 16ms/step
result:  ./test/5.jpg 5
1/1 [==============================] - 0s 16ms/step
result:  ./test/6.jpg 6
1/1 [==============================] - 0s 16ms/step
result:  ./test/7.jpg 7
1/1 [==============================] - 0s 16ms/step
result:  ./test/8.jpg 8
1/1 [==============================] - 0s 0s/step
result:  ./test/9.jpg 7
1/1 [==============================] - 0s 0s/step
result:  ./test/13.jpg 6 ##13没有识别出来
'''