import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
from PIL import Image
from keras import backend as K

def doRec(path,model,index):
    name = reverseBgColor(path, index)

    # 加载本地图像
    img = cv2.imread(name, 0)
    print(name)

    # 转换图像形状
    img_rows, img_cols = 28, 28
    if K.image_data_format() == 'channels_first':
        img = img.reshape(1, 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        img = img.reshape(1, img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    # 转换图像为MNIST格式
    img = img.astype('float32')
    img /= 255

    # 进行预测
    pred = model.predict(img)
    # print(pred)

    # 计算预测的精度
    pred_label = np.argmax(pred, axis=1)
    print('Predicted label:', path, pred_label)
    # image = cv2.imread(name, 0)
    # img_array = np.array(image).astype(np.float32)/255
    # img_array = tf.expand_dims(img_array, -1)  # Create batch axis
    # predictions = model.predict(img_array)
    # predictions = np.argmax(predictions)
    # print("result: ", path, predictions)

def main():
    model = load_model("final-100-epochs.model")
    # 加载模型
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])
    arr = [
        # "./test/0.jpg",
        # "./test/1.jpg",
        # "./test/2.jpg",
        # "./test/3.jpg",
        # "./test/4.jpg",
        # "./test/5.jpg", 
        # "./test/6.jpg",
        # "./test/7.jpg",
        # "./test/8.jpg",
        # "./test/9.jpg",
           "./../opencv_research/images/new_i_1.jpg", 
           "./../opencv_research/images/new_i_2.jpg",
           "./../opencv_research/images/new_i_3.jpg",
    ]
    # arr = ["./test/3.jpg", "./test/9.jpg", "./../opencv_research/images/new_i_1.jpg", "./../opencv_research/images/new_i_2.jpg","./../opencv_research/images/new_i_3.jpg",]
    for index in range(len(arr)):
        doRec(arr[index],model,index)

def reverseBgColor(path, index):
    #因为训练集的图片都是黑背景，白色字体，所以需要将输入的图片做相似处理
    image = cv2.imread(path, 0)
    image = cv2.resize(image,(28, 28))
    height, width = image.shape
    dst = np.zeros((height, width), np.uint8)
    for i in range(height):
        for j in range(width):
            dst[i,j] = 255-image[i,j]
    
    image = dst
    name = './test/new_x_%d.jpg' % index
    cv2.imwrite(name, image)#保存图片
    return name

main()