import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
from keras.models import load_model
from PIL import Image
from keras import backend as K

def main():
    model = load_model("./../simple_MNIST/final-100-epochs.model")
    # 加载模型
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

    image = cv2.imread("./images/139.jpg")
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_gaussian = cv2.GaussianBlur(imgray, (5, 5), 0)
    ret, thresh = cv2.threshold(gray_gaussian, 0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)#使用Otsu的二值化算法进行图像二值化处理
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = contours

    # heightArr = []
    saveBox = []
    areaMap = {}

    nowStart = 0
    imgShowQueue = {}
    imgShowArr = {}

    # 在下面中 index = 0 会被过滤，统一解释一下：
    # 因为index = 0是openCV识别的整个数字的边界，而不是单个数字的边界，所以需要直接跳过
    # index > 0 则为每个识别数字的轮廓索引

    # 首先取出每个识别对象的4个订单坐标
    for index in range(len(cnt)):
        x,y,w,h = cv2.boundingRect(cnt[index])
        area = cv2.contourArea(cnt[index])
        areaMap[index] = area
        # heightArr.append(h)
        # 保存每个识别区域4个顶点坐标
        saveBox.append([x,y, x+w,y+h])
        imgShowQueue[x] = index

    sortedArr = sorted(imgShowQueue)
    print("sortedArr", sortedArr)
    print("imgShowQueue", imgShowQueue)
    for index in range(len(sortedArr)):
        imgShowArr[imgShowQueue[sortedArr[index]]] = {'index':imgShowQueue[sortedArr[index]]}
    
    print("imgShowArr", imgShowArr)
    # 再去判断每个识别对象是否包含其他的对象
    hitIndex = []
    for index in range(len(saveBox)):
        if index == 0:
            continue
        for iindex in range(len(saveBox)):
            if iindex == 0:
                continue
            if index == iindex:
                continue
            # 如果2个识别区域a 和b中有面积相交，则优先使用面积大的数据
            if compare(saveBox[index], saveBox[iindex]):
                hitIndex.append([index, iindex])
    
    print("hitIndex",hitIndex)
    # 如果2个对象a和b的4个顶点坐标出现相交，则优先取面积大的那个坐标
    # 将小的对应的索引记录下来，方便后续过滤
    notSaveIndex = []
    for index in range(len(hitIndex)):
        if index == 0:
            continue
        oneArea = areaMap[hitIndex[index][0]]
        twoArea = areaMap[hitIndex[index][1]]
        if oneArea >= twoArea:
            notSaveIndex.append(hitIndex[index][1])
        else:
            notSaveIndex.append(hitIndex[index][0])

    print("notSaveIndex",notSaveIndex)

    # 顺序读取每个数字的切片信息
    for index in range(len(cnt)):
        if index == 0:
            continue
        if inArray(notSaveIndex, index):
            continue 
        x,y,w,h = cv2.boundingRect(cnt[index])
        dst = imgray.copy()
        rest = dst[y:y+h, x:x+w]
        name = './images/new_i_%d.jpg' % index
        cv2.imwrite(name, rest)#保存图片
        img_a = Image.open(name)
        # 计算原始图片的缩放比例
        size = 24
        bgSize = 28
        width, height = img_a.size
        ratio = min(size/width, size/height)
        new_size = (int(width*ratio), int(height*ratio))

        # 缩放原始图片，使它在 28x28 的图片上面保持比例不变
        img_a = img_a.resize(new_size)

        # 创建空白的背景图片
        img_bg = Image.new('RGB', (bgSize, bgSize), color='white')

        # 计算图片放置位置
        x = (bgSize - new_size[0]) // 2
        y = (bgSize - new_size[1]) // 2

        # 在背景图片上粘贴原始图片
        img_bg.paste(img_a, (x, y))
        img_bg.save(name)

        width, height = img_bg.size
        for x in range(width):
            for y in range(height):
                r, g, b = img_bg.getpixel((x, y))
                if r == g == b and r != 255:
                    img_bg.putpixel((x, y), (0, 0, 0))
        
        img_bg.save(name)
        imgShowArr[index]["img"] = name

    print("imgShowArr--", imgShowArr)
    resultNumStr = ""
    # 开始识别多数字
    for index in imgShowArr:
        if "img" in imgShowArr[index].keys():
            oneNumStr = doRec(imgShowArr[index]["img"],model,index)
            resultNumStr += oneNumStr
    
    print("result == ", resultNumStr)

def inArray(arr, val):
    for index in range(len(arr)):
        if arr[index] == val:
            return True
    return False

def compare(rect1, rect2):
    if rect1[0] > rect2[2] or rect2[0] > rect1[2] or rect1[1] > rect2[3] or rect2[1] > rect1[3]:
        return False
    else:
        return True

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
    if len(pred_label) > 0:
        return str(pred_label[0])
    return ""

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
    name = './images/new_x_%d.jpg' % index
    cv2.imwrite(name, image)#保存图片
    return name

main()