import cv2
import numpy as np
from PIL import Image

def main():
    image = cv2.imread("./images/139.jpg")
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_gaussian = cv2.GaussianBlur(imgray, (5, 5), 0)
    ret, thresh = cv2.threshold(gray_gaussian, 0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)#使用Otsu的二值化算法进行图像二值化处理
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = contours

    heightArr = []
    saveBox = []
    areaMap = {}

    for index in range(len(cnt)):
        x,y,w,h = cv2.boundingRect(cnt[index])
        area = cv2.contourArea(cnt[index])
        areaMap[index] = area
        heightArr.append(h)
        saveBox.append([x,y, x+w,y+h])

    hitIndex = []
    for index in range(len(saveBox)):
        if index == 0:
            continue
        for iindex in range(len(saveBox)):
            if iindex == 0:
                continue
            if index == iindex:
                continue
            if compare(saveBox[index], saveBox[iindex]):
                hitIndex.append([index, iindex])
    
    print("hitIndex",hitIndex)
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

main()