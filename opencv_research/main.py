import cv2
import numpy as np

def main():
    image = cv2.imread("./images/136.jpg")
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_gaussian = cv2.GaussianBlur(imgray, (5, 5), 0)
    ret, thresh = cv2.threshold(gray_gaussian, 0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)#使用Otsu的二值化算法进行图像二值化处理
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = contours

    all = cv2.drawContours(imgray, contours, -1, (0,255,0),2)
    differenceNum = 20#接收误差在20之内
    heightArr = []

    for index in range(len(cnt)):
        if index == 0:
            continue
        x,y,w,h = cv2.boundingRect(cnt[index])
        heightArr.append(h)
        print( x,y,w,h)
    
    middleNum = np.median(heightArr)
    for index in range(len(cnt)):
        if index == 0:
            continue
        x,y,w,h = cv2.boundingRect(cnt[index])
        if h <= (middleNum + differenceNum) and h >= (middleNum - differenceNum):
            dst = imgray.copy()
            dst = cv2.rectangle(dst, (x,y), (x+w, y+h), (0,255,0), 3)
            cv2.imshow('dts%d'%index, dst)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()