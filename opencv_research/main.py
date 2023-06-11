import cv2
import numpy as np

def main():
    image = cv2.imread("./images/138.jpg")
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 150,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = contours

    all = cv2.drawContours(imgray, contours, -1, (0,255,0),2)
    #cv2.imshow('all', all)
    '''
    cnt = contours[1]
    dst = cv2.drawContours(image, cnt, -1, (255,0,0),4)

    area = cv2.contourArea(cnt)
    print("area:", area)#面积
    

    perimeter = cv2.arcLength(cnt, True)
    print("perimeter:", perimeter)#周长
    

    epsilon = 0.05 * perimeter #设置精度（从轮廓到近似轮廓的最大距离）

    approx = cv2.approxPolyDP(cnt, epsilon, True)

    drawImg = image.copy()
    res = cv2.drawContours(drawImg, [approx], -1, (0,0,255), 1)
    '''
    ## 绘制边界
    for index in range(len(cnt)):
        if index == 0:
            continue
        x,y,w,h = cv2.boundingRect(cnt[index])
        print(x,y,w,h)
        dst = imgray.copy()
        dst = cv2.rectangle(dst, (x,y), (x+w, y+h), (0,255,0), 3)
        cv2.imshow('dts%d'%index, dst)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()