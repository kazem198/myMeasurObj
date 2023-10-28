import cv2
import numpy as np


def reOrder(points):

    newPoint = np.zeros_like(points)
    points = np.reshape(points, (4, 2))
    add = points.sum(1)
    newPoint[0] = points[np.argmin(add)]
    newPoint[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    newPoint[1] = points[np.argmin(diff)]
    newPoint[2] = points[np.argmax(diff)]
    return newPoint


def findContours(img, showCanny=False, CTher=[50, 150], drawContours=False, minArea=1000):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 2)
    imgCanny = cv2.Canny(imgBlur, CTher[0], CTher[1])
    if showCanny:
        cv2.imshow("canny", imgCanny)

    kernel = np.ones((5, 5))
    imgdilate = cv2.dilate(imgCanny, kernel, iterations=3)
    imgErod = cv2.erode(imgdilate, kernel, iterations=2)
#     cv2.imshow("imgerod", imgErod)
    contours, _ = cv2.findContours(
        imgErod, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    biggestContours = []
    for con in contours:
        area = cv2.contourArea(con)
        # print(area)

        if (area > minArea):
            peri = cv2.arcLength(con, True)
            approx = cv2.approxPolyDP(con, 0.02*peri, True)
            bbox = cv2.boundingRect(approx)
        #     print(approx)
            if drawContours:
                cv2.drawContours(img, con, -1, (0, 0, 255), 3)
            newPoints = reOrder(approx)

            biggestContours.append([newPoints, con, bbox])

    return biggestContours


def wrapImage(img, newPoints, width, height, cut, showImg=False):
    pt1 = np.float32(newPoints)
    pt2 = np.float32(
        [[0, 0], [width, 0], [0, height], [width, height]])

    matrix = cv2.getPerspectiveTransform(pt1, pt2)

    wrapImg = cv2.warpPerspective(img, matrix, (width, height))

    wrapImg = wrapImg[cut:height-cut, cut:width-cut]

    if showImg:
        cv2.imshow("wrapImg", wrapImg)

    return wrapImg


def distance(pt1, pt2):
    return ((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)**0.5 / 10


def reverseWrap(biggestCon, wrap,  size):

    pt1 = np.float32([[0, 0], [wrap.shape[1], 0], [0, wrap.shape[0]], [
                     wrap.shape[1], wrap.shape[0]]])

    pt2 = np.float32(biggestCon)

    matrix = cv2.getPerspectiveTransform(pt1, pt2)

    finalWarp = cv2.warpPerspective(wrap, matrix, size)

    return finalWarp
