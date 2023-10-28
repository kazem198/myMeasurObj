import cv2
import numpy as np
import utils

webFrame = False
path = "images/2022011014150371.jpg"
scale = 3
hP = 297*scale
wP = 210*scale

cap = cv2.VideoCapture(0)
cap.set(10, 160)  # cv2.CAP_PROP_BRIGHTNESS
cap.set(3, 1920)
cap.set(4, 1080)

while True:
    if webFrame:
        success, img = cap.read()
    else:
        img = cv2.imread(path)

    biggestCon = utils.findContours(img, minArea=50000)

    wrapImg = utils.wrapImage(img, biggestCon[0][0], wP, hP, 20, )

    objCon = utils.findContours(
        wrapImg, minArea=5000, )

    for i in objCon:
        # print("i", (i[0]))

        points = i[0]
        pt1 = points[0][0]
        pt2 = points[1][0]
        pt3 = points[2][0]
        # print("pt1", pt1)

        dis1 = utils.distance(pt1//scale, pt2//scale)

        cv2.arrowedLine(wrapImg, (pt1[0], pt1[1]),
                        (pt2[0], pt2[1]), (255, 0, 0), 4)
        cv2.putText(
            wrapImg, f'{str(round(dis1, 1))}cm', (pt1[0], pt1[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

        dis2 = utils.distance(pt1//scale, pt3//scale)

        cv2.arrowedLine(wrapImg, (pt1[0], pt1[1]),
                        (pt3[0], pt3[1]), (0, 255, 0), 2)
        cv2.putText(
            wrapImg, f'{str(round(dis2, 1))}cm', (pt3[0], pt3[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    finalWarp = utils.reverseWrap(biggestCon[0][0], wrapImg,
                                  (img.shape[1], img.shape[0]))
    finalWarp = cv2.resize(finalWarp, (wP, hP), None)

    cv2.imshow("finalWarp", finalWarp)

    img = cv2.resize(img, (wP, hP), None)

    add = cv2.addWeighted(img, 1, finalWarp, 1, 1)

    cv2.imshow("imgOriginal", img)
    # cv2.imshow("wrapImg", wrapImg)
    cv2.imshow("add", add)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break
