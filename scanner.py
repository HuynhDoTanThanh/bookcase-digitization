import cv2
import numpy as np
import Utlis as utlis
import os

########################################################################
heightImg = 960
widthImg = 720
########################################################################

def scanner(pathImage):
    FJoin = os.path.join
    files = [FJoin(pathImage, f) for f in os.listdir(pathImage)]

    images = []

    for path in files:
        img = cv2.imread(path)
        img = cv2.resize(img, None, fx = 0.3, fy = 0.3)  # RESIZE IMAGE
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
        imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0)  # ADD GAUSSIAN BLUR
        imgThreshold = cv2.Canny(imgBlur, 30, 50)  # APPLY CANNY BLUR
        kernel = np.ones((5, 5))
        imgDial = cv2.dilate(imgThreshold, kernel, iterations = 2)  # APPLY DILATION
        imgThreshold = cv2.erode(imgDial, kernel, iterations = 1)  # APPLY EROSION

        ## FIND ALL COUNTOURS
        imgBigContour = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
        contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS

        # FIND THE BIGGEST COUNTOUR
        biggest, maxArea = utlis.biggestContour(contours)  # FIND THE BIGGEST CONTOUR
        if biggest.size != 0:
            biggest = utlis.reorder(biggest)
            cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)  # DRAW THE BIGGEST CONTOUR
            imgBigContour = utlis.drawRectangle(imgBigContour, biggest, 2)
            pts1 = np.float32(biggest)  # PREPARE POINTS FOR WARP
            pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # PREPARE POINTS FOR WARP
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
            images.append(imgWarpColored)
        else:
            print('not scanner image ', path)
            img = cv2.resize(img, (widthImg, heightImg))
            images.append(img)

    return images
