import random
import cv2
import imutils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import numpy as np

cam = cv2.VideoCapture(0)

# Define color thresholding and mask

# Lower blue
lower_green = np.array([25, 50, 0])
high_green = np.array([102, 255, 255])

points = []
color = (0, 0, 0)

while(True):
    ret, frame = cam.read()

    frame = cv2.flip(frame, 1)

    if not ret:
        print("Failed to grab cam")
        break

    # cv2.imshow("Test", frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, high_green)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        print("Escape hit, closing...")
        break

    if k % 256 == 32:
        points.clear()
    
    if k % 256 == 122:
        color = (random.randrange(0, 255, 1), random.randrange(0, 255, 1), random.randrange(0, 255, 1))

    result = cv2.bitwise_and(frame, frame, mask=mask)
    result = cv2.erode(result, None, iterations = 4)
    result = cv2.dilate(result, None, iterations = 4)

    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    cnts = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    output = result.copy()

    boxes = []

    for p in range(len(points)):
            cv2.circle(frame, (points[p][0], points[p][1]), 5, color, thickness=-1)

    if len(cnts) > 0:
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            boxes.append([x,y, x+w,y+h])

        boxes = np.asarray(boxes)
        left, top = np.min(boxes, axis=0)[:2]
        right, bottom = np.max(boxes, axis=0)[2:]

        # center = (left + right // 2, top + bottom // 2

        points.append(boxes[0])
        # cv2.rectangle(frame, (left,top), (right,bottom), (255, 0, 0), 2)
    cv2.imshow("Frame", frame)
    # print("Loop end")

cam.release()
cv2.destroyAllWindows()







# cv2.imshow("original", image)
# # cv2.imshow("images", thresh)
# cv2.waitKey(0)
