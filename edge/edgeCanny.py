import cv2

import os

baseDir = os.getcwd()
sampleDir = os.path.join(baseDir, os.pardir, "sample")

frame = cv2.imread(os.path.join(sampleDir, "test_image.png"))
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

canny = cv2.Canny(frame_gray, 30, 150)

cv2.imwrite("canny.png", canny)

