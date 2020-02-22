import cv2
import os

baseDir = os.getcwd()
sampleDir = os.path.join(baseDir, os.pardir, "sample")

frame = cv2.imread(os.path.join(sampleDir, "test_image.png"))
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

laplacian = cv2.Laplacian(frame_gray, cv2.CV_8U, ksize=3)

cv2.imwrite("laplacian.png", laplacian)

