import cv2
import os

baseDir = os.getcwd()
sampleDir = os.path.join(baseDir, os.pardir, "sample")

frame = cv2.imread(os.path.join(sampleDir, "test_image.png"))
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


_, img_binary = cv2.threshold(frame_gray, 170, 255, 0)
contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    cv2.drawContours(frame, [cnt], 0, (255, 0, 0), 3)  # blue

cv2.imwrite("contour.png", frame)
cv2.imwrite("test.png", img_binary)

