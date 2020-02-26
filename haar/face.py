import cv2
import os

baseDir = os.getcwd()
sampleDir = os.path.join(baseDir, os.pardir, "sample")
haarDir = os.path.join(baseDir, os.pardir, "haar", "data", "haarcascades")

faceCascade = cv2.CascadeClassifier(os.path.join(haarDir, "haarcascade_frontalface_default.xml"))

frame = cv2.imread(os.path.join(sampleDir, "group_face.jpg"))
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(frame_gray, 1.2, 7)
print(faces)

for (x,y,w,h) in faces :
    cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)

cv2.imwrite("faceRecognition.png", frame)
