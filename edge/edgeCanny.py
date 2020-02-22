import cv2

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True :
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edge = cv2.Canny(frame_gray, 30, 150)

    cv2.imshow("canny", edge)

    cv2.imshow("VideoFrame", frame)
    if cv2.waitKey(1) > 0: break

cap.release()
cv2.destroyAllWindows()
