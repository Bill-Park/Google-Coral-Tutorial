import cv2

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True :
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, img_binary = cv2.threshold(frame_gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        cv2.drawContours(frame, [cnt], 0, (255, 0, 0), 3)  # blue

    cv2.imshow("VideoFrame", frame)

    cv2.imshow("binary", img_binary)

    if cv2.waitKey(1) > 0: break

cap.release()
cv2.destroyAllWindows()
