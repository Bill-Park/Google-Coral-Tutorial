import cv2

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True :
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    img_sobel_x = cv2.Sobel(frame_gray, cv2.CV_64F, 1, 0, ksize=3)
    img_sobel_x = cv2.convertScaleAbs(img_sobel_x)

    img_sobel_y = cv2.Sobel(frame_gray, cv2.CV_64F, 0, 1, ksize=3)
    img_sobel_y = cv2.convertScaleAbs(img_sobel_y)

    img_sobel = cv2.addWeighted(img_sobel_x, 1, img_sobel_y, 1, 0)

    cv2.imshow("sobel", img_sobel)

    cv2.imshow("VideoFrame", frame)
    if cv2.waitKey(1) > 0: break

cap.release()
cv2.destroyAllWindows()
