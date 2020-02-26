from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/')
def index() :
    return render_template('index.html')

def get_frame() :
    cap = cv2.VideoCapture(0)
    while True :
        _, frame = cap.read()
        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lowSkin = np.array([0, 40, 80])
        highSkin = np.array([20, 255, 255])
        mask = cv2.inRange(frameHSV, lowSkin, highSkin)
        skin = cv2.bitwise_and(frame, frame, mask=mask)
        
        frameGray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
        _, imgBinary = cv2.threshold(frameGray, 80, 255, 0)

        contours, hierarchy = cv2.findContours(imgBinary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        maxArea = [0, 0]
        for cnt in contours :
            size = cv2.contourArea(cnt)
            if size > maxArea[0] :
                maxArea[0] = size
                maxArea[1] = cnt

        if maxArea[0] != 0 :
            cv2.drawContours(frame, [maxArea[1]], 0, (0, 0, 255), 3)

        imgencode = cv2.imencode('.jpg', frame)[1]
        stringData = imgencode.tostring()
        yield (b'--frame\r\n'
                b'Content-Type: text/plain\r\n\r\n' + stringData + b'\r\n')

    del(cap)

@app.route('/calc')
def calc() :
    return Response(get_frame(),
            mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__' :
    app.run(host='0.0.0.0', debug=True, threaded=True)
