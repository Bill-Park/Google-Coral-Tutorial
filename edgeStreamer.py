from flask import Flask, render_template, Response
import cv2
import numpy as np
from edgetpu.classification.engine import ClassificationEngine
import os
from PIL import Image
import time

modelPath = os.path.join(os.getcwd(), "ml", "models", "model_edgetpu.tflite")

app = Flask(__name__)

@app.route('/')
def index() :
    return render_template('index.html')

def get_frame() :
    cap = cv2.VideoCapture(0)
    engine = ClassificationEngine(modelPath)
    prevTime = 0
    while True :
        _, frame = cap.read()
        curTime = time.time()
        sec = curTime - prevTime
        prevTime = curTime
        fps = 1 / (sec)
        fpsText = "FPS : {:.2f}".format(fps)

        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        framePIL = Image.fromarray(frameRGB)
        classify = engine.classify_with_image(framePIL)
        
        label = classify[0][0]
        if label == 0 :
            labelText = "rock"
        elif label == 1 :
            labelText = "paper"
        elif label == 2 :
            labelText = "scissors"

        score = round(classify[0][1], 3)
        
        print(labelText, score)

        cv2.putText(frame, labelText + " " + str(score), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.putText(frame, fpsText, (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
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
