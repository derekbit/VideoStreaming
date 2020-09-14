from flask import Flask, Response, render_template
from camera import VideoCamera
import argparse
import cv2
from yolo import prepare_net

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def generate(cap):
    while True:
        frame = cap.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
@app.route('/video_feed')
def video_feed():
    return Response(generate(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-w', '--weights',
        type=str,
        default='./yolov3-tiny/yolov3-tiny.weights',
        help='Path to the file which contains the weights for YOLOv3.')

    parser.add_argument('-c', '--config',
        type=str,
        default='./yolov3-tiny/yolov3-tiny.cfg',
        help='Path to the configuration file for YOLOv3 model.')

    parser.add_argument('-l', '--labels',
        type=str,
        default='./yolov3-tiny/coco.names',
        help='Path to the labels file for YOLOv3 model.')

    args, unparsed = parser.parse_known_args()
    prepare_net(args.config, args.weights, args.labels)

    app.run(host='0.0.0.0', port=5000, debug=True)
