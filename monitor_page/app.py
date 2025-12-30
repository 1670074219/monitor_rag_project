import cv2
import threading
import time
import json
import os
from flask import Flask, render_template, Response

# Force FFmpeg to use TCP for RTSP to improve stability and avoid UDP packet loss/timeouts
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

app = Flask(__name__)

# Configuration path
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'video_process', 'camera_config.json')

class Camera:
    def __init__(self, camera_id, rtsp_url):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            # Use CAP_FFMPEG explicitly
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            
            if not cap.isOpened():
                print(f"Warning: Could not open video source for {self.camera_id}. Retrying in 5 seconds...")
                time.sleep(5)
                continue
            
            # Reduce buffer size to minimize latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            print(f"Successfully connected to {self.camera_id}")
            
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # Resize to reduce bandwidth/CPU usage (Monitor doesn't need 4K)
                    # This also helps prevent memory issues with multiple high-res streams
                    try:
                        height, width = frame.shape[:2]
                        target_width = 640
                        if width > target_width:
                            scale = target_width / width
                            target_height = int(height * scale)
                            frame = cv2.resize(frame, (target_width, target_height))
                        
                        with self.lock:
                            self.frame = frame
                    except Exception as e:
                        print(f"Error processing frame for {self.camera_id}: {e}")
                else:
                    print(f"Lost connection to {self.camera_id}. Reconnecting...")
                    break
                # Small sleep to prevent busy loop if read() is non-blocking (though usually it blocks)
                # time.sleep(0.005) 
            
            cap.release()
            if self.running:
                time.sleep(2)

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                # Return a black placeholder or None
                return None
            
            try:
                # Encode as JPEG
                ret, jpeg = cv2.imencode('.jpg', self.frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if not ret:
                    return None
                return jpeg.tobytes()
            except Exception as e:
                print(f"Error encoding frame for {self.camera_id}: {e}")
                return None

    def stop(self):
        self.running = False
        self.thread.join()

cameras = {}

def load_cameras():
    global cameras
    if not os.path.exists(CONFIG_PATH):
        print(f"Config file not found at {CONFIG_PATH}")
        return

    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    for cam_conf in config.get('camera_config', []):
        cam_id = cam_conf['camera_id']
        url = cam_conf['camera_url']
        # Only add if not already present (or handle updates)
        if cam_id not in cameras:
            print(f"Initializing camera {cam_id}...")
            cameras[cam_id] = Camera(cam_id, url)
            # Stagger startup to avoid network/CPU spike
            time.sleep(0.5)

# Initialize cameras on startup
load_cameras()

def generate(camera_id):
    cam = cameras.get(camera_id)
    if not cam:
        return
    
    while True:
        frame = cam.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            # Return a blank image or wait
            time.sleep(0.1)

@app.route('/')
def index():
    camera_list = list(cameras.keys())
    return render_template('index.html', cameras=camera_list)

@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    return Response(generate(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Run threaded to handle multiple requests
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
