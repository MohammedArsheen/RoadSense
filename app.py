from flask import Flask, render_template, Response, jsonify
import cv2
import time
import argparse
import imutils
import threading
from traffic_analyzer import TrafficAnalyzer
from accident_detector import AccidentDetector

app = Flask(__name__)

# GLOBAL SHARED DATA
LATEST_FRAME = None
TRAFFIC_DATA = {
    "level": "LOW",
    "vehicle_count": 0,
    "color": "green",
    "alerts": [],
    "last_updated": time.time()
}

class ThreadedStream:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.analyzer = TrafficAnalyzer(fps=25)
        self.obstruction_detector = AccidentDetector(fps=25, stationary_duration=30)
        self.running = True
        self.lock = threading.Lock()
        print(f"--- TACTICAL SYSTEM INITIALIZING: {video_path} ---")

    def run(self):
        global LATEST_FRAME, TRAFFIC_DATA
        frame_idx = 0
        while self.running:
            success, frame = self.cap.read()
            if not success:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # AI Inference
            frame = imutils.resize(frame, width=1024)
            try:
                # 1. ANALYZE (YOLO INSIDE - ALREADY FILTERED FOR VEHICLES)
                vehicle_count, level, color, boxes, alerts = self.analyzer.process_frame(frame)
                _, _, stationary_info, is_congestion = self.obstruction_detector.process_frame(boxes)
                
                # 2. DRAW PRECISION OVERLAYS
                # Draw Blue Tactical Boxes for all vehicles correctly (x, y, w, h)
                for box in boxes:
                    x, y, w, h = map(int, box)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1) # Tactical Blue

                # Draw Red Highlight for Stationary Vehicles
                for info in stationary_info:
                    x, y, w, h = info['bbox']
                    f_still = info['stationary_count']
                    if f_still > 15:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3) # Warning Red
                        cv2.putText(frame, f"STILL: {f_still//25}s", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Draw Global HUD Overlay
                hub_col = (0, 255, 0) if level == "LOW" else (0, 255, 255) if level == "MEDIUM" else (0, 0, 255)
                cv2.rectangle(frame, (15, 15), (500, 80), (40, 40, 40), -1) # HUD Backdrop
                cv2.putText(frame, f"AI SURVEILLANCE: {level} ({vehicle_count} VEHICLES)", (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, hub_col, 2)
                
                if is_congestion:
                    cv2.putText(frame, "!!! GLOBAL CONGESTION !!!", (520, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 3)

                # 3. UPDATE GLOBAL STATE
                with self.lock:
                    TRAFFIC_DATA = {
                        "level": level,
                        "vehicle_count": vehicle_count,
                        "color": "green" if level == "LOW" else "yellow" if level == "MEDIUM" else "red",
                        "alerts": alerts,
                        "last_updated": time.time()
                    }
                    LATEST_FRAME = frame.copy()
                
                frame_idx += 1
                if frame_idx % 20 == 0: print(f"PROTOCOL: AI VISION ACTIVE (Frame {frame_idx})")
                
            except Exception as e:
                print(f"CRITICAL AI ERROR: {e}")
                time.sleep(1)

    def stop(self):
        self.running = False
        self.cap.release()

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    def stream():
        global LATEST_FRAME
        while True:
            if LATEST_FRAME is not None:
                ret, buffer = cv2.imencode('.jpg', LATEST_FRAME)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.04)
    return Response(stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def status():
    return jsonify({"Guindy Road": TRAFFIC_DATA})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="data/Traffic.mp4")
    args = parser.parse_args()
    
    streamer = ThreadedStream(args.video)
    t = threading.Thread(target=streamer.run)
    t.daemon = True
    t.start()
    
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
