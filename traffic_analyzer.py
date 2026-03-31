import cv2
import time
from ultralytics import YOLO

class TrafficAnalyzer:
    def __init__(self, fps=30, high_traffic_threshold=15):
        self.fps = fps
        
        # Traffic Levels
        self.LOW_MAX = 14     # So <15 is Green
        self.MEDIUM_MAX = 25  # So >=15 and <=25 is Yellow, >25 is Red
        
        # High traffic tracking (Alert System Requirements)
        self.high_traffic_consecutive_frames = 0
        # Condition: traffic remains HIGH for > 15 minutes (15 * 60 * fps)
        self.high_traffic_alert_threshold = 15 * 60 * fps 
        
        # Accident Alert: sudden congestion spike detection
        self.prev_count = 0
        self.spike_threshold = 15 # Sudden jump of 15 vehicles in a single frame search
        
        # Initialize YOLOv8
        # ultralytics will auto-download yolov8n.pt if not found locally
        print("Loading YOLOv8 model... This might take a few seconds on first run.")
        self.model = YOLO('yolov8n.pt')
        
        # YOLOv8 COCO classes for vehicles
        self.vehicle_classes = [2, 3, 5, 7] # 2: car, 3: motorcycle, 5: bus, 7: truck
        
    def process_frame(self, frame, min_contour_area=None):
        """Processes a frame using YOLOv8, counts vehicles, determining traffic level, tracks duration."""
        
        # Run YOLO inference
        # verbose=False avoids spamming the console with predictions every frame
        results = self.model(frame, verbose=False, conf=0.15, imgsz=640) # Speed-Optimized Size
        
        bounding_boxes = []
        vehicle_count = 0
        
        # Parse YOLO results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get the class ID
                cls = int(box.cls[0])
                
                # Check if it's a vehicle
                if cls in self.vehicle_classes:
                    # Confidence threshold could go here if needed (e.g., float(box.conf[0]) > 0.3)
                    
                    # Get bounding box coordinates in xywh format (center x, center y, width, height)
                    # We need it in format (x_top_left, y_top_left, w, h) for our downstream tasks
                    x1, y1, x2, y2 = box.xyxy[0]
                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                    
                    bounding_boxes.append((x, y, w, h))
                    vehicle_count += 1
                
        # Classify Traffic Level
        if vehicle_count <= self.LOW_MAX:
            level = "LOW"
            color = (0, 255, 0) # Green
            self.high_traffic_consecutive_frames = 0 # reset
            
        elif vehicle_count <= self.MEDIUM_MAX:
            level = "MEDIUM"
            color = (0, 255, 255) # Yellow
            self.high_traffic_consecutive_frames = 0 # reset
            
        else:
            level = "HIGH"
            color = (0, 0, 255) # Red
            self.high_traffic_consecutive_frames += 1

        # Check for alerts
        current_alerts = []
        
        # 1. High Persistence Check (15 Minutes)
        if self.high_traffic_consecutive_frames >= self.high_traffic_alert_threshold:
            current_alerts.append("CRITICAL: HIGH TRAFFIC >> 15 MINUTES DETECTED")
            self.high_traffic_consecutive_frames = 0 # reset to prevent spam
            
        # 2. Sudden Spike Check (Accident Detected)
        if vehicle_count - self.prev_count >= self.spike_threshold:
            current_alerts.append("ACCIDENT DETECTED: SUDDEN SPIKE IN VEHICLE FLOW")
            
        self.prev_count = vehicle_count # update for next frame
            
        return vehicle_count, level, color, bounding_boxes, current_alerts
