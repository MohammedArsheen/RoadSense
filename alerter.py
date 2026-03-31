import os
import cv2
from datetime import datetime

class Alerter:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def trigger_traffic_alert(self, duration_seconds):
        """Triggers an alert if traffic is HIGH for a prolonged duration."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] ALERT: HIGH traffic persists for {duration_seconds} seconds! Please dispatch aid.")

    def trigger_parking_alert(self, frame_count, frame, boxes):
        """Captures a snapshot and triggers an alert if illegal parking/obstruction is detected."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"illegal_parking_{timestamp}_frame{frame_count}.jpg")
        
        # Draw bounding boxes on the saved snapshot for context
        snapshot = frame.copy()
        for (x, y, w, h) in boxes:
            cv2.rectangle(snapshot, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
        cv2.imwrite(filename, snapshot)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ALERT: Illegal Parking / Obstruction detected. Evidence saved to {filename}")
