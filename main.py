import cv2
import argparse
import imutils
from traffic_analyzer import TrafficAnalyzer
from accident_detector import AccidentDetector
from alerter import Alerter

def main():
    parser = argparse.ArgumentParser(description="Smart Traffic Monitoring System")
    parser.add_argument("--video", "-v", type=str, default="data/demo_video.mp4", help="Path to input video file")
    # For demo, lower the alert threshold to 15 seconds. Real world: 15 * 60 = 900
    parser.add_argument("--high_traffic_threshold", "-t", type=int, default=15, help="Seconds for HIGH traffic alert")
    args = parser.parse_args()
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.video}. Please place a test video in data/demo_video.mp4")
        return
        
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30 # fallback
        
    print(f"Starting analysis on {args.video} at {fps} FPS.")
        
    analyzer = TrafficAnalyzer(fps=fps, high_traffic_threshold=args.high_traffic_threshold)
    # 30 seconds wait before flagging as illegal/obstruction
    obstruction_detector = AccidentDetector(fps=fps, stationary_duration=30, min_movement=3, detection_zone_y_min=100)
    alerter = Alerter(output_dir="output")
    
    frame_count = 0
    paused = False
    
    # PERFORMANCE: Process every N-th frame to increase speed
    process_every_n_frames = 1
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream.")
                break
                
            frame_count += 1
            
            # Skip frames to speed up processing
            if frame_count % process_every_n_frames != 0:
                continue
                
            frame = imutils.resize(frame, width=1280)
            
            # 1. Detect & Analyze Traffic
            vehicle_count, level, color, bounding_boxes, trigger_traffic_alert, duration = analyzer.process_frame(frame, min_contour_area=600)
            
            # 2. Check for Traffic Alert
            if trigger_traffic_alert:
                alerter.trigger_traffic_alert(duration)
                
            # 3. Detect Obstructions / Illegal Parking
            is_obstruction, obstruction_boxes, stationary_info, is_congestion = obstruction_detector.process_frame(bounding_boxes)
            
            if is_obstruction:
                alerter.trigger_parking_alert(frame_count, frame, obstruction_boxes)
                cv2.putText(frame, "ILLEGAL PARKING / OBSTRUCTION", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            if is_congestion:
                cv2.putText(frame, "CONGESTION / SIGNAL DETECTED", (350, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

            # 4. Draw UI
            # We use stationary_info to decide box color
            for info in stationary_info:
                (x, y, w, h) = info['bbox']
                frames_still = info['stationary_count']
                
                if frames_still > 30: # If still for > 1 second (approx)
                    # RED BOX for stationary
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    # Show seconds
                    sec = frames_still // fps
                    cv2.putText(frame, f"{sec}s", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    # BLUE BOX for moving
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Overlay info
            cv2.rectangle(frame, (10, 10), (300, 120), (0, 0, 0), -1)
            cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Traffic Level: {level}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            if level == "HIGH":
                current_duration_sec = analyzer.high_traffic_consecutive_frames // fps
                cv2.putText(frame, f"HIGH Duration: {current_duration_sec}s", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
        # 5. Show
        cv2.imshow("Smart Traffic Monitoring", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
