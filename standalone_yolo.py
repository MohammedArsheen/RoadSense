import cv2
import argparse
import imutils
import time
from traffic_analyzer import TrafficAnalyzer
from accident_detector import AccidentDetector

def run_standalone(video_path):
    print(f"--- INITIALIZING PRECISION VEHICLE-ONLY VISION: {video_path} ---")
    analyzer = TrafficAnalyzer(fps=25)
    obstruction_detector = AccidentDetector(fps=25, stationary_duration=30)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video file {video_path}")
        return

    cv2.namedWindow("PRECISION VEHICLE SURVEILLANCE HUB", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("PRECISION VEHICLE SURVEILLANCE HUB", 1280, 720)

    print("--- VEHICLE-ONLY AI ENGINE LIVE! Press 'q' to exit ---")
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Video Finished. Restarting...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # 1. OPTIMIZE
        frame = imutils.resize(frame, width=1280)
        
        # 2. RUN AI (VEHICLE-ONLY)
        try:
            # analyzer.process_frame already filters for Cars (2), Bikes (3), Buses (5), Trucks (7)
            vehicle_count, level, color, boxes, alerts = analyzer.process_frame(frame)
            _, _, stationary_info, is_congestion = obstruction_detector.process_frame(boxes)
            
            # 3. DRAW TACTICAL HUB
            # DRAW BLUE DETECTION BOXES (FOR VEHICLES ONLY)
            for box in boxes:
                # box comes in format (x, y, w, h)
                x, y, w, h = map(int, box)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1) # Precision Tactical Blue

            # DRAW RED HIGHLIGHTS (STATIONARY VEHICLES)
            for info in stationary_info:
                x, y, w, h = info['bbox']
                f_still = info['stationary_count']
                if f_still > 15:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3) # Focus Warning Red
                    cv2.putText(frame, f"STILL: {f_still//25}s", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # DRAW AI MASTER STATUS
            hub_col = (0, 255, 0) if level == "LOW" else (0, 255, 255) if level == "MEDIUM" else (0, 0, 255)
            cv2.rectangle(frame, (25, 25), (550, 95), (40, 40, 40), -1) # Dark HUD Backdrop
            cv2.putText(frame, f"VEHICLE VISION: {level} | {vehicle_count} UNITS", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, hub_col, 3)
            
            # 4. DISPLAY
            cv2.imshow("PRECISION VEHICLE SURVEILLANCE HUB", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"AI ERROR: {e}")
            continue

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="data/Traffic.mp4")
    args = parser.parse_args()
    
    run_standalone(args.video)
