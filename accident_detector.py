import math

class AccidentDetector:
    def __init__(self, fps=30, stationary_duration=60, min_movement=5, detection_zone_y_min=0):
        # Time to consider a car stationary as an accident (seconds)
        self.anomaly_threshold = fps * stationary_duration  # Total frames it needs to be still
        self.min_movement = min_movement  # Pixel movement threshold to reset counter
        self.detection_zone_y_min = detection_zone_y_min # Only track cars below this 'y' (ignore distant noise)

        self.tracked_objects = {}  # Format: {id: {"centroid": (x, y), "stationary_count": 0, "bbox": (x,y,w,h)}}
        self.next_id = 0

    def process_frame(self, bounding_boxes):
        """Processes current bounding boxes to detect stationary clusters/obstructions."""
        current_centroids = []
        for box in bounding_boxes:
            x, y, w, h = box
            if y > self.detection_zone_y_min:
                cx, cy = x + w // 2, y + h // 2
                current_centroids.append((cx, cy, box))

        # Match to existing tracked objects
        new_tracked_objects = {}
        unmatched_new = []
        
        num_stationary = 0
        num_moving = 0

        for (cx, cy, box) in current_centroids:
            matched = False
            best_id_match = None
            min_dist = float('inf')

            # Find closest tracked object
            for t_id, t_data in self.tracked_objects.items():
                old_cx, old_cy = t_data['centroid']
                dist = math.hypot(cx - old_cx, cy - old_cy)
                
                # If it's close enough, assume it's the same object
                if dist < 50 and dist < min_dist:
                    min_dist = dist
                    best_id_match = t_id

            if best_id_match is not None:
                # Update existing
                t_data = self.tracked_objects[best_id_match]
                stationary_count = t_data['stationary_count']
                
                # Check if it moved
                if min_dist < self.min_movement:
                    stationary_count += 1
                    num_stationary += 1
                else:
                    stationary_count = 0  # Reset if moved
                    num_moving += 1
                
                new_tracked_objects[best_id_match] = {
                    "centroid": (cx, cy),
                    "stationary_count": stationary_count,
                    "bbox": box
                }
                
                # Remove matched id so we don't match it again
                del self.tracked_objects[best_id_match]
                matched = True
            
            if not matched:
                unmatched_new.append((cx, cy, box))
                num_moving += 1 # New objects are considered moving initially

        # Add new objects
        for (cx, cy, box) in unmatched_new:
            new_tracked_objects[self.next_id] = {
                "centroid": (cx, cy),
                "stationary_count": 0,
                "bbox": box
            }
            self.next_id += 1

        self.tracked_objects = new_tracked_objects
        
        # LOGIC: Check for traffic jam/congestion
        total_tracked = num_stationary + num_moving
        is_global_traffic = False
        if total_tracked > 3:
            if (num_stationary / total_tracked) > 0.7:
                is_global_traffic = True
        
        # Prepare list for visualization: (bbox, stationary_seconds)
        stationary_info = []
        for t_id, t_data in self.tracked_objects.items():
            stationary_info.append({
                "bbox": t_data['bbox'],
                "stationary_count": t_data['stationary_count']
            })

        # Trigger logic
        accident_detected = False
        accident_boxes = []
        if not is_global_traffic:
            for t_id, t_data in self.tracked_objects.items():
                if t_data['stationary_count'] > self.anomaly_threshold:
                    accident_detected = True
                    accident_boxes.append(t_data['bbox'])

        return accident_detected, accident_boxes, stationary_info, is_global_traffic

    def reset(self):
        """Clears tracked memory."""
        self.tracked_objects = {}
        self.next_id = 0
