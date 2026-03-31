from ultralytics import YOLO
print("Downloading YOLOv8 model weights...")
model = YOLO('yolov8n.pt')
print("Model loaded successfully!")
