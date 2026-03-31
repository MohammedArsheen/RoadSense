import cv2
import numpy as np
import os

if not os.path.exists("data"):
    os.makedirs("data")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('data/demo_video.mp4', fourcc, 30.0, (800, 600))

# Create 100 frames
for i in range(100):
    frame = np.ones((600, 800, 3), dtype=np.uint8) * 100
    # draw a moving "car"
    x = int(100 + i * 5)
    y = 300
    cv2.rectangle(frame, (x, y), (x+50, y+30), (0, 0, 255), -1)
    # another moving car below
    x2 = int(600 - i * 4)
    y2 = 400
    cv2.rectangle(frame, (x2, y2), (x2+40, y2+25), (255, 0, 0), -1)
    
    # an accident car (stationary)
    cv2.rectangle(frame, (400, 500), (450, 530), (0, 255, 0), -1)
    
    out.write(frame)

out.release()
print("Synthetic video created.")
