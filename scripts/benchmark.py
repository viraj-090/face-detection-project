import time
import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.onnx")
video = cv2.VideoCapture(0) 
times =[]
for _ in range(100):
    ret, frame = video.read()
    if not ret:
        break
    start_time = time.time()
    model(frame)
    end_time = time.time()
    times.append(end_time - start_time)

video.release()
print("Avg Latency:", sum(times)/len(times))
print("FPS:", 1/(sum(times)/len(times)))