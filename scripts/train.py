
from ultralytics import YOLO
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = YOLO("models/yolov8n.pt")
model.train(
    data="data/widerface.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=device
)
