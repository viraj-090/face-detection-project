from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO("runs/detect/train/weights/best.onnx")
results =model.val(
    data="data/widerface.yaml",
    imgsz=640,
    batch=8,
    plots=True
)

res = results.box
print('precision: ',res.p)
print('recall: ',res.r)
print('mAP50: ',res.map50)
