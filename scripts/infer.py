from ultralytics import YOLO
import sys

def main(source):
    model = YOLO("runs/detect/train/weights/best.onnx")
    model.predict(
        source=source,
        conf=0.5,
        iou=0.5,
        show=True,   # shows live video/webcam
        save=True
    )

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else 0  
    main(src)
