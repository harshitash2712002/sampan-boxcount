from PIL import Image
from ultralytics import YOLO
import cv2
import sys

# Load a pretrained YOLOv8n model


if(sys.argv[1]):
    if(sys.argv[1]=="resume"):
        model = YOLO('runs/detect/train/weights/last.pt', verbose=True)
        model.train(resume=True)
else:
    model = YOLO('yolov8n.pt', verbose=True)
    model.train(data="dataset/package_detection_dataset/data.yaml", epochs=30, imgsz=256, save=True)
