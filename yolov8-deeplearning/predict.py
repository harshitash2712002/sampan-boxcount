from PIL import Image
from ultralytics import YOLO
import cv2
import sys

# Load a pretrained YOLOv8n model
model = YOLO('weights/yolov8-box-best.pt', verbose=True)

input = cv2.imread(sys.argv[1])

# Run inference on image
results = model(input)  # results list

print("Number of Objects: ", len(results[0]))

frame = results[0].plot()

# Display the annotated frame
cv2.imwrite("result.jpg", frame)
cv2.imshow("Detections", frame)
cv2.waitKey()
cv2.destroyAllWindows() 
