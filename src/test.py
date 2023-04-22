import cv2
from ultralytics import YOLO
# Load model
model = YOLO('./Models/yolov8m-bobby.pt')

#load image
img = cv2.imread('D:/Coding/PROJECTS/Bobby/Datasets/Bobby/data/images/train/frame598.jpg')

# Detect trash
results = model(img, stream=True)

# Display results
for i in results:
    print(i.boxes.xywh)
    print(i.boxes.cls)
    # for j in i.boxes.xywh:
    j = i.boxes.xywh[0]
    cv2.rectangle(img, (int(j[0]), int(j[1])), (int(j[2]), int(j[3])), (0, 0, 255), 2)

#display image with boxes
img = cv2.resize(img, (540, 960))
cv2.imshow('frame', img)
cv2.waitKey(0)
cv2.destroyAllWindows()