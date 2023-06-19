from ultralytics import YOLO

model = YOLO('Models/yolov8n.pt')
results = model.train(data='c:/Users/Adwaith S/Downloads/Trash Detection/data.yaml', epochs = 150)