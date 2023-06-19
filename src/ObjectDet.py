from ultralytics import YOLO
import easyocr
import cv2
from person import box, create_person


class ObjectDetection:
    def __init__(self):
        self.pose_model = YOLO('./Models/yolov8n-pose.pt')
        self.trash_model = YOLO('./Models/yolov8n-trash.pt')
        self.plate_model = YOLO('./Models/yolov8n-plate.pt')
        self.reader = easyocr.Reader(['en'])

    def detect_pose(self, frame):
        pose_result = self.pose_model(frame, stream=True)
        return self.pose_reader(pose_result)

    def detect_trash(self, frame):
        trash_result = self.trash_model(frame, stream=True)
        return self.trash_reader(trash_result)

    def detect_plate(self, frame):
        plate_result = self.plate_model(frame, stream=True)
        return self.plate_reader(plate_result)

    def detect(self, frame):
        return [self.detect_pose(frame), self.detect_trash(frame), self.detect_plate(frame)]

    def plate_reader(self, frame):
        plate_text = ''
        for i in frame:
            plate_frame = i.plot()
            if i.boxes.data.shape[0] != 0:

                x = i.boxes.xywh[0, 0]
                y = i.boxes.xywh[0, 1]
                w = i.boxes.xywh[0, 2]
                h = i.boxes.xywh[0, 3]

                # Extract text from plate using EasyOCR
                img_x = int(y-h)
                img_w = int(y+h)
                img_y = int(x-w)
                img_h = int(x+w)

                plate_img = i.orig_img[img_x:img_w, img_y:img_h]
                if plate_img.shape[1] !=0:
                    plate_text = self.reader.readtext(plate_img)
        return [plate_frame, plate_text]

    def trash_reader(self, frame):
        boxes = None
        for i in frame:
            trash_frame = i.plot()
            if i.boxes.data.shape[0] != 0:
                for j, k in enumerate(i.boxes.cls):
                    if k == 2:
                        boxes = i.boxes.data[j]
                        boxes = box(boxes[0], boxes[1], boxes[2], boxes[3])
                        boxes.draw(trash_frame, (0, 255, 0))

        return [trash_frame, boxes]

    def pose_reader(self, frame):
        for i in frame:
            pose_frame = i.plot()
            if i.keypoints is not None:
                persons = create_person(i)
        return [pose_frame, persons]
