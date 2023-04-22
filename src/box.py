import cv2
class box:
    def __init__(self, x, y, w, h) -> None:
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    
    def draw(self, frame):
        box_frame = frame
        box_frame = cv2.rectangle(box_frame, (int(self.x), int(self.y)), (int(self.w), int(self.h)), (255, 0, 0), 2)
        return box_frame