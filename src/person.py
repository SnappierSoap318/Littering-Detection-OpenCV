import cv2
from box import box

class person:
    def __init__(self, l_hand = (0, 0), r_hand = (0, 0)) -> None:
        self.l_hand = l_hand
        self.r_hand = r_hand
        self.l_box = box(int(self.l_hand[0] - 75), int(self.l_hand[1] + 25), int(self.l_hand[0] + 75), int(self.l_hand[1] + 125))
        self.r_box = box(int(self.r_hand[0] - 75), int(self.r_hand[1] + 25) , int(self.r_hand[0] + 75), int(self.r_hand[1] + 125))
        
    def draw_hands(self, frame):
        cv2.circle(frame, (int(self.l_hand[0]), int(self.l_hand[1])), 5, (0, 0, 255), -1)
        cv2.circle(frame, (int(self.r_hand[0]), int(self.r_hand[1])), 5, (0, 0, 255), -1)
        return frame

    def draw_box(self, frame):
        box_frame = frame
        box_frame = cv2.rectangle(box_frame, (self.l_box.x, self.l_box.y), (self.l_box.w, self.l_box.h), (0, 255, 0), 2)
        box_frame = cv2.rectangle(box_frame, (self.r_box.x, self.r_box.y), (self.r_box.w, self.r_box.h),(0, 255, 0), 2)
        return box_frame

def create_person(pose_results):    
    l_hand = pose_results.keypoints[0, 10]
    r_hand = pose_results.keypoints[0, 9]
    new_person = person(l_hand, r_hand)
    return new_person