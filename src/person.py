import cv2


class Person:
    def __init__(self, l_hand=(0, 0), r_hand=(0, 0)) -> None:
        self.l_hand = l_hand
        self.r_hand = r_hand
        self.l_box = box(int(l_hand[0] - 75), int(l_hand[1] + 0),
                         int(l_hand[0] + 75), int(l_hand[1] + 100))
        self.r_box = box(int(r_hand[0] - 75), int(r_hand[1] + 0),
                         int(r_hand[0] + 75), int(r_hand[1] + 100))

    def draw_hands(self, frame):
        cv2.circle(frame, (int(self.l_hand[0]), int(
            self.l_hand[1])), 5, (0, 0, 255), -1)
        cv2.circle(frame, (int(self.r_hand[0]), int(
            self.r_hand[1])), 5, (0, 0, 255), -1)
        return frame

    def draw_box(self, frame):
        box_frame = frame
        box_frame = self.l_box.draw(box_frame, (0, 255, 0))
        box_frame = self.r_box.draw(box_frame, (0, 0, 255))

        return box_frame


def create_person(pose_results):
    new_person = []
    for i in range(pose_results.keypoints.shape[0]):
        l_hand = pose_results.keypoints[i, 10]
        r_hand = pose_results.keypoints[i, 9]

        new_person.append(Person(l_hand, r_hand))

    return new_person


class box:
    def __init__(self, x, y, w, h) -> None:
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def draw(self, frame, colour: tuple):
        box_frame = frame
        box_frame = cv2.rectangle(box_frame, (int(self.x), int(
            self.y)), (int(self.w), int(self.h)), colour, 2)
        return box_frame


def dist_bet_points(x, y):
    x1, y1, x2, y2 = x[0], x[1], y[0], y[1]
    return ((x1-x2)**2 + (y1-y2)**2)**0.5
