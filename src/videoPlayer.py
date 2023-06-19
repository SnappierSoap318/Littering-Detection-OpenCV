import cv2
from threading import Thread
from ObjectDet import ObjectDetection
from person import box

class VideoPlayer(Thread):
    def __init__(self, video_path):
        Thread.__init__(self)
        self.video_path = video_path
        self.paused = False
        self.screenshot = False
        self.exit = False

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        flag = 0
        od = ObjectDetection()

        while True:
            if self.exit:
                break

            if not self.paused:
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect objects
                pose_results, trash_results, plate_frame = od.detect(frame)
                
                persons = pose_results[1]
                trash_frame, boxes = trash_results

                # Display frames
                frame = cv2.resize(frame, (540, 960))
                trash_frame = cv2.resize(trash_frame, (540, 960))
                plate_frame = cv2.resize(plate_frame, (540, 960))

                # Check for collision
                if persons:
                    for person in persons:
                        if collision(person.l_box, boxes) or collision(person.r_box, boxes):
                            print('Trash in hand!')
                            flag = 0
                        else:
                            flag = 1
                            print('No trash in hand')
                            # Save frame to folder
                            if self.screenshot:
                                cv2.imwrite('defaulters_screenshot/frame_%03d.jpg' % frame_count, frame)
                                self.screenshot = False

                if flag == 1:
                    break

                frame_count += 1

            key = cv2.waitKey(1)
            # Quit if 'q' is pressed
            if key == ord('q'):
                self.exit = True
                break
            # Pause if 'p' is pressed
            if key == ord('p'):
                self.paused = True
                cv2.waitKey(0)
                self.paused = False

        cap.release()
        cv2.destroyAllWindows()

def collision(box1, box2):
    if box1.x < box2.w and box1.w > box2.x and box1.y < box2.h and box1.h > box2.y:
        return True
    else:
        return False

