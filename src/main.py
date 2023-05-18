# Description: Main file for the project

# Import libraries
from ultralytics import YOLO
import cv2
from person import create_person, box


#collision detection between 2 boxes 
def collision(box1, box2):
    if box1.x < box2.w and box1.w > box2.x and box1.y < box2.h and box1.h > box2.y:
        return True
    else:
        return False

# Load model
pose_model = YOLO('./Models/yolov8l-pose.pt')
trash_model = YOLO('./Models/yolov8n-bobby.pt')

# Load video
cap = cv2.VideoCapture('Littering Dataset/1.mp4')

frame_count = 0

while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break

    boxes = box(0, 0, 0, 0)

    # Detect pose
    pose_results = pose_model(frame, stream=True)
    
    # Detect trash
    trash_results = trash_model(frame, stream=True)

    # Display results
    for i in pose_results:
        persons = create_person(i)
        pose_frame = i.plot()

    for i in trash_results:
        trash_frame = i.plot()
        for j, k in enumerate(i.boxes.cls):
            if k == 2:
                boxes = i.boxes.data[j]
                boxes = box(boxes[0], boxes[1], boxes[2], boxes[3])
                boxes.draw(frame)

    # draw hands
    for person in persons:
        person.draw_box(frame)

    # draw boxes
    frame = cv2.resize(frame, (540, 960))
    trash_frame = cv2.resize(trash_frame, (540, 960))
    pose_frame = cv2.resize(pose_frame, (540, 960))

    # Display frame
    cv2.imshow('frame', frame)
    cv2.imshow('trash', trash_frame)
    cv2.imshow('pose', pose_frame)

    # Check for collision
    for person in persons:
        if collision(person.l_box, boxes) or collision(person.r_box, boxes):
            print('Trash in hand!')
        else:
            print('No trash in hand')
            # save frame to folder
            cv2.imwrite('defaulters_screenshot/frame_%03d.jpg' %
                        frame_count, frame)
            
            cv2.waitKey(0)
            break

    key = cv2.waitKey(1)
    # Quit if 'q' is pressed
    if key == ord('q'):
        break
    # Pause if 'p' is pressed
    if key == ord('p'):
        cv2.waitKey(0)

    frame_count += 1
cap.release()
cv2.destroyAllWindows()

