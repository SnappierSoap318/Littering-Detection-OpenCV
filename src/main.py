# Description: Main file for the project

# Import libraries
from ultralytics import YOLO
import cv2
from person import create_person
from detect import collision
from box import box

# Load model
pose_model = YOLO('./Models/yolov8l-pose.pt')
trash_model = YOLO('./Models/yolov8m-bobby.pt')

# Load video
cap = cv2.VideoCapture('Littering Dataset/2.mp4')

frame_count = 0

while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break

    # Detect pose
    pose_results = pose_model(frame,stream=True)

    # Detect trash
    trash_results = trash_model(frame, stream=True)

    # Display results
    for i in pose_results:
        person = create_person(i)

    for i in trash_results:
        for j, k in enumerate(i.boxes.cls):
            if k == 1:
                boxes = i.boxes.data[j]
                boxes = box(boxes[0], boxes[1], boxes[2], boxes[3])
        boxes.draw(frame)

    #draw hands
    person.draw_box(frame)
    
    #draw boxes
    frame = cv2.resize(frame, (540, 960))

    # Display frame
    cv2.imshow('frame', frame)
    
    # Check for collision
    if collision(person.l_box, boxes) or collision(person.r_box, boxes):
        print('Trash in hand!')
    else:
        print('No trash in hand')
        # save frame to folder
        cv2.imwrite('defaulters_screenshot/frame_%03d.jpg' % frame_count, frame)
        cv2.waitKey(0)
        
        break

    key = cv2.waitKey(1)
    # Quit if 'q' is pressed
    if key == ord('q'):
        break
    #Pause if 'p' is pressed
    if key == ord('p'):
        cv2.waitKey(0)

    frame_count += 0
cap.release()
cv2.destroyAllWindows()