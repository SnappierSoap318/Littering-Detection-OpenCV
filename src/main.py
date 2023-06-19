import PySimpleGUI as sg
from gui import create_gui, update_gui
import cv2
from ObjectDet import ObjectDetection


def main():

    filename = sg.popup_get_file('Filename to play')
    if filename is None:
        return

    vidFile = cv2.VideoCapture(filename)
    num_frames = vidFile.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = vidFile.get(cv2.CAP_PROP_FPS)

    window = create_gui(num_frames)

    timeout = int(1000/fps)

    slider_elem = window['-SLIDER-']

    cur_frame = 0

    while vidFile.isOpened():
        event, values = window.read(timeout=timeout)

        if event in ['Exit', None, sg.WINDOW_CLOSED]:
            break

        od = ObjectDetection()

        ret, frame = vidFile.read()
        if not ret:
            break

        if int(values['-SLIDER-']) != cur_frame-1:
            cur_frame = int(values['-SLIDER-'])
            vidFile.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)
        slider_elem.update(cur_frame)
        cur_frame += 1

        # Detect objects
        pose_results, trash_results, plate_frame = od.detect(frame)

        persons = pose_results[1]
        trash_frame, boxes = trash_results

        for i in persons:
            i.draw_box(trash_frame)

        update_gui(window, frame, trash_frame, pose_results[0], plate_frame)

        # Check for collision
        if persons and boxes:
            for person in persons:
                if collision(person.l_box, boxes) or collision(person.r_box, boxes):
                    print('Trash in hand!')
                    flag = 0
                else:
                    flag = 1
                    print('No trash in hand')
                    # Save frame to folder
                    cv2.imwrite('defaulters_screenshot/frame_%03d.jpg' %
                                cur_frame, frame)

        if flag == 1:
            break

    window.close()


def collision(box1, box2):
    if box1.x < box2.w and box1.w > box2.x and box1.y < box2.h and box1.h > box2.y:
        return True
    else:
        return False


if __name__ == '__main__':
    main()
