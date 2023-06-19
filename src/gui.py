import PySimpleGUI as sg
import cv2


def create_gui(num_frames):
    sg.theme('DarkAmber')

    layout = [
        [sg.Text('OpenCV Trash and Littering Detection',
                 size=(100, 1), font='Helvetica 20')],
        [sg.Text('Original Frame', size=(25, 1), font='Helvetica 14'),
         sg.Text('Trash Detection', size=(25, 1), font='Helvetica 14'),
         sg.Text('Pose Detection', size=(25, 1), font='Helvetica 14'),
         sg.Text('License Plate : ', size=(15, 1), font='Helvetica 14'),
         sg.Text('', size=(15, 1), font='Helvetica 14', key='plate_num')],
        [sg.Image(key='frame', tooltip='Original Frame'),
         sg.Image(key='trash', tooltip='Trash Detection'),
         sg.Image(key='pose', tooltip='Pose Detection'),
         sg.Image(key='plate', tooltip='License Plate Detection')],
        [sg.Slider(range=(0, num_frames), size=(60, 10),
                   orientation='h', key='-SLIDER-')],
        [sg.Push(), sg.Button('Exit', font='Helvetica 14')]
    ]

    window = sg.Window('Object Detection', layout, finalize=True)
    return window


def update_gui(window, frame, trash, pose, plate):

    # Display frames
    frame = cv2.resize(frame, (270, 480))
    trash = cv2.resize(trash, (270, 480))
    plate_frame = cv2.resize(plate[0], (270, 480))
    pose = cv2.resize(pose, (270, 480))

    window['frame'].update(data=cv2.imencode('.png', frame)[1].tobytes())
    window['trash'].update(data=cv2.imencode('.png', trash)[1].tobytes())
    window['pose'].update(data=cv2.imencode('.png', pose)[1].tobytes())
    window['plate'].update(data=cv2.imencode('.png', plate_frame)[1].tobytes())

    plate_num = ''

    for i in plate[1]:
        plate_num += str(i[1])
 
    window['plate_num'].update(plate_num)
