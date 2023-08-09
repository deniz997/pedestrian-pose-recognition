import socket
import cv2

from openpose.op_utils import get_keypoints_image_from_data, get_camera_stream_and_display, get_stream_and_display


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Strg+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
def send_label_to_pi(label: str):
    ip = "192.168.2.206"  # IP of Raspberry Pi

    # connect to server
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((ip, 8080))
    print("CLIENT: connected")

    # send a message
    msg = label
    client.send(msg.encode())

    # recive a message and print it
    from_server = client.recv(4096).decode()
    print("Recieved: " + from_server)

    # exit
    client.close()


def cmd_classification() -> str:
    pass


if __name__ == '__main__':
    cap = cv2.VideoCapture('http://192.168.2.206:8000/stream.mjpg')
    #get_camera_stream_and_display()

    while True:
        ret, frame = cap.read()
        #cv2.imshow('Video', frame)

        if ret is True:
            img, key_points = get_stream_and_display(frame)
            cv2.imshow('Human Pose Estimation', img)

        #TODO: do classification here
        label = cmd_classification()

        #send_label_to_pi(label)

        if cv2.waitKey(1) == 27:
            exit(0)
