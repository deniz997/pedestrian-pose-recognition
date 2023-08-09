import socket
import cv2
import numpy as np
from torchvision import transforms
from torch import nn
import torch

from openpose.op_utils import get_keypoints_image_from_data, get_camera_stream_and_display, get_stream_and_display
from video_stream.ra_gcn_command_recognition import RA_GCN, labels_to_learn, Graph, Data_transform, Occlusion_part, \
    Occlusion_time


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


def load_model(path):
    model_stream = 3
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_class = len(labels_to_learn)
    graph = Graph(max_hop=2)
    A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False).to(DEVICE)
    eval_model = RA_GCN((6, 90, 17, 1), num_class, A, 0.5, [5, 2], model_stream).to(DEVICE)
    eval_model = nn.DataParallel(eval_model)
    eval_model.load_state_dict(torch.load(path))
    eval_model.eval()
    return eval_model


def preprocess_data(frames):
    transform = transforms.Compose([
        Data_transform(data_transform=True),
        Occlusion_part([]),
        Occlusion_time(0),
    ])

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = np.zeros((2, 90, 17, 1))
    for frame in frames:
        for i in range(17):
            x[0, frame, i, 0] = frame[i][0]
            x[1, frame, i, 0] = frame[i][1]
    x = transform(x)
    x = torch.from_numpy(x).float()
    x = x[None, :, :, :, :]
    x = x.to(DEVICE)
    return x
if __name__ == '__main__':
    #cap = cv2.VideoCapture('http://192.168.2.206:8000/stream.mjpg')
    cap = cv2.VideoCapture(0)
    path = ''
    #get_camera_stream_and_display()

    frames = []

    while True:
        ret, frame = cap.read()
        #cv2.imshow('Video', frame)

        img, key_points = get_stream_and_display(frame)

        #cv2.imshow('Human Pose Estimation', img)

        if len(frame) < 90:
            frame.add(key_points)
        else:
            eval_model = load_model(path)
            out, _ = eval_model(key_points)
            pred = out.max(1, keepdim=True)[1]
            label = labels_to_learn[pred]

            #send_label_to_pi(label)
            print(f'Send label to pi: {label}')
            frame = []

        if cv2.waitKey(1) == 27:
            exit(0)
