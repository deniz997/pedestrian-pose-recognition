import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc
from matplotlib.animation import FuncAnimation
from sklearn.preprocessing import MinMaxScaler


def keypoints_to_dict(jaad_dict):
    ret_list = []
    for data in jaad_dict:
        keypoint_dict = {}
        pose_keypoints = data['pose_keypoints_2d']
        for i in range(0, len(pose_keypoints)-1, 3):
            keypoint_dict['x_' + str(i)] = pose_keypoints[i]
            keypoint_dict['y_' + str(i)] = pose_keypoints[i + 1]
        keypoint_dict['look'] = data['look']
        keypoint_dict['action'] = data['action']
        keypoint_dict['cross'] = data['cross']
        keypoint_dict['hand_gesture'] = data['hand_gesture']
        keypoint_dict['nod'] = data['nod']
        ret_list.append(keypoint_dict)
    return ret_list


def map_text_to_scalar(label_type, value):
    """
    Maps a text label in XML file to scalars
    :param label_type: The label type
    :param value: The text to be mapped
    :return: The scalar value
    """
    map_dic = {'occlusion': {'none': 0, 'part': 1, 'full': 2},
               'action': {'standing': 0, 'walking': 1},
               'nod': {'__undefined__': 0, 'nodding': 1},
               'look': {'not-looking': 0, 'looking': 1},
               'hand_gesture': {'__undefined__': 0, 'greet': 1, 'yield': 1,
                                'rightofway': 1, 'other': 1},
               'reaction': {'__undefined__': 0, 'clear_path': 1, 'speed_up': 2,
                            'slow_down': 3},
               'cross': {'not-crossing': 0, 'crossing': 1, 'irrelevant': -1},
               'age': {'child': 0, 'young': 1, 'adult': 2, 'senior': 3},
               'designated': {'ND': 0, 'D': 1},
               'gender': {'n/a': 0, 'female': 1, 'male': 2},
               'intersection': {'no': 0, 'yes': 1},
               'motion_direction': {'n/a': 0, 'LAT': 1, 'LONG': 2},
               'traffic_direction': {'OW': 0, 'TW': 1},
               'signalized': {'n/a': 0, 'NS': 1, 'S': 2},
               'vehicle': {'stopped': 0, 'moving_slow': 1, 'moving_fast': 2,
                           'decelerating': 3, 'accelerating': 4},
               'road_type': {'street': 0, 'parking_lot': 1, 'garage': 2},
               'traffic_light': {'n/a': 0, 'red': 1, 'green': 2}}

    if type(value) is list or label_type not in map_dic:
        return value

    return map_dic[label_type][value]


def convert_jaad_dict_to_df(jaad_dict):
    updated_list = []
    for j_dict in jaad_dict:
        j_dict = {k: map_text_to_scalar(k, v) for k, v in j_dict.items()}
        updated_list.append(j_dict)
    data_dict = keypoints_to_dict(updated_list)
    data = pd.DataFrame(data_dict)
    data_y = data[['look', 'action', 'hand_gesture', 'nod']]
    data_x = data.drop(columns=['look', 'action', 'cross', 'hand_gesture', 'nod'])
    scaler = MinMaxScaler()
    scaler.fit(data_x)
    x_st = scaler.transform(data_x)
    return x_st, data_y.astype('float')


def get_jaad_data(file_dir):
    """
    Will read JSON files from the provided file directory and subdirectories and return a list of dicts
    :param file_dir: directory where to look for input JSON files
    :return: list of dicts converted from JSON files
    """
    print('Starting to read JAAD json files!')
    json_list = []

    for dirpath, dirnames, filenames in os.walk(file_dir):
        jaad_list = [pos_json for pos_json in filenames if pos_json.endswith('.json')]
        for js in jaad_list:
            with open(os.path.join(dirpath, js)) as json_file:
                json_list.append(json.load(json_file))

    return json_list


def get_data_tcg(file_dir: str) -> (np.ndarray, dict):
    """
    Read in the TCG dataset and its annotation
    :param file_dir: directory where data and annotations from TCG are stored
    :return: Data as np array and annotations as dict
    """
    tcg_data = np.load(file_dir + 'tcg_data.npy', allow_pickle=True)

    with open(file_dir + 'tcg.json') as json_file:
        tcg_json = json.load(json_file)

    return tcg_data, tcg_json


def get_data_hri(file_dir: str) -> (np.ndarray, list):
    """
    Reads in the whole HRI dataset and creates a data array and their corresponding labels
    :param file_dir: Path to root directory of HRI dataset (example: '../data/HRI_gestures')
    :return: (data, labels) -> both as np.ndarray
    """
    action_class = {'A001': 'Stop', 'A002': 'Go Right', 'A003': 'Go Left', 'A004': 'Come Here', 'A005': 'Follow me',
                    'A006': 'Go Away', 'A007': 'Agree', 'A008': 'Disagree', 'A009': 'Go there', 'A010': 'Get Attention',
                    'A011': 'Be Quiet', 'A012': 'Dont Know', 'A013': 'Turn Around', 'A014': 'Take This',
                    'A015': 'Pick Up', 'A016': 'Standing Still', 'A017': 'Being Seated', 'A018': 'Walking Towards',
                    'A019': 'Walking Away', 'A020': 'Talking on Phone'}
    joint_dict = {'Nose': 0, 'LEye': 1, 'REye': 2, 'LEar': 3, 'REar': 4, 'LShoulder': 5, 'RShoulder': 6, 'LElbow': 7,
                  'RElbow': 8, 'LWrist': 9, 'RWrist': 10, 'LHip': 11, 'RHip': 12, 'LKnee': 13, 'RKnee': 14,
                  'LAnkle': 15, 'RAnkle': 16}
    hri_dataset = []
    hri_labels = []
    for dirpath, dirnames, filenames in os.walk(file_dir):
        for file in filenames:
            with open(dirpath + '/' + file, 'r') as f:
                content = f.readlines()
                f.close()
            frames = int(content[0])
            file_data = []
            file_label = []
            cur_label = action_class[file[:4]]
            next_position = 1
            for i in range(frames):
                joints = np.zeros((17, 5), dtype='float64')
                joint_nr = int(content[next_position])
                next_position += 1
                for joint in range(joint_nr):
                    v = content[next_position].split(' ')
                    label = v[0]
                    joints[joint_dict[label]] = v[1:]
                    next_position += 1
                file_data.append(joints)
                file_label.append(cur_label)
            hri_dataset.append(np.array(file_data))
            hri_labels.append(cur_label)

    return hri_dataset, hri_labels


def load_data_hri(file_dir: str) -> (np.ndarray, np.ndarray):
    """
    Loads the HRI_gestures dataset from the saved .npy files
    :param file_dir: Path to the directory both files are saved
    :return: data and labels as np.ndarray
    """
    hri_data = np.load(file_dir + 'HRI_data.npy')
    hri_labels = np.load(file_dir + 'HRI_labels.npy')

    return hri_data, hri_labels


def transform_to_hri(keypoints):
    """
    Transforms keypoints retrieved by OpenPose (25-keypoints) to keypoints matching the
    HRI gestures dataset (17 keypoints) and normalizing the data.
    NOTE: The data is intended to work for our own recorded data, therefore the resolution is hardcoded.
    :param keypoints:   Array with length 25 containing keypoints from OpenPose
    :results:           Normalized keypoints matching the HRI dataset
    """
    hri_dict = {'Nose': 0, 'LEye': 1, 'REye': 2, 'LEar': 3, 'REar': 4, 'LShoulder': 5, 'RShoulder': 6, 'LElbow': 7,
                'RElbow': 8, 'LWrist': 9, 'RWrist': 10, 'LHip': 11, 'RHip': 12, 'LKnee': 13, 'RKnee': 14,
                'LAnkle': 15, 'RAnkle': 16}
    op_dict = {'Nose': 0, 'Neck': 1, 'RShoulder': 2, 'RElbow': 3, 'RWrist': 4, 'LShoulder': 5, 'LElbow': 6,
               'LWrist': 7, 'MidHip': 8, 'RHip': 9, 'RKnee': 10, 'RAnkle': 11, 'LHip': 12, 'LKnee': 13, 'LAnkle': 14,
               'REye': 15, 'LEye': 16, 'REar': 17, 'LEar': 18, 'LBigToe': 19, 'LSmallToe': 20, 'LHeel': 21,
               'RBigToe': 22, 'RSmallToe': 23, 'RHeel': 24, 'Background': 25}
    hri_keypoints = np.zeros((17, 2), dtype='float64')
    for key, index in hri_dict.items():
        hri_keypoints[index] = keypoints[op_dict[key]][:2]
    # print((hri_keypoints - np.mean(hri_keypoints, axis=0)) / [1080, 1920])
    hri_keypoints = (hri_keypoints - np.mean(hri_keypoints, axis=0)) / [1080, 1920] * 2
    return hri_keypoints


def show_skeleton(skeleton, reverse=True, as_video=False, save_with_name="", frame=0):
    """
    :param skeleton:        (frame_count, 17, num of dimensions, e.g. 2)
    :param reverse:         The plot is upside down, set it to False when the original plot is needed
    :param as_video:        Generates and displays a video from the skeleton sequences
    :param save_with_name:  Saves a video from the skeleton sequences with the given file name,
                            ignored if as_video is set to False
    :param frame:           the frame number to plot the skeleton from,
                            should not exceed the total frame count of the sequence
    """

    frame_count = skeleton.shape[0]  # Total frame count
    if frame_count <= frame:
        print("Frame count exceeded!")
        return

    # Create a new figure
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Define the connections between keypoints (assuming a specific order)
    connections = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8),
                   (7, 9), (8, 10), (5, 11), (6, 12), (5, 6), (11, 12), (11, 13),
                   (12, 14), (13, 15), (14, 16)]

    def update_frame(updated_frame):
        ax.cla()
        # Example skeleton data
        skeleton_data = np.array(skeleton)[
            updated_frame]  # List of skeleton data, each element is a matrix of shape (Frame count, 17,
        # 2) representing the 2D locations

        if reverse:
            skeleton_data[:, 1] = skeleton_data[:, 1] * -1

        # Plot keypoints
        ax.scatter(skeleton_data[:, 0], skeleton_data[:, 1], marker='o')

        # Plot connections
        for connection in connections:
            ax.plot([skeleton_data[connection[0], 0], skeleton_data[connection[1], 0]],
                    [skeleton_data[connection[0], 1], skeleton_data[connection[1], 1]])

        # Add keypoint labels
        for j, point in enumerate(skeleton_data):
            ax.text(point[0], point[1], str(j), fontsize=8)
        # Set plot limits and labels
        ax.set_xlim([-1, 1])  # Specify the width of your image
        ax.set_ylim([-1, 1])  # Specify the height of your image
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    if as_video:
        rc('animation', html='jshtml')
        animation = FuncAnimation(fig, update_frame, frames=frame_count, interval=200)
        if save_with_name != "":
            animation.save(save_with_name + ".mp4")
        return animation
    else:
        update_frame(frame)
        plt.show()
