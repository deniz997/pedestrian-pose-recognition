import os
import json
import numpy as np
import sklearn.preprocessing
from sklearn.preprocessing import normalize
from openpose import op_utils
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.animation import FuncAnimation


def get_JAAD_data(file_dir):
    """
    Will read JSON files from the provided file directory and subdirectories and return a list of dicts
    :param file_dir: directory where to look for input JSON files
    :return: list of dicts converted from JSON files
    """
    print('Starting to read JAAD json files!')
    json_list = []

    for dirpath, dirnames, filenames in os.walk(file_dir):
        JAAD_list = [pos_json for pos_json in filenames if pos_json.endswith('.json')]
        for js in JAAD_list:
            with open(os.path.join(dirpath, js)) as json_file:
                json_list.append(json.load(json_file))

    return json_list


def get_data_TCG(file_dir: str) -> (np.ndarray, dict):
    """
    Read in the TCG dataset and its annotation
    :param file_dir: directory where data and annotations from TCG are stored
    :return: Data as np array and annotations as dict
    """
    tcg_data = np.load(file_dir + 'tcg_data.npy', allow_pickle=True)

    with open(file_dir + 'tcg.json') as json_file:
        tcg_json = json.load(json_file)

    return tcg_data, tcg_json


def get_data_HRI(file_dir: str) -> (np.ndarray, list):
    '''
    Reads in the whole HRI dataset and creates a data array and their corresponding labels
    :param file_dir: Path to root directory of HRI dataset (example: '../data/HRI_gestures')
    :return: (data, labels) -> both as np.ndarray
    '''
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


def load_data_HRI(file_dir: str) -> (np.ndarray, np.ndarray):
    '''
    Loads the HRI_gestures dataset from the saved .npy files
    :param file_dir: Path to the directory both files are saved
    :return: data and labels as np.ndarray
    '''
    hri_data = np.load(file_dir + 'HRI_data.npy')
    hri_labels = np.load(file_dir + 'HRI_labels.npy')

    return hri_data, hri_labels


def transform_to_HRI(keypoints):
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
    ###
    # Skeleton: (frame_count, 17, num of dimentions, e.g. 2)
    # reverse: The plot is upside down, set it to False when the original plot is needed
    # as_video: Generates and displays a video from the skeleton sequences
    # save_with_name: Saves a video from the skeleton sequences with the given file name,
    #                 ignored if as_video is set to False
    # frame: the frame number to plot the skeleton from,
    #         should not exceed the total frame count of the sequence
    ###

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


# example usage
# data = np.load('../data/hri_keypoints_robert.npy', allow_pickle=True)
# show_skeleton(data[0][0][None])

hri_labels = ['Come Here', 'Follow Me', 'Follow Me', 'Follow Me', 'Follow Me',
              'Get Attention', 'Get Attention', 'Get Attention', 'Go Left', 'Go Left',
              'Go Right', 'Go Right', 'Standing Still', 'Standing Still', 'Standing Still',
              'Stop', 'Stop', 'Stop', 'Stop', 'Stop']
np.save('../data/hri_labels_robert.npy', np.array(hri_labels, dtype='str'))
