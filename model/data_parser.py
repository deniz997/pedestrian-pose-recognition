import os
import json
import numpy as np
from sklearn.preprocessing import normalize
from openpose import op_utils


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

# example usage
#data, lab = load_data_HRI('data/HRI_gestures/')
#print(data.shape)
#print(lab.shape)
