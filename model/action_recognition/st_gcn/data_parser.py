import os
import json
import numpy as np
import pandas as pd
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

# def convert_jaad_dict_to_df(jaad_dict):
#     updated_list = []
#     for j_dict in jaad_dict:
#         j_dict = {k: map_text_to_scalar(k, v) for k, v in j_dict.items()}
#         updated_list.append(j_dict)
#     data_dict = keypoints_to_dict(updated_list)
#     data = pd.DataFrame(data_dict)
#     data_y = data[['look', 'action', 'cross', 'hand_gesture', 'nod']]
#     data_x = data.drop(columns=['look', 'action', 'cross', 'hand_gesture', 'nod'])
#     df_norm = (data_x - data_x.mean()) / (data_x.max() - data_x.min())
#     return df_norm.astype('float'), data_y.astype('float')
#--------------update:------------------------------------------------------

def MinMaxScaling(data):
    scaler = MinMaxScaler()
    data=scaler.fit_transform(data)
    return data

def convert_jaad_dict_to_df(jaad_dict):
    updated_list = []
    for j_dict in jaad_dict:
        j_dict = {k: map_text_to_scalar(k, v) for k, v in j_dict.items()}
        updated_list.append(j_dict)
    data_dict = keypoints_to_dict(updated_list)
    data = pd.DataFrame(data_dict)
    # data_y = data[['look', 'action',  'hand_gesture', 'nod']]
    # data_x = data.drop(columns=['look', 'action', 'hand_gesture', 'nod'])
    data_y = data[['look', 'action', 'cross', 'hand_gesture', 'nod']]
    data_x = data.drop(columns=['look', 'action', 'cross', 'hand_gesture', 'nod'])
    # df_norm = (data_x - data_x.mean()) / (data_x.max() - data_x.min())
    df_norm = MinMaxScaling(data_x)
    return df_norm, data_y.astype('float')

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


# def convert_format(data):
#     new_data = {}
#
#     # Copy person_id and id
#     new_data["person_id"] = data["person_id"]
#     new_data["id"] = data["attributes"][0]["attribute"]
#
#     # Copy pose_keypoints_2d
#     new_data["pose_keypoints_2d"] = data["pose_keypoints_2d"]
#
#     # Convert attributes list to dictionary
#     attributes_dict = {}
#     for attribute in data["attributes"]:
#         attributes_dict[attribute["name"]] = attribute["attribute"]
#     new_data["attributes"] = attributes_dict
#
#     return new_data



# example usage
#tcg_d, tcg_j = get_data_TCG('../data/TCG/')
#print(len(tcg_d))
#print(tcg_j.keys())
#print(len(tcg_j['sequences']))
#print(tcg_d[0][0])
import json
import numpy as np
import os


def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return np.array(data)


def read_all_jsons(directory):
    # Read all files
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]

    # Create a new list
    all_data = []

    # Read all data from json-files
    for json_file in json_files:
        file_path = os.path.join(directory, json_file)
        data = read_json(file_path)
        all_data.append(data[:,:2])

    return np.array(all_data)

def read_all_folder(main_directory):
    arr = np.empty([0,25,2])
    for root, dirs, files in os.walk(main_directory):
        for dir_name in dirs:
            subdirectory_path = os.path.join(root, dir_name)
            arr = np.concatenate((arr,read_all_jsons(subdirectory_path)),axis=0)
    res = arr
    return res
# def read_folder(base_path,list_of_name):
#     """
#         Concatenate all data which contained by the given list
#
#         Parameters:
#         - base_path:path of files
#         - list_of_name:name of selected training or testing video
#
#         Returns:
#         array: training or testing data
#
#         """
#     arr = np.empty([0,25,2])
#     for name in list_of_name:
#         filepath = os.path.join(base_path, name)
#         arr = np.concatenate((arr,read_all_jsons(filepath)),axis=0)
#     return arr


