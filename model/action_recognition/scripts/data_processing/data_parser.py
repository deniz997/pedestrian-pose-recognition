import json
import numpy as np
import os

from model.action_recognition.scripts.data_processing import data_preprocessing


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
        arr = data_preprocessing.apply_scaler_to_dataset(data_preprocessing.centering_data(read_all_jsons(file_path)))
        all_data.append(arr.reshape(arr.shape[0],-1))

    return np.array(all_data)

# def read_all_folder(main_directory):
#     arr = np.empty([0,25,2])
#     for root, dirs, files in os.walk(main_directory):
#         for dir_name in dirs:
#             subdirectory_path = os.path.join(root, dir_name)
#             arr = np.concatenate((arr,read_all_jsons(subdirectory_path)),axis=0)
#     res = arr
#     return res
def read_folder(base_path,list_of_name):
    """
        Concatenate all data which contained by the given list

        Parameters:
        - base_path:path of files
        - list_of_name:name of selected training or testing video

        Returns:
        array: training or testing data

        """
    arr = np.empty([0,25,2])
    for name in list_of_name:
        filepath = os.path.join(base_path, name)
        arr = np.concatenate((arr,read_all_jsons(filepath)),axis=0)
    return arr


def read_selected_folder(base_path,list_of_name):
    """
        Concatenate all data which contained by the given list

        Parameters:
        - base_path:path of files
        - list_of_name:name of selected training or testing video

        Returns:
        list: contains array of training or testing data

        """
    lst = []
    for name in list_of_name:
        filepath = os.path.join(base_path,name)
        #centering and scaling
        arr = data_preprocessing.apply_scaler_to_dataset(data_preprocessing.centering_data(read_all_jsons(filepath)))
        #shape of arr:(N,25,2)-->(N,50)
        lst.append(arr.reshape(arr.shape[0],-1))
    return lst

# 1. Read txt file
def read_txt_file(txt_path):
    with open(txt_path, 'r') as f:
        filenames = f.readlines()
    return [name.strip() for name in filenames]
