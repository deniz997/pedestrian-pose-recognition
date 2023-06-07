import os
import json
import numpy as np


def get_data(file_dir):
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


# get_data('../data/JAAD_output_JSON')

# example usage
tcg_d, tcg_j = get_data_TCG('../data/TCG/')
print(len(tcg_d))
print(tcg_j.keys())
print(len(tcg_j['sequences']))
print(tcg_d[0][0])
