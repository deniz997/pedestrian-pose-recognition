import json
import os
from pathlib import Path

import cv2
import numpy as np

from openpose.op_utils import get_keypoints_from_video

# File written by Maximilian Bartels

def convert_ndarrays_to_arrays(ndarray_list):
    array_list = [np.array(ndarray) for ndarray in ndarray_list]
    return array_list

def process_file_content(file, dir, filename):
    # Replace this with your actual processing logic
    # For demonstration purposes, this code just prints the content
    out_dir = os.path.join(dir, 'out')
    out_file = os.path.join(out_dir, filename)
    f_name = filename.replace('.avi', '')
    ret = get_keypoints_from_video(file, save_video=True, output_file=out_file)
    i = 1
    for frame in ret:
        json_file = str(i) + f_name + '.json'
        with open(os.path.join(out_dir, json_file), "w") as json_file:
            json.dump(frame[0].tolist(), json_file)
        i += 1

def process_files_in_directory(directory):
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            process_file_content(os.path.join(directory, filename), directory, filename)

if __name__ == '__main__':
    directory_path = "C:/Users/max00/Downloads/handwaving"
    process_files_in_directory(directory_path)

