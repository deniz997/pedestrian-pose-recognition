import csv
import math
import os
import json
import re

from model.action_recognition.scripts.xml_to_dict import xml_to_dict
from pathlib import Path


def infer_frame_number(filename: str):
    pattern = r'.*_(\d{12})_.*'  # Assuming frame number is always 12 digits
    match = re.match(pattern, filename)
    if match:
        frame_number = match.group(1)
        return int(frame_number)
    else:
        raise ValueError('Invalid file name format. Cannot infer frame number.')


def read_xml_file(path: str):
    # Reading the data inside the xml
    # file to a variable under the name
    # data
    with open(path, 'r') as f:
        data = f.read()
    return data


def get_pedestrians_from_dict(annotations: dict):
    if 'track' not in annotations:
        return {}
    tracks = annotations['track']
    pedestrians = {}
    if type(tracks) is list:
        for track in tracks:
            label = track['label']
            if label == "pedestrian":
                boxes = track['box']
                first_frame_for_box = boxes[0]
                attributes = first_frame_for_box['attribute']
                pedestrian_id = attributes[0]['attribute']
                pedestrians[pedestrian_id] = track
    else:
        label = tracks['label']
        if label == "pedestrian":
            boxes = tracks['box']
            first_frame_for_box = boxes[0]
            attributes = first_frame_for_box['attribute']
            pedestrian_id = attributes[0]['attribute']
            pedestrians[pedestrian_id] = tracks

    return pedestrians


def get_pedestrian_labels_from_frame(pedestrian: dict, frame_number: int):
    for frame_infos in pedestrian['box']:
        if frame_number == int(frame_infos['frame']):
            return frame_infos['attribute']
    raise Exception("Frame number not found")


def check_if_in_box(data, box):
    xbr = float(box['xbr'])
    xtl = float(box['xtl'])
    ybr = float(box['ybr'])
    ytl = float(box['ytl'])
    person_list = []
    for person in data['people']:
        keypoints = person['pose_keypoints_2d']
        i = 0
        j = 1
        number_of_hits = 0
        for z in range(len(keypoints)):
            if i == len(keypoints):
                break
            xp = keypoints[i]
            yp = keypoints[j]
            if xtl <= xp and xp <= xbr and ytl <=yp and yp <= ybr :
                number_of_hits +=1
            i += 3
            j += 3
        person_list.append(number_of_hits)
    max_value = max(person_list)
    index = person_list.index(max_value)
    found_pedestrian = data['people'][index]
    return found_pedestrian


def attributes_to_dict(person_in_box, attribute_list):
    attribute_dict = {}
    for attribute in attribute_list:
        person_in_box[attribute['name']] = attribute['attribute']
    return attribute_dict


def merge_jaad_with_labels(data: dict, box):
    person_in_box = check_if_in_box(data, box)
    person_in_box['attributes'] = attributes_to_dict(person_in_box, box['attribute'])
    return person_in_box



def save_data(update_data: dict, output_folder: str, filename: str):
    path = output_folder / filename
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as outfile:
        json.dump(update_data, outfile)


if __name__ == "__main__":
    filenames_without_ending = ['video_0173', 'video_0304', 'video_0276', 'video_0083', 'video_0021', 'video_0328',
                                'video_0255', 'video_0185', 'video_0097', 'video_0273', 'video_0090', 'video_0162',
                                'video_0227', 'video_0260', 'video_0272', 'video_0287', 'video_0186', 'video_0126',
                                'video_0230', 'video_0140', 'video_0037', 'video_0289', 'video_0341', 'video_0085',
                                'video_0010', 'video_0232', 'video_0284', 'video_0222', 'video_0137', 'video_0158',
                                'video_0079', 'video_0169', 'video_0038', 'video_0292', 'video_0280', 'video_0246',
                                'video_0256', 'video_0229', 'video_0239', 'video_0011', 'video_0243', 'video_0312',
                                'video_0261', 'video_0160', 'video_0271', 'video_0004', 'video_0241', 'video_0088',
                                'video_0316', 'video_0045', 'video_0017', 'video_0337', 'video_0339', 'video_0167',
                                'video_0023', 'video_0262', 'video_0054', 'video_0192', 'video_0008', 'video_0319',
                                'video_0056', 'video_0238', 'video_0170', 'video_0009', 'video_0073', 'video_0285',
                                'video_0042', 'video_0030', 'video_0108', 'video_0198', 'video_0268', 'video_0176',
                                'video_0046', 'video_0300', 'video_0096', 'video_0070', 'video_0047', 'video_0254',
                                'video_0265', 'video_0248', 'video_0035', 'video_0205', 'video_0195', 'video_0166',
                                'video_0193', 'video_0275', 'video_0050', 'video_0207', 'video_0041', 'video_0091',
                                'video_0322', 'video_0329', 'video_0294', 'video_0003', 'video_0044', 'video_0104',
                                'video_0318', 'video_0118', 'video_0163', 'video_0161', 'video_0032', 'video_0252',
                                'video_0197', 'video_0086', 'video_0102', 'video_0120', 'video_0084', 'video_0129',
                                'video_0068', 'video_0168', 'video_0217', 'video_0308', 'video_0333', 'video_0024',
                                'video_0288', 'video_0007', 'video_0259', 'video_0060', 'video_0066', 'video_0211',
                                'video_0123', 'video_0321', 'video_0101', 'video_0012', 'video_0115', 'video_0212',
                                'video_0027', 'video_0100', 'video_0155', 'video_0082', 'video_0107', 'video_0315',
                                'video_0057', 'video_0099', 'video_0295', 'video_0048', 'video_0286', 'video_0218',
                                'video_0061', 'video_0346', 'video_0014', 'video_0181', 'video_0196']



    video_files = [f + ".mp4" for f in filenames_without_ending]
    annotation_files = [f + ".xml" for f in filenames_without_ending]

    dir_jaad = Path("C:/Users/max00/Documents/PoseRecognition/pedestrian-pose-recognition/data/JAAD_output_JSON")
    path_labels = Path("C:/Users/max00/Downloads/JAAD-JAAD_2.0/JAAD-JAAD_2.0/annotations/")
    output_folder = Path("C:/Users/max00/Documents/PoseRecognition/pedestrian-pose-recognition/data/JAAD_JSON_Labels/")
    pedestrian_new = []
    for video_name in filenames_without_ending:
        annotation_file = path_labels / (video_name + ".xml")
        annotations = xml_to_dict(read_xml_file(annotation_file))
        pedestrian = get_pedestrians_from_dict(annotations)
        pedestrian = list(pedestrian.values())[0]
        for box in pedestrian['box']:
            frame = box['frame']
            xbr = box['xbr']
            xtl = box['xtl']
            ybr = box['ybr']
            ytl = box['ytl']
            json_file = video_name + "_" + frame.zfill(12) + "_" + 'keypoints' + ".json"
            f = open(dir_jaad / video_name / json_file)
            data = json.load(f)
            if len(data['people']) == 0:
                continue
            p_new = merge_jaad_with_labels(data, box)
            pedestrian_new.append(p_new)
            save_data(p_new, output_folder / video_name, json_file)