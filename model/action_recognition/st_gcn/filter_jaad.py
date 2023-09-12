import json
import os
# from model.action_recognition.scripts.merge_jaad_with_labels import read_xml_file, get_pedestrians_from_dict
# from model.action_recognition.scripts.xml_to_dict import xml_to_dict
from scripts.merge_jaad_with_labels import read_xml_file, get_pedestrians_from_dict
from scripts.xml_to_dict import xml_to_dict



def get_single_pedestrians_from_annotations():
    dir_jaad = "D:/APP-RAS/annotations/annotations"
    files = []
    for filename in os.listdir(dir_jaad):
        file = os.path.join(dir_jaad, filename)
        annotations = xml_to_dict(read_xml_file(file))
        pedestrians = get_pedestrians_from_dict(annotations)
        if len(pedestrians) == 1 and filename not in files:
            files.append(filename.split('.')[0])

    return files


def get_single_pedestrians_from_json():
    # file_dir = "C:/Users/max00/Documents/PoseRecognition/pedestrian-pose-recognition/data/JAAD_output_JSON"
    file_dir = "C:/Users/HP/Desktop/APP-RAS/pedestrian-pose-recognition-action_recognition/pedestrian-pose-recognition-action_recognition/data/JAAD_output_JSON"
    json_list = []

    for dirpath, dirnames, filenames in os.walk(file_dir):
        JAAD_list = [pos_json for pos_json in filenames if pos_json.endswith('.json')]
        for js in JAAD_list:
            video_name = js.replace("_", " ", 1)
            video_name = video_name.split("_")[0]
            video_name = video_name.replace(" ", "_")
            if video_name in json_list:
                continue
            path = os.path.join(dirpath, js)
            with open(path) as json_file:
                data = json.load(json_file)
                people_list = data['people']
                if len(people_list) == 1:
                    json_list.append(js)

    video_list = ["_".join(j.split("_", 2)[:2]) for j in json_list]
    video_list = list(set(video_list))
    return json_list, video_list


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))



if __name__ == "__main__":

    json_single_pedestrian, video_list = get_single_pedestrians_from_json()
    jaad_single_pedestrians = get_single_pedestrians_from_annotations()
    save_single_pedestrians = intersection(video_list, jaad_single_pedestrians)
    files = [f + ".mp4" for f in save_single_pedestrians]
    update_json_single_pedestrian = []
    for js in json_single_pedestrian:
        video_name = js.replace("_", " ", 1)
        video_name = video_name.split("_")[0]
        video_name = video_name.replace(" ", "_")
        if video_name in save_single_pedestrians:
            update_json_single_pedestrian.append(js)

    print(f"There are {len(save_single_pedestrians)} file with only one pedestrian")
    print(save_single_pedestrians)