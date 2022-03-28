# Source: https://stackoverflow.com/a/61074404/13497164

import configparser
import json
import os
import shutil

from numpy import ndarray
from numpy import uint16, uint8, int16
from scipy import io


CONF = configparser.ConfigParser()
CONF.read("./conf.ini")


MPII_MAT = io.loadmat(CONF.get("MPII", "ANNOTATIONS_MAT"), struct_as_record=False)["RELEASE"]

MUST_BE_LIST = ["annolist", "annorect", "point", "img_train", "single_person",
                "act", "video_list"]

def generate_dataset_obj(obj):
    if isinstance(obj, ndarray):
        dim = obj.shape[0]
        if dim == 1:
            ret = generate_dataset_obj(obj[0])
        else:
            ret = []
            for i in range(dim):
                val = generate_dataset_obj(obj[i])
                if isinstance(val, (uint8, uint16, int16)):
                    val = int(val)
                ret.append(val)
    elif isinstance(obj, io.matlab.mat_struct):
        ret = {}
        for field_name in obj._fieldnames:
            field = generate_dataset_obj(obj.__dict__[field_name])
            if field_name in MUST_BE_LIST and not isinstance(field, list):
                field = [field]
            if isinstance(field, (uint8, uint16, int16)):
                    field = int(field)
            ret[field_name] = field
    else:
        ret = obj
    return ret

MPII_DICT = generate_dataset_obj(MPII_MAT)
MPII_LIST_TRAIN, MPII_LIST_VAL = [], []

for index, value in enumerate(MPII_DICT["img_train"]):
    dest_dir, dest_list = (
        (CONF.get("MPII", "DIR_IMAGES_TRAIN"), MPII_LIST_TRAIN) 
        if value == 1
        else (CONF.get("MPII", "DIR_IMAGES_VAL"), MPII_LIST_VAL)
    )
    file_name = MPII_DICT["annolist"][index]["image"]["name"]
    file_path = os.path.join(CONF.get("MPII", "DIR_IMAGES"), file_name)
    try:
        shutil.copy(file_path, dest_dir)
    except FileNotFoundError as err:
        print("annotation without file: ", file_name)
        print("index: ", index, "\n")
        continue

    annotations_of_image = []
        
    for person in MPII_DICT["annolist"][index]["annorect"]:
        if "objpos" not in person:
            person["objpos"] = {}
        if person["objpos"] == []:
            person["objpos"] = {}

        annotation_of_person = {
            "scale": person.get("scale"),
            "objpos": {
                "x": person["objpos"].get("x"),
                "y": person["objpos"].get("y")
            },
            "head_rectangle": {
                "x1": person.get("x1"),
                "y1": person.get("y1"),
                "x2": person.get("x2"),
                "y2": person.get("y2")
            },
            "joints": {
                "0": {"x": -1, "y": -1, "is_visible": -1},
                "1": {"x": -1, "y": -1, "is_visible": -1},
                "2": {"x": -1, "y": -1, "is_visible": -1},
                "3": {"x": -1, "y": -1, "is_visible": -1},
                "4": {"x": -1, "y": -1, "is_visible": -1},
                "5": {"x": -1, "y": -1, "is_visible": -1},
                "6": {"x": -1, "y": -1, "is_visible": -1},
                "7": {"x": -1, "y": -1, "is_visible": -1},
                "8": {"x": -1, "y": -1, "is_visible": -1},
                "9": {"x": -1, "y": -1, "is_visible": -1},
                "10": {"x": -1, "y": -1, "is_visible": -1},
                "11": {"x": -1, "y": -1, "is_visible": -1},
                "12": {"x": -1, "y": -1, "is_visible": -1},
                "13": {"x": -1, "y": -1, "is_visible": -1},
                "14": {"x": -1, "y": -1, "is_visible": -1},
                "15": {"x": -1, "y": -1, "is_visible": -1}
            }
        }

        if "annopoints" not in person:
            person["annopoints"] = {}
        if person["annopoints"] == []:
            person["annopoints"] = {}
        if "point" not in person["annopoints"]:
            person["annopoints"]["point"] = []
        for joint in person["annopoints"]["point"]:
            annotation_of_person["joints"][str(joint["id"])] = {
                "x": joint.get("x", -1),
                "y": joint.get("y"),
                "is_visible": (joint.get("is_visible")
                               if joint.get("is_visible") != []
                               else -1)
            }

        annotations_of_image.append(annotation_of_person)
    
    
    if len(MPII_DICT["annolist"][index]["annorect"]) == 0:
        annotations_of_image = [{
            "scale": -1,
            "objpos": {
                "x": -1,
                "y": -1
            },
            "head_rectangle": {
                "x1": -1,
                "y1": -1,
                "x2": -1,
                "y2": -1
            },
            "joints": {
                "0": {"x": -1, "y": -1, "is_visible": -1},
                "1": {"x": -1, "y": -1, "is_visible": -1},
                "2": {"x": -1, "y": -1, "is_visible": -1},
                "3": {"x": -1, "y": -1, "is_visible": -1},
                "4": {"x": -1, "y": -1, "is_visible": -1},
                "5": {"x": -1, "y": -1, "is_visible": -1},
                "6": {"x": -1, "y": -1, "is_visible": -1},
                "7": {"x": -1, "y": -1, "is_visible": -1},
                "8": {"x": -1, "y": -1, "is_visible": -1},
                "9": {"x": -1, "y": -1, "is_visible": -1},
                "10": {"x": -1, "y": -1, "is_visible": -1},
                "11": {"x": -1, "y": -1, "is_visible": -1},
                "12": {"x": -1, "y": -1, "is_visible": -1},
                "13": {"x": -1, "y": -1, "is_visible": -1},
                "14": {"x": -1, "y": -1, "is_visible": -1},
                "15": {"x": -1, "y": -1, "is_visible": -1}
            }
        }]
    
    dest_list.append({
        "image_name": file_name,
        "single_person": MPII_DICT["single_person"][index],
        "act": MPII_DICT["act"][index],
        "list_of_people": annotations_of_image
    })

try:
    with open(CONF.get("MPII", "ANNOTATIONS_TRAIN"), "w") as file:
        file.write(json.dumps(MPII_LIST_TRAIN))
except FileNotFoundError as err:
    print("Path to train annotations file not specified. Please edit your config file.")

try:
    with open(CONF.get("MPII", "ANNOTATIONS_VAL"), "w") as file:
        file.write(json.dumps(MPII_LIST_VAL))
except FileNotFoundError as err:
    print("Path to val annotations file not specified. Please edit your config file.")
