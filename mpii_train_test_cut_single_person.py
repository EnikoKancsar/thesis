"""
This script cuts the MPII annotations.JSON and images into a train and test
dataset and annotation file.
While performing the cut
1. it also restructures the annotations in the way described in the README.md,
2. filters the images to keep only the ones with a single person on them
"""

import configparser
import json
import os
import shutil


CONF = configparser.ConfigParser()
CONF.read("./conf.ini")


try:
    with open(CONF.get("MPII", "ANNOTATIONS_JSON"), 'r') as file:
        MPII_DICT = json.load(file)
except FileNotFoundError as err:
    print("Path to json annotations file not specified."
          "Please edit your config file.")


MPII_LIST_TRAIN, MPII_LIST_VAL = [], []

for index, value in enumerate(MPII_DICT["img_train"]):
    if MPII_DICT["single_person"][index] not in [1, [1]]:
        continue
    if len(MPII_DICT["annolist"][index]["annorect"]) > 1:
        continue

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

    if len(MPII_DICT["annolist"][index]["annorect"]) == 0:
        MPII_DICT["annolist"][index]["annorect"].append({})

    for person in MPII_DICT["annolist"][index]["annorect"]:
        if "objpos" not in person:
            person["objpos"] = {}
        if person["objpos"] == []:
            person["objpos"] = {}
        if "annopoints" not in person:
            person["annopoints"] = {}
        if person["annopoints"] == []:
            person["annopoints"] = {}
        if "point" not in person["annopoints"]:
            person["annopoints"]["point"] = []

        annotation_of_person = {
            "scale": float(person.get("scale", -1)),
            "objpos": {
                "x": float(person["objpos"].get("x", -1)),
                "y": float(person["objpos"].get("y", -1))
            },
            "head_rectangle": {
                "x1": float(person.get("x1", -1)),
                "y1": float(person.get("y1", -1)),
                "x2": float(person.get("x2", -1)),
                "y2": float(person.get("y2", -1))
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

        for joint in person["annopoints"]["point"]:
            annotation_of_person["joints"][str(joint["id"])] = {
                "x": float(joint.get("x", -1)),
                "y": float(joint.get("y", -1)),
                "is_visible": (int(joint.get("is_visible", -1))
                               if joint.get("is_visible") != []
                               else -1)
            }

        annotations_of_image.append(annotation_of_person)
    
    dest_list.append({
        "image_name": file_name,
        "single_person": MPII_DICT["single_person"][index],
        "activity": MPII_DICT["act"][index],
        "list_of_people": annotations_of_image
    })

try:
    with open(CONF.get("MPII", "ANNOTATIONS_TRAIN"), "w") as file:
        file.write(json.dumps(MPII_LIST_TRAIN))
except FileNotFoundError as err:
    print("Path to train annotations file not specified."
          "Please edit your config file.")

try:
    with open(CONF.get("MPII", "ANNOTATIONS_VAL"), "w") as file:
        file.write(json.dumps(MPII_LIST_VAL))
except FileNotFoundError as err:
    print("Path to val annotations file not specified."
          "Please edit your config file.")
