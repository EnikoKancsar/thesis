# Source: https://stackoverflow.com/a/61074404/13497164

"""
This script converts the .mat annotations file
to a .json file with the same structure.
"""

import configparser
import json

from numpy import ndarray
from numpy import uint16, uint8, int16
from scipy import io


CONF = configparser.ConfigParser()
CONF.read("./conf.ini")

try:
    path_annotations_mat = CONF.get("MPII", "ANNOTATIONS_MAT")
except FileNotFoundError as err:
    print("Path to annotations file not specified. "
          "Please edit your conf.ini file.")
    exit()

MPII_MAT = io.loadmat(path_annotations_mat, struct_as_record=False)["RELEASE"]

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

try:
    with open(CONF.get("MPII", "ANNOTATIONS_JSON"), "w") as file:
        file.write(json.dumps(MPII_DICT))
except FileNotFoundError as err:
    print("Path to annotations file not specified. "
          "Please edit your conf.ini file.")
