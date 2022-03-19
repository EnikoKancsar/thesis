# Source: https://stackoverflow.com/a/61074404/13497164

import json
import os

from numpy import ndarray
from numpy import uint16, uint8, int16
from scipy import io

from conf import MPII_ANNOTATIONS_PATH

MPII_MAT = io.loadmat(MPII_ANNOTATIONS_PATH, struct_as_record=False)["RELEASE"]

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
                # val = generate_dataset_obj(obj[i])
                # if isinstance(val, (uint8, uint16, int16)):
                #     # val = int(val)
                #     val = str(val)

                # I'm not sure if this is actually okay :///
                ret.append(str(generate_dataset_obj(obj[i])))

    # elif type(obj) == io.matlab.mio5_params.mat_struct:
    # DeprecationWarning: Please use `mat_struct` from the `scipy.io.matlab` 
    # namespace, the `scipy.io.matlab.mio5_params` namespace is deprecated.
    elif isinstance(obj, io.matlab.mat_struct):
        ret = {}
        for field_name in obj._fieldnames:
            field = generate_dataset_obj(obj.__dict__[field_name])
            if field_name in MUST_BE_LIST and not isinstance(field, list):
                field = [field]
            ret[field_name] = field
    else:
        ret = obj
    return ret

MPII_DICT = generate_dataset_obj(MPII_MAT)
MPII_STR = json.dumps(MPII_DICT)

path_without_extension, mat_extension = os.path.splitext(MPII_ANNOTATIONS_PATH)
json_path = f'{path_without_extension}.json'

with open(json_path, 'w') as file:
    file.write(MPII_STR)

"""
MPII_DICT
dict_keys(['annolist', 'img_train', 'version', 'single_person', 'act', 'video_list'])

annolist (type: list) is a list of dicts, 24987 length
first element:
{'image': {'name': '037454012.jpg'}, 'annorect': [{'scale': 3.8807339512004684, 'objpos': {'x': '601', 'y': '380'}}], 
'frame_sec': [], 'vididx': []}

img_train (type: list), 24987 length
0-1 values wether it is train or test
# print(MPII_DICT['img_train'].count(0))
# print(MPII_DICT['img_train'].count(1))
# print(len(MPII_DICT['img_train']))
# 6908
# 18079
# 24987

version (type: numpy.str_)
12

single_person (type: list), 24987 length
how many people on the picture
int, list of ints (guesses?), empty list (unknown)

act (type: list), 24987 length
{'cat_name': [], 'act_name': [], 'act_id': '-1'} 
{'cat_name': 'sports', 'act_name': 'curling', 'act_id': '1'}
{'cat_name': 'transportation', 'act_name': 'driving automobile or light truck', 'act_id': '939'},
{'cat_name': 'transportation', 'act_name': 'pushing car', 'act_id': '972'},

videolist (type: list), 2821 length
'zuhzWKbRq6s', 'zvMWkSAcSVc', 'zwqQrtD2L84', 'zz5DvBqit8A'
https://www.youtube.com/watch?v=...
"""