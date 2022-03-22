# Source: https://stackoverflow.com/a/61074404/13497164

import configparser
import json
import shutil

from numpy import ndarray
from numpy import uint16, uint8, int16
from scipy import io


CONF = configparser.ConfigParser()
CONF.read('./conf.ini')


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

for index, value in enumerate(MPII_DICT['img_train']):
    dest_dir, annotation_list = (
        (CONF.get("MPII", "DIR_IMAGES_TRAIN"), MPII_LIST_TRAIN) 
        if value == 1
        else (CONF.get("MPII", "DIR_IMAGES_VAL"), MPII_LIST_VAL)
    )
    file_name = MPII_DICT['annolist'][index]['image']['name']
    try:
        shutil.copy(CONF.get("MPII", "DIR_IMAGES") + '\\' + file_name, dest_dir)
    except FileNotFoundError as err:    
        continue
    annotation_list.append({
        'annolist': MPII_DICT['annolist'][index],
        'single_person': MPII_DICT['single_person'][index],
        'act': MPII_DICT['act'][index]
    })


try:
    with open(CONF.get("MPII", "ANNOTATIONS_TRAIN"), 'w') as file:
        file.write(json.dumps(MPII_LIST_TRAIN))
except FileNotFoundError as err:
    print('Path to train annotations file not specified. Please edit your config file.')

try:
    with open(CONF.get("MPII", "ANNOTATIONS_VAL"), 'w') as file:
        file.write(json.dumps(MPII_LIST_VAL))
except FileNotFoundError as err:
    print('Path to val annotations file not specified. Please edit your config file.')


"""How to access and use these files

```
loaded = None
with open(conf.MPII_FILE_ANNOTATIONS_JSON_VAL, 'r') as file:
    loaded = json.load(file)
print(loaded[1234])
```
{'act': {'act_id': -1, 'act_name': [], 'cat_name': []},
 'annolist': {'annorect': [{'objpos': {'x': 424, 'y': 382},
                            'scale': 2.563847109326139},  
                           {'objpos': {'x': 615, 'y': 426},
                            'scale': 1.8672246785001532}],
              'frame_sec': [],
              'image': {'name': '001386214.jpg'},
              'vididx': []},
 'single_person': [1, 2]}
"""
