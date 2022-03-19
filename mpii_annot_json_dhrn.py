import json
from pprint import pprint

import conf

paths = [
    conf.MPII_DHRN_DIR + r'\test.json',
    conf.MPII_DHRN_DIR + r'n\train.json',
    conf.MPII_DHRN_DIR + r'\trainval.json',
    conf.MPII_DHRN_DIR + r'\valid.json'
]
for path in paths:
    with open(path, 'r') as file:
        MPII = json.load(file)
        print(path)
        print(len(MPII))
        print(MPII[0].keys())
        pprint(MPII[0])
        print('\n')
