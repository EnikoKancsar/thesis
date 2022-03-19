import json
from pprint import pprint

paths = [
    r'C:\Users\Eniko\College\Szakdolgozat\Datasets\MPII\json\test.json',
    r'C:\Users\Eniko\College\Szakdolgozat\Datasets\MPII\json\train.json',
    r'C:\Users\Eniko\College\Szakdolgozat\Datasets\MPII\json\trainval.json',
    r'C:\Users\Eniko\College\Szakdolgozat\Datasets\MPII\json\valid.json'
]
for path in paths:
    with open(path, 'r') as file:
        MPII = json.load(file)
        print(path)
        print(len(MPII))
        print(MPII[0].keys())
        pprint(MPII[0])
        print('\n')
