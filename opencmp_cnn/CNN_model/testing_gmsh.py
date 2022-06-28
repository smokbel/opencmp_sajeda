import os
from pathlib import Path
import json

p = Path('../../../../cnn_data')
subdirectories = [x for x in p.iterdir() if x.is_dir()]

for sub in subdirectories:
    print(sub)
    jsonfile = str(sub) + '/info.json'
    with open(jsonfile) as json_file:
        param = json.load(json_file)
    target_file = param["target_sol"]
    for file in os.listdir(sub):
        if file in [target_file]:
            print(file)
    #print(sub)
