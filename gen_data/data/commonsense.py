import json
import os

object_dict = json.load(open('vh/data_gene/gen_data/data/commonsense.json', 'r'))

for i in range(7):
    print(i)
    file = 'vh/data_gene/gen_data/data/object_info{}.json'.format(i+1)
    if os.path.isfile(file):
        data = json.load(open(file, 'r'))
        for key in data.keys():
            if key in object_dict:
                data[key] = object_dict[key]
        json.dump(data, open(file, 'w+'), indent=4)