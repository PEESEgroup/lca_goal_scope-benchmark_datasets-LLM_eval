import os
import json

# get the directory
directory_path = "/data/recalculated/"

# for each file in the directory, iterate through
for entry_name in os.listdir(directory_path):
    # get file path
    file_path = os.path.join(directory_path, entry_name)
    # open file and remove LCIA calculated values
    with open(file_path, 'r') as f:
        data = json.load(f)
        print(data)
