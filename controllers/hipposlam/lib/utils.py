import json
import os.path
import pickle
import pandas as pd
import numpy as np
import copy

def midedges(edges):
    return (edges[:-1] + edges[1:]) / 2

def save_pickle(save_path, data):
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

def read_pickle(read_path):
    with open(read_path, 'rb') as f:
        data = pickle.load(f)
    return data

# Function to append a nested dictionary to a file
def append_dict_to_json(data, filename):
    with open(filename, 'a') as file:
        json.dump(data, file)
        file.write('\n')

def load_dict_from_json(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            # Load each line (which should contain a JSON object) and append to data list
            data.append(json.loads(line))
    return data


class Recorder:
    def __init__(self, *args):
        self.records_dict = {key:[] for key in args}
        self.records_df = None

    def record(self, **kwargs):
        for key, val in kwargs.items():
            self.records_dict[key].append(val)

    def return_avers(self):
        return {key:np.mean(val) for key, val in self.records_dict.items()}

    def to_csv(self, save_pth):
        self.records_df = pd.DataFrame(self.records_dict)
        self.records_df.to_csv(save_pth)
    def clear_records_dict(self):
        for key in self.records_dict.keys():
            self.records_dict[key] = []

    def append_to_pickle(self, data_pth):
        if os.path.exists(data_pth):
            data = read_pickle(data_pth)  # list of records_dict
            data.append(self.records_dict)
            save_pickle(data_pth, data)
        else:
            self.save_as_pickle_for_append(data_pth)

    def save_as_pickle_for_append(self, data_pth):
        save_pickle(data_pth, [self.records_dict])


    def __getitem__(self, key):
        return self.records_dict[key]



def breakroom_avoidance_policy(x, y, dsval, noise=0.3):

    a = 0
    if dsval < 95:
        if x > 2:  # First room
            a = 2  # right
        elif (x < 2 and x > -2.7) and (y < 1.45):  # middle room, lower half
            a = 2  # right

        elif (x < 2 and x > -2.7) and (y > 1.45):  # middle room, upper half
            a = 1  # left
        else:  # Final Room
            a = 1  # left
    else:
        a = 0

    if np.random.rand() < noise:
        a = int(np.random.randint(0, 4))

    return a






