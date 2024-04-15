import json
import os.path
import pickle
import pandas as pd
import numpy as np
import copy

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


class TrajWriter:
    def __init__(self, filename, *args):
        self.filename = filename
        self.save_traj_pth = filename + '.csv'
        self.save_fsigma_pth = filename + '.json'
        self.traj_fh = None
        self.fsigma_fh = None
        for pth in [self.save_traj_pth, self.save_fsigma_pth]:
            if os.path.exists(pth):
                os.remove(pth)

        self.traj_fh = open(self.save_traj_pth, 'a')
        self.fsigma_fh = open(self.save_fsigma_pth, 'a')

        self._write_args_per_line(self.traj_fh, args)

    def write(self, data, *args):
        json.dump(data, self.fsigma_fh)
        self.fsigma_fh.write('\n')
        self._write_args_per_line(self.traj_fh, args)

    def _write_args_per_line(self, fh, args):
        header = ''.join([f'{arg}, ' for arg in args])
        fh.write(header[:-2] + '\n')

    def close_fh(self):
        if self.traj_fh:
            self.traj_fh.close()
        if self.fsigma_fh:
            self.fsigma_fh.close()





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






