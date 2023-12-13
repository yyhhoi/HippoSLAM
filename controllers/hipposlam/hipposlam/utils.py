import pickle


def save_pickle(save_path, data):
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

def read_pickle(read_path):
    with open(read_path, 'rb') as f:
        data = pickle.load(f)
    return data