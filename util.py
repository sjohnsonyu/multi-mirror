import numpy as np


def read_write_initialization_vals(arr, name, exp_path, write_initialization, read_initialization):
    full_path = exp_path + name
    if write_initialization:
        np.savetxt(full_path, arr)
    if read_initialization:
        arr = np.loadtxt(full_path)
    return arr


def read_write_initialization_pickle(obj, name, exp_path, write_initialization, read_initialization):
    import pickle
    full_path = exp_path + name
    if write_initialization:
        with open(full_path, 'wb') as f:
            pickle.dump(obj, f)

    if read_initialization:
        with open(full_path, 'rb') as f:
            obj = pickle.load(f)
        return obj

