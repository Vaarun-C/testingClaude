import numpy as np

import re
import os

def combine_files(dir_path, regex=r"d_\d+\.npy$", is_10s=False):
    pattern = re.compile(regex)
    np_files = [np.load(os.path.join(dir_path, f)) for f in os.listdir(dir_path) if pattern.match(f)]

    if is_10s:
        np_files = [row for np_file in np_files for row in np_file]

    flattened_list = np.array([item for sublist in np_files for item in sublist])

    return flattened_list

def convert_10s_to_single(file_path):
    np_file = np.load(file_path, allow_pickle=True)
    flattened_list = np.array([item for row in np_file for item in row])
    # print(flattened_list)
    return flattened_list

if __name__ == '__main__':
    s = convert_10s_to_single('dataset/newdata/noise/noise_0011.npy')
    print(s.shape)