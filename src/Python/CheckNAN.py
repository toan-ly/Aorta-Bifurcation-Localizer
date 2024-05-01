# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 13:07:31 2024

@author: tly
"""
          
import glob
import numpy as np
import scipy.io as sio
from tqdm import tqdm
from multiprocessing import Pool

def check_nan(file):
    data = sio.loadmat(file)
    imStack, imLabel = data['imStack'], data['imLabel']
    if np.isnan(imStack).any() or np.isnan(imLabel).any():
        return 1
    return 0

if __name__ == '__main__':
    data_paths = [r'Project\DataPatches\Data1']
    file_paths = []
    for data_path in data_paths:
        for grp in ['Val', 'Train']:
            file_paths.extend(glob.glob(data_path + '/' + grp + '/*.mat'))

    with Pool(8) as pool:
        nan_counts = list(tqdm(pool.imap(check_nan, file_paths), total=len(file_paths)))

    i = sum(nan_counts)
    print(i)
