# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 11:30:04 2024

@author: tly
"""
import glob
import scipy.io
import os
from tqdm import tqdm

directory = r'Project\DataPatches\Data3\TrainWarp'

def check_file_sizes(directory):
    error_files = []
    mat_files = glob.glob(os.path.join(directory, '*.mat'))
    print(f'Total number of files: {len(mat_files)}')
    for filepath in tqdm(mat_files, desc='Checking files', unit='file'):
        data = scipy.io.loadmat(filepath)
        imStack = data['imStack']
        imLabel = data['imLabel']
        if imStack is None or imLabel is None:
            error_files.append(filepath)
            continue
        if imStack.shape != (128, 128, 5) or imLabel.shape != (128, 128):
            error_files.append(filepath)
            if error_files:
                return error_files
    return error_files

error_files = check_file_sizes(directory)

if error_files:
    print(f"\n{len(error_files)} files with incorrect sizes found:")
    for filepath in error_files:
        print(filepath)
else:
    print("No files with incorrect sizes found. All pass!!!")


