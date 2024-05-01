# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 2024

@author: Toan Ly

"""
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from scipy.io import loadmat, savemat
from tqdm import tqdm


gpu_to_use = 0  #0 or 1 for 109

# GPU set-up
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[gpu_to_use], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[gpu_to_use], True)
        # # Currently, memory growth needs to be the same across GPUs
        # for gpu in gpus:
        #     tf.config.experimental.set_memory_growth(gpu, True)
        # logical_gpus = tf.config.list_logical_devices('GPU')
        # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


target_sz = [128, 128, 5]
overlap = [64, 64, 2]

def extract_patches(im, patch_sz=(128, 128, 5), overlap=(64, 64, 2), is_label=0):
    """
    Extract equal-size 3D or 2D patches from a 3D image with overlap.

    Parameters:
    im (np.ndarray): Input 3D image.
    patchSz (tuple): Patch size [patchSizeX, patchSizeY, patchSizeZ].
    overlap (tuple): Overlap size [overlapX, overlapY, overlapZ].
    isLabel (bool): Indicates whether patches are labels or not.

    Returns:
    np.ndarray: Array containing the extracted patches.
    """
    step = np.array(patch_sz) - np.array(overlap)

    # Calculate the number of patches along each dimension
    num_patches = [int(np.ceil((im.shape[i] - patch_sz[i]) / step[i])) + 1 for i in range(3)]
    patches = np.empty(num_patches, dtype=object)

    # Extract patches
    for x in range(num_patches[0]):
        for y in range(num_patches[1]):
            for z in range(num_patches[2]):    
                start = [x * step[0], y * step[1], z * step[2]]
                end = [min(start[i] + patch_sz[i], im.shape[i]) for i in range(3)]

                # Adjust start and end indices to ensure patches are of equal size
                start = [end[i] - patch_sz[i] for i in range(3)]

                if is_label:
                    # Extract the middle slice for labels
                    mid_slice = start[2] + (patch_sz[2] // 2)
                    patches[x, y, z] = im[start[0]:end[0], start[1]:end[1], mid_slice]
                else:
                    patches[x, y, z] = im[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

    return patches

def create_target_vol(vol, location, method='sphere'):
    """
    Create a new target space that has a fuzzy sphere around the aorta bifurcation.

    Parameters:
    vol (np.ndarray): Original 3D image.
    location (tuple): XYZ location of the aorta bifurcation.

    Returns:
    np.ndarray: A 3D mask representing the fuzzy sphere around the aorta bifurcation.
    """
    # Create a blank space
    vol_mask = np.zeros_like(vol)
    
    aorta_bifur = location - 1 # MATLAB starts with 1, Python starts with 0
    x, y, z = np.meshgrid(np.arange(vol.shape[0]), np.arange(vol.shape[1]), np.arange(vol.shape[2]))

    if method == 'sphere':
        vol_mask = (x - aorta_bifur[0])**2 + (y - aorta_bifur[1])**2 + (z - aorta_bifur[2])**2 < 100
    elif method == 'fuzzy':
        sigma = 3
        distances = np.sqrt((x - aorta_bifur[0])**2 + (y - aorta_bifur[1])**2 + (z - aorta_bifur[2])**2)
    
        # Create Gaussian
        gaussian = np.exp(-0.5 * (distances**2) / (sigma**2))
        gaussian = (gaussian - np.min(gaussian)) / (np.max(gaussian) - np.min(gaussian))
        gaussian[gaussian < 0.1] = 0
        vol_mask = gaussian

    return vol_mask

def process_file(file_info, original_path, new_path, tbl):
    dataset, filename, group = file_info
    file_path = os.path.join(original_path, dataset, filename)
    patch_path = os.path.join(new_path, group)
    vol = loadmat(file_path)['vol']

    if group == 'Test':
        save_name = f'{dataset}_{filename}'
        savemat(os.path.join(patch_path, save_name), {'vol': vol}, do_compression=True)
    else:
        patches = extract_patches(vol, target_sz, overlap, False)

        location_idx = np.where((tbl['dataset'] == dataset) & (tbl['pat'] == filename))[0]
        location_cols = ['xyzAortaBifur_1', 'xyzAortaBifur_2', 'xyzAortaBifur_3']
        xyz_location = tbl.loc[location_idx, location_cols].values[0]
        vol_mask = create_target_vol(vol, xyz_location, method='fuzzy')
        labels = extract_patches(vol_mask, target_sz, overlap, True)
        
        filename = filename.split('.')[0]
        for x in range(len(patches)):
            for y in range(len(patches[0])):
                for z in range(len(patches[0][0])):
                    im_stack = patches[x, y, z]
                    im_label = np.single(labels[x, y, z])

                    label_filename = f"{dataset}_{filename}_{x+1}_{y+1}_{z+1}_negative.mat"
                    if np.any(im_label > 0):
                        label_filename = f"{dataset}_{filename}_{x+1}_{y+1}_{z+1}_positive.mat"

                    savemat(os.path.join(patch_path, label_filename), {'imStack': im_stack, 'imLabel': im_label}, do_compression=True)

def create_folder(new_path):
    os.makedirs(new_path, exist_ok=True)
    for folder in ['Train', 'Test', 'Val']:
        folder_path = os.path.join(new_path, folder)
        os.makedirs(folder_path, exist_ok=True)

def split_data():
    """
    Combine datasets from aorta bifurcation and sacrum mini project 1,
    extract patches and labels, randomly split into 3 new datasets with
    400 training, 50 test, and 50 validation, and store into new folder.
    """
    
    data = pd.read_excel('C:/Users/tly/Downloads/Riverain/Project/DataGroup.xlsx')
    tbl = pd.read_csv('C:/Users/tly/Downloads/Riverain/Project/LocationTbl.csv')
    
    original_path = r'Project/InputData'
    new_path = r'Project/DataPatches/Data'
    create_folder(new_path)
    
    # Loop though excel file
    for i in tqdm(range(len(data))):
        dataset = data.loc[i, 'Data Set']
        filename = data.loc[i, 'Name']
        group = data.loc[i, 'Group']
        process_file((dataset, filename, group), original_path, new_path, tbl)
        
    print('Done Splitting Data!')


if __name__ == '__main__':
    split_data()
