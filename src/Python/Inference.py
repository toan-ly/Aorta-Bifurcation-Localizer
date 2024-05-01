# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 2024

@author: Toan Ly

"""

import os
import glob
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.io as sio
import tensorflow as tf
from tensorflow.keras.models import load_model


# -----------------------------------------------------------------------------
# GPU set-up
# -----------------------------------------------------------------------------
gpu_to_use = 0  #0 or 1 for 109
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

# -----------------------------------------------------------------------------
# Function Definitions
# -----------------------------------------------------------------------------
# def process_test_dir(input_dir, output_dir):
#     """Process the whole test directory"""
#     os.makedirs(output_dir, exist_ok=True)
#     image_files = glob.glob(os.path.join(input_dir, '*.mat'))
#     for image_file in tqdm(image_files, desc='Processing images'):
#         print(' ', os.path.basename(image_file))
#         process_image_file(image_file, output_dir)

def process_test_dir(input_dir, output_dir):
    """Process the whole test directory"""
    os.makedirs(output_dir, exist_ok=True)
    image_files = glob.glob(os.path.join(input_dir, '*.mat'))
    with tqdm(total=len(image_files), desc='Processing images', leave=True, ncols=100) as pbar:
        for image_file in image_files:
            pbar.set_postfix(file=os.path.basename(image_file))
            process_image_file(image_file, output_dir)
            pbar.update(1)
    
def process_image_file(image_file, output_dir):
    """Process each image file, do inference, and save as mat file"""
    vol = sio.loadmat(image_file)['vol']
    pred_vol, pred_centroid = inference(vol)
    
    file_name = os.path.basename(image_file)
    dataset, pat = file_name.split('_')
    location_idx = np.where((tbl['dataset'] == dataset) & (tbl['pat'] == pat))[0]
    location_cols = ['xyzAortaBifur_1', 'xyzAortaBifur_2', 'xyzAortaBifur_3']
    xyz_location = tbl.loc[location_idx, location_cols].values[0]
    gt_vol = create_target_vol(vol, xyz_location, method='fuzzy')
    
    output_file = os.path.join(output_dir, file_name)
    sio.savemat(output_file, {
        'gt_vol': gt_vol,
        'pred_vol': pred_vol,
        'gt_centroid': xyz_location,
        'pred_centroid': pred_centroid},
        do_compression=True
    )
    
def create_target_vol(vol, location, method='sphere'):
    """Create target vol with sphere at aorta bifurcation point"""
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

def preprocess(vol, window_rng=(-500, 1300)):
    """Preprocess input CT vol"""
    vol = (vol - window_rng[0]) / (window_rng[1] - window_rng[0])
    vol = np.clip(vol, 0, 1)
    
    return vol

def postprocess(vol, threshold=0.5):
    """Postprocess predicted vol"""
    vol_mask = (vol > threshold).astype(np.uint8)
    centroid = get_centroid(vol_mask)
    
    return vol_mask, centroid

def get_centroid(vol_mask):
    y, x, z = np.where(vol_mask)
    centroid_x = np.mean(x)
    centroid_y = np.mean(y)
    centroid_z = np.mean(z)
    centroid = np.array([centroid_x, centroid_y, centroid_z])
    
    return centroid

def prepad_vol(vol, num_slices):
    """Pad input CT vol for first and last slices with replication"""
    padded_vol = np.pad(vol, ((0, 0), (0, 0), (num_slices, num_slices)), mode='edge')
  
    return padded_vol

def extract_patches(vol, patch_sz=(128, 128, 5), stride=(64, 64, 1)):
    """Extract patches from input vol"""
    vol = prepad_vol(vol, 2)
    num_patches = [int(np.ceil((vol.shape[i] - patch_sz[i]) / stride[i])) + 1 for i in range(3)]
    
    patches = []
    # Extract patches
    for x in range(num_patches[0]):
        for y in range(num_patches[1]):
            for z in range(num_patches[2]):    
                start = [x * stride[0], y * stride[1], z * stride[2]]
                end = [min(start[i] + patch_sz[i], vol.shape[i]) for i in range(3)]
                start = [end[i] - patch_sz[i] for i in range(3)] # Ensure patches are of equal size

                patch = vol[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
                patches.append(patch)

    return np.array(patches)

def predict(model, patches):
    """Predict using trained model"""
    # return model.predict(patches)[..., 1][..., np.newaxis]
    return model.predict(patches)
    
def assemble_patches(patches, vol_shape, patch_sz=(128, 128, 1), stride=(64, 64, 1)):
    """Assemble patches back into output 3D CT vol"""
    pred_vol = np.zeros(vol_shape)
    num_patches = [int(np.ceil((vol_shape[i] - patch_sz[i]) / stride[i])) + 1 for i in range(3)]
    idx = 0
    for x in range(num_patches[0]):
        for y in range(num_patches[1]):
            for z in range(num_patches[2]):   
                start = [x * stride[0], y * stride[1], z * stride[2]]
                end = [min(start[i] + patch_sz[i], vol_shape[i]) for i in range(3)]
                start = [end[i] - patch_sz[i] for i in range(3)]
                                
                pred_vol[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = np.maximum(
                    pred_vol[start[0]:end[0], start[1]:end[1], start[2]:end[2]], patches[idx]
                )
                idx += 1
    
    return pred_vol

def inference(vol):
    """Inference code for each input vol"""
    preprocessed_vol = preprocess(vol, window_rng=(-1150, 350))
    patches = extract_patches(preprocessed_vol)
    predictions = predict(model, patches)
    pred_vol = assemble_patches(predictions, vol.shape)
    vol_mask, centroid = postprocess(pred_vol, 0.5)
    
    return vol_mask, centroid
        
# -----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------- 
if __name__ == '__main__':
    start_time = time.time() 
    start_time_str = time.strftime("%Y/%m/%d %H:%M", time.localtime(start_time))
    print('-------------------------------------')  
    print('Running main script at', start_time_str, '\n')
               
    # Load the trained model
    model_path = r'Project/Models/Model1-2-AlmostFullSphere/2024-03-07-TF2.5.0-Net-CP017.h5'
    model = load_model(model_path)
    # Load ground truth location table
    location_tbl_path = 'Project/LocationTbl.csv'
    tbl = pd.read_csv(location_tbl_path)
    
    input_dir = r'Project\DataPatches\Data3\Test'
    output_dir = r'Project\InferenceOutput\OutputData5'
    process_test_dir(input_dir, output_dir)
    
    print('Inference outputs are saved to ', output_dir)
    
    end_time = time.time() 
    end_time_str = time.strftime("%Y/%m/%d %H:%M", time.localtime(end_time))
    print('\nDone with Inference at', end_time_str, '!')

   
    
    
    
    
        
   
  
    

    

    
  
