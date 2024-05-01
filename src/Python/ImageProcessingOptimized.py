# -*- coding: utf-8 -*-
"""
Created on Fri Mar 6 2024

@author: Toan Ly

"""
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from scipy.io import loadmat, savemat
from tqdm import tqdm
import multiprocessing
import nibabel as nib
import time
import glob

class ImageProcesser:
    def __init__(self, data_group_path, location_tbl_path, input_data_path, output_data_path, method='fuzzy', save_nifty=False, orientation='Ax'):
        self.data_group_path = data_group_path
        self.location_tbl_path = location_tbl_path
        self.tbl = pd.read_csv(location_tbl_path)
        self.input_data_path = input_data_path
        self.output_data_path = output_data_path
        self.method = method
        self.target_sz = [128, 128, 5]
        self.overlap = [64, 64, 2]
        self.save_nifty = save_nifty
        self.orientation = orientation;
        
        
        # GPU set-up
        self.setup_gpu()

    def setup_gpu(self):
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

    def create_dataset_folders(self):
        for folder in ['Train', 'Test', 'Val']:
            os.makedirs(os.path.join(self.output_data_path, folder), exist_ok=True)

    def extract_patches(self, vol, patch_sz=(128, 128, 5), overlap=(64, 64, 2), is_2_5d=False):
        """
        Extract equal-size 3D or 2D patches from a 3D image with overlap.

        Parameters:
        im (np.ndarray): Input 3D image.
        patch_sz (tuple): Patch size [patchSizeX, patchSizeY, patchSizeZ].
        overlap (tuple): Overlap size [overlapX, overlapY, overlapZ].
        is_2_5d (bool): Indicates whether patches are labels or not.

        Returns:
        np.ndarray: Array containing the extracted patches.
        """
        step = np.array(patch_sz) - np.array(overlap)
        num_patches = [int(np.ceil((vol.shape[i] - patch_sz[i]) / step[i])) + 1 for i in range(3)]
        patches = np.empty(num_patches, dtype=object)
            
        for x in range(num_patches[0]):
            for y in range(num_patches[1]):
                for z in range(num_patches[2]):
                    start = [x * step[0], y * step[1], z * step[2]]
                    end = [min(start[i] + patch_sz[i], vol.shape[i]) for i in range(3)]
                    start = [end[i] - patch_sz[i] for i in range(3)]

                    if is_2_5d:
                        mid_slice = start[2] + (patch_sz[2] // 2)
                        patches[x, y, z] = vol[start[0]:end[0], start[1]:end[1], mid_slice]
                    else:
                        patches[x, y, z] = vol[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

        return patches
    

    def create_target_vol(self, vol, location, method='sphere'):
        """
        Create a new target space that has a fuzzy sphere around the aorta bifurcation.

        Parameters:
        vol (np.ndarray): Original 3D image.
        location (tuple): XYZ location of the aorta bifurcation.

        Returns:
        np.ndarray: A 3D mask representing the fuzzy sphere around the aorta bifurcation.
        """
        vol_mask = np.zeros_like(vol)
        aorta_bifur = np.array(location) - 1
        x, y, z = np.meshgrid(*[np.arange(dim) for dim in vol.shape], indexing='ij')

        if method == 'sphere':
            vol_mask = (x - aorta_bifur[0])**2 + (y - aorta_bifur[1])**2 + (z - aorta_bifur[2])**2 < 25
        elif method == 'fuzzy':
            sigma = 7
            distances = np.sqrt((x - aorta_bifur[0])**2 + (y - aorta_bifur[1])**2 + (z - aorta_bifur[2])**2)
            gaussian = np.exp(-0.5 * (distances**2) / (sigma**2))
            gaussian = (gaussian - np.min(gaussian)) / (np.max(gaussian) - np.min(gaussian))
            gaussian[gaussian < 0.1] = 0
            vol_mask = gaussian

        return vol_mask
    
    def get_aorta_bifurcation(self, dataset, filename):
        location_idx = np.where((self.tbl['dataset'] == dataset) & (self.tbl['pat'] == filename))[0]
        location_cols = ['xyzAortaBifur_3', 'xyzAortaBifur_1', 'xyzAortaBifur_2']
        # location_cols = [location_cols[i] for i in axes]
        
        xyz_location = self.tbl.loc[location_idx, location_cols].values[0]
        return xyz_location
    
    def extract_more_around_target(self, vol, vol_mask, xyz_location, filename, patch_path, increased_overlap=(96, 96, 4)):           
        sub_vol, sub_vol_mask = self.bbox_extract(vol, vol_mask, xyz_location)
        patches = self.extract_patches(sub_vol, self.target_sz, increased_overlap, is_2_5d=False)
        labels = self.extract_patches(sub_vol_mask, self.target_sz, increased_overlap, is_2_5d=True)
        for x in range(len(patches)):
            for y in range(len(patches[0])):
                for z in range(len(patches[0][0])):
                    im_stack = patches[x, y, z]
                    im_label = np.single(labels[x, y, z])

                    label_filename = f"{filename}_{x+1}_{y+1}_{z+1}_additional_negative.mat"
                    if np.any(im_label > 0):
                        label_filename = f"{filename}_{x+1}_{y+1}_{z+1}_additional_positive.mat"
                    if im_stack.shape != (128, 128, 5) or im_label != (128, 128):
                        print(f'\nWarning: Invalid dimensions for imStack or imLabel in {label_filename}\n' +
                              f'imStack: {im_stack.shape}, expected (128, 128, 5)\n' +
                              f'imLabel: {im_label.shape}, expected (128, 128)')
                            
                    savemat(os.path.join(patch_path, label_filename), {'imStack': im_stack, 'imLabel': im_label}, do_compression=True)
    
    def bbox_extract(self, vol, vol_mask, xyz_location, subvol_size=(160, 160, 50)):
        start = np.maximum(0, xyz_location - np.array(subvol_size) // 2)
        end = np.minimum(vol.shape, xyz_location + np.array(subvol_size) // 2)
        
        actual_size = end - start
        
        if np.any(actual_size < subvol_size):
            expand_size = subvol_size - actual_size
            if np.any(start == 0):
                end += expand_size
            if np.any(end == vol.shape):
                start -= expand_size
        
        sub_vol = vol[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        sub_vol_mask = vol_mask[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        
        return sub_vol, sub_vol_mask
    
    def process_warp_file(self, input_folder):
        warped_files = glob.glob(os.path.join(input_folder, '*.mat'))
            
        num_workers = 10
        pool = multiprocessing.Pool(processes=num_workers)
        list(tqdm(pool.imap_unordered(self.process_warp_helper, warped_files),
                      total=len(warped_files),
                      desc='Processing files',
                      leave=True,
                      ncols=100))
            
        pool.close()
        pool.join()
        # with multiprocessing.Pool(processes=num_workers) as pool:
        #     for _ in tqdm(pool.imap_unordered(process_warp_helper, warped_files), total=num_files, desc='Processing warped files'):
        #         pass
    def process_warp_helper(self, file):
           filename = os.path.basename(file).split('.')[0]
           
           data = loadmat(file)
           vol, xyz_location = data['warpedVol'], np.squeeze(data['xyzLocation'])
           xyz_location[0], xyz_location[1] = xyz_location[1], xyz_location[0]
           
           vol_mask = self.create_target_vol(vol, xyz_location, method=self.method)
           self.extract_more_around_target(vol, vol_mask, xyz_location, filename, self.output_data_path)       

    def process_file(self, args, double_positives=False):
        dataset, filename, group = args
        original_path = os.path.join(self.input_data_path, dataset, filename)
        patch_path = os.path.join(self.output_data_path, group)
        
        axes_order = {'Ax': (0, 1, 2), 'Cor': (2, 1, 0), 'Sag': (0, 2, 1)}
        axes = axes_order[self.orientation]
        
        vol = loadmat(original_path)['vol']
        vol = np.transpose(vol, axes) 
        
        if group == 'Train' or group == 'Val':
            patches = self.extract_patches(vol, self.target_sz, self.overlap, is_2_5d=False)
            
            xyz_location = self.get_aorta_bifurcation(dataset, filename)            
            if np.isnan(xyz_location).any():
                print(f'{filename} contains NaN values')
                return
            
            vol_mask = self.create_target_vol(vol, xyz_location, method=self.method)
            labels = self.extract_patches(vol_mask, self.target_sz, self.overlap, is_2_5d=True)
            
            filename = filename.split('.')[0]
            for x in range(len(patches)):
                for y in range(len(patches[0])):
                    for z in range(len(patches[0][0])):
                        im_stack = patches[x, y, z]
                        im_label = np.single(labels[x, y, z])
                        
                        label_filename = f"{dataset}_{filename}_{x+1}_{y+1}_{z+1}_negative.mat"
                        if np.any(im_label > 0):
                            label_filename = f"{dataset}_{filename}_{x+1}_{y+1}_{z+1}_positive.mat"
                            
                        if im_stack.shape != (128, 128, 5) or im_label != (128, 128):
                            print(f'Invalid dimensions for imStack or imLabel in {label_filename}\n' +
                                  f'imStack: {im_stack.shape}, expected (128, 128, 5)\n' +
                                  f'imLabel: {im_label.shape}, expected (128, 128)')
                            break

                        # if self.save_nifty:
                        #     label_filename = label_filename.replace('.mat', '.nii.gz')
                        #     nifti_im = nib.Nifti1Image(im_stack)
                        # else:
                        savemat(os.path.join(patch_path, label_filename), {'imStack': im_stack, 'imLabel': im_label}, do_compression=True)
            
            # Double positives for training
            if double_positives and group == 'Train':
                self.extract_more_around_target(vol, vol_mask, xyz_location, dataset, filename, patch_path)
        elif group == 'Test':
            filename = filename.split('.')[0]
            save_name = f'{dataset}_{filename}'
            output_dir = f'{patch_path}/{save_name}'
            # os.makedirs(output_dir, exist_ok=True)
            if self.save_nifty:
                nifti_im = nib.Nifti1Image(vol, affine=np.eye(4))
                nib.save(nifti_im, os.path.join(output_dir, 'image.nii.gz'))
            else:
                savemat(os.path.join(patch_path, f'{save_name}.mat'), {'vol': vol}, do_compression=True)
    
    def split_data(self, is_optimized=False):
        data = pd.read_excel(self.data_group_path)
        self.create_dataset_folders()

        if not is_optimized:
            for i in tqdm(range(len(data))):
                dataset = data.loc[i, 'Data Set']
                filename = data.loc[i, 'Name']
                group = data.loc[i, 'Group']
                self.process_file((dataset, filename, group))
        else:
            # batch_size = 10
            num_workers = 10
            pool = multiprocessing.Pool(processes=num_workers)
            args_list = [(data.loc[i, 'Data Set'], data.loc[i, 'Name'], data.loc[i, 'Group']) for i in range(len(data))]
            list(tqdm(pool.imap_unordered(self.process_file, args_list),
                      total=len(args_list),
                      desc='Processing files',
                      leave=True,
                      ncols=100))
            # args_batches = [data.iloc[i:i+batch_size] for i in range(0, len(data), batch_size)]
    
            # for batch in tqdm(args_batches, desc='Processing files in batches', leave=True, ncols=100):
            #     args_list = [(row['Data Set'], row['Name'], row['Group']) for _, row in batch.iterrows()]
            #     list(tqdm(pool.imap_unordered(self.process_file, args_list), total=len(args_list), desc='Processing batch', leave=True, ncols=100))
           
            pool.close()
            pool.join()

        print('Patches saved to ', self.output_data_path)    
            
# if __name__ == '__main__':
#     start_time = time.time()
#     start_time_str = time.strftime("%Y/%m/%d %H:%M", time.localtime(start_time))
#     print('-------------------------------------')
#     print('Running Image Processing script at', start_time_str, '\n')
    
#     image_processor = ImageProcesser(
#         data_group_path='Project/DataGroup.xlsx',
#         location_tbl_path='Project/LocationTbl.csv',
#         input_data_path='Project/InputData',
#         output_data_path='Project/DataPatches/Data3',
#         method='fuzzy',
#         save_nifty=False,
#         orientation='Cor'
#     )
#     image_processor.split_data(is_optimized=True)
    
#     print('Data patches are saved to ', image_processor.output_data_path)
    
#     end_time = time.time()
#     end_time_str = time.strftime("%Y/%m/%d %H:%M", time.localtime(end_time))
#     print('Done Splitting Data at ', end_time_str, '!')

if __name__ == '__main__':
    start_time = time.time()
    start_time_str = time.strftime("%Y/%m/%d %H:%M", time.localtime(start_time))
    print('-------------------------------------')
    print('Running Image Processing script at', start_time_str, '\n')
    
    image_processor = ImageProcesser(
        data_group_path='Project/DataGroup.xlsx',
        location_tbl_path='Project/LocationTbl.csv',
        input_data_path='Project/InputData',
        output_data_path=r'Project\DataPatches\Data3\TrainWarped',
        method='fuzzy',
        save_nifty=False,
        orientation='Cor'
    )
    warp_folder = r'Project\InputData\WarpedTrain'
    image_processor.process_warp_file(warp_folder)
    
    end_time = time.time()
    end_time_str = time.strftime("%Y/%m/%d %H:%M", time.localtime(end_time))
    print('Done Splitting Data at ', end_time_str, '!')
    


