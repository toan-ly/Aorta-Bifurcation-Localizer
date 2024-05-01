import os
import glob
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.io as sio
import tensorflow as tf
from tensorflow.keras.models import load_model
import nibabel as nib
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

class Inference:
    def __init__(self, model_path, location_tbl_path, input_dir, output_dir, save_nifty=False, window_rng=(-1150, 350), target_method='fuzzy', centroid_method='mean', orientation='Ax'):
        self.model = load_model(model_path)
        self.tbl = pd.read_csv(location_tbl_path)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.save_nifty = save_nifty
        self.target_method = target_method
        self.centroid_method = centroid_method
        self.window_rng = window_rng
        self.threshold = 0.5
        self.stride = (64, 64, 1)
        self.orientation = orientation
        
        # GPU set-up
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

    def process_test_dir(self, is_optimized=False):
        """Process the whole test directory"""
        os.makedirs(self.output_dir + '/MATLAB', exist_ok=True)
        os.makedirs(self.output_dir + '/NIFTI', exist_ok=True)
        image_files = glob.glob(os.path.join(self.input_dir, '*.mat'))
        if is_optimized:
            batch_size = 10
            num_workers = 10
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                   for i in range(0, len(image_files), batch_size):
                       batch_files = image_files[i:i+batch_size]
                       futures = [executor.submit(self.process_image_file, image_file) for image_file in batch_files]
                       for future in tqdm(as_completed(futures), 
                                          total=len(futures),
                                          desc=f'Processing batch {i+1}-{min(i+batch_size, len(image_files))}',
                                          leave=True,
                                          ncols=100):
                           pass
        else:
            with tqdm(total=len(image_files), desc='Processing images', leave=True, ncols=100) as pbar:
                for image_file in image_files:
                    pbar.set_postfix(file=os.path.basename(image_file))
                    self.process_image_file(image_file)
                    pbar.update(1)
     
    def get_aorta_bifurcation(self, dataset, filename):
        location_idx = np.where((self.tbl['dataset'] == dataset) & (self.tbl['pat'] == filename))[0]
        location_cols = ['xyzAortaBifur_3', 'xyzAortaBifur_1', 'xyzAortaBifur_2']
        # location_cols = [location_cols[i] for i in axes]
        
        xyz_location = self.tbl.loc[location_idx, location_cols].values[0]
        return xyz_location
    
    def process_image_file(self, image_file):
        """Process each image file, do inference, and save as mat file"""
        axes_order = {'Ax': (0, 1, 2), 'Cor': (2, 1, 0), 'Sag': (2, 0, 1)}
        axes = axes_order[self.orientation]
        
        vol = sio.loadmat(image_file)['vol']
        vol = np.transpose(vol, axes)
        
        pred_vol, pred_vol_post, pred_centroid = self.inference(vol)

        file_name = os.path.basename(image_file)
        dataset, pat = file_name.split('_')
        
        xyz_location = self.get_aorta_bifurcation(dataset, pat)
        gt_vol = self.create_target_vol(vol, xyz_location)

        output_file = os.path.join(self.output_dir, 'MATLAB', file_name)
        
        print('Gt shape: ', gt_vol.shape)
        print('Pred shape: ', pred_vol.shape)
        print('Gt centroid: ', xyz_location)
        print('Pred centroid: ', pred_centroid)
        sio.savemat(output_file, {
            'gt_vol': gt_vol,
            'pred_vol': pred_vol,
            'pred_vol_post': pred_vol_post,
            'gt_centroid': xyz_location,
            'pred_centroid': pred_centroid},
            do_compression=True
        )
        
        if self.save_nifty:
            output_file = os.path.join(self.input_dir, file_name[:-4])
            nib.save(nib.Nifti1Image(pred_vol.astype(np.float32), np.eye(4)), output_file + '/pred.nii.gz')
            nib.save(nib.Nifti1Image(pred_vol_post.astype(np.float32), np.eye(4)), output_file + '/pred_post.nii.gz')
            nib.save(nib.Nifti1Image(gt_vol.astype(np.float32), np.eye(4)), output_file + '/gt.nii.gz')

    def create_target_vol(self, vol, location):
        """Create target vol with sphere at aorta bifurcation point"""
        # Create a blank space
        vol_mask = np.zeros_like(vol)

        aorta_bifur = location - 1  # MATLAB starts with 1, Python starts with 0
        x, y, z = np.meshgrid(*[np.arange(dim) for dim in vol.shape], indexing='ij')

        if self.target_method == 'sphere':
            vol_mask = (x - aorta_bifur[0]) ** 2 + (y - aorta_bifur[1]) ** 2 + (z - aorta_bifur[2]) ** 2 < 100
        elif self.target_method == 'fuzzy':
            sigma = 7
            distances = np.sqrt((x - aorta_bifur[0]) ** 2 + (y - aorta_bifur[1]) ** 2 + (z - aorta_bifur[2]) ** 2)
            gaussian = np.exp(-0.5 * (distances ** 2) / (sigma ** 2))
            gaussian = (gaussian - np.min(gaussian)) / (np.max(gaussian) - np.min(gaussian))
            gaussian[gaussian < 0.1] = 0
            vol_mask = gaussian

        return vol_mask

    def preprocess(self, vol):
        """Preprocess input CT vol"""
        vol = (vol - self.window_rng[0]) / (self.window_rng[1] - self.window_rng[0])
        vol = np.clip(vol, 0, 1)

        return vol

    def postprocess(self, pred_vol):
        """Postprocess predicted vol"""
        pred_vol_post = (pred_vol > self.threshold).astype(np.uint8)
        centroid = self.get_centroid(pred_vol)

        return pred_vol_post, centroid

    def get_centroid(self, pred_vol):
        if self.centroid_method == 'mean':
            mask = pred_vol > self.threshold
            x, y, z = np.where(mask)
            centroid_x = np.mean(x)
            centroid_y = np.mean(y)
            centroid_z = np.mean(z)
            centroid = np.array([centroid_x, centroid_y, centroid_z])
        elif self.centroid_method == 'peak':
            peak_idx = np.unravel_index(np.argmax(pred_vol), pred_vol.shape)
            centroid = np.array([peak_idx[0], peak_idx[1], peak_idx[2]])
        else:
            centroid = np.array([np.nan, np.nan, np.nan])
            
        return centroid

    def prepad_vol(self, vol, num_slices):
        """Pad input CT vol for first and last slices with replication"""
        padded_vol = np.pad(vol, ((0, 0), (0, 0), (num_slices, num_slices)), mode='edge')

        return padded_vol

    def extract_patches(self, vol, patch_sz=(128, 128, 5)):
        """Extract patches from input vol"""
        vol = self.prepad_vol(vol, 2)
        num_patches = [int(np.ceil((vol.shape[i] - patch_sz[i]) / self.stride[i])) + 1 for i in range(3)]
        
        patches = []
        # Extract patches
        for x in range(num_patches[0]):
            for y in range(num_patches[1]):
                for z in range(num_patches[2]):
                    start = [x * self.stride[0], y * self.stride[1], z * self.stride[2]]
                    end = [min(start[i] + patch_sz[i], vol.shape[i]) for i in range(3)]
                    start = [end[i] - patch_sz[i] for i in range(3)]  # Ensure patches are of equal size

                    patch = vol[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
                    patches.append(patch)

        return np.array(patches)

    def predict(self, patches):
        """Predict using trained model"""
        if self.target_method == 'sphere':
            return self.model.predict(patches)[..., 1][..., np.newaxis] 
        return self.model.predict(patches)

    def assemble_patches(self, patches, vol_shape, patch_sz=(128, 128, 1)):
        """Assemble patches back into output 3D CT vol"""
        pred_vol = np.zeros(vol_shape)
        num_patches = [int(np.ceil((vol_shape[i] - patch_sz[i]) / self.stride[i])) + 1 for i in range(3)]
        idx = 0
        for x in range(num_patches[0]):
            for y in range(num_patches[1]):
                for z in range(num_patches[2]):
                    start = [x * self.stride[0], y * self.stride[1], z * self.stride[2]]
                    end = [min(start[i] + patch_sz[i], vol_shape[i]) for i in range(3)]
                    start = [end[i] - patch_sz[i] for i in range(3)]

                    pred_vol[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = np.maximum(
                        pred_vol[start[0]:end[0], start[1]:end[1], start[2]:end[2]], patches[idx]
                    )
                    idx += 1

        return pred_vol

    def inference(self, vol):
        """Inference code for each input vol"""
        preprocessed_vol = self.preprocess(vol)
        patches = self.extract_patches(preprocessed_vol)
        predictions = self.predict(patches)
        pred_vol = self.assemble_patches(predictions, vol.shape)
        pred_vol_post, centroid = self.postprocess(pred_vol)

        return pred_vol, pred_vol_post, centroid

# -----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------- 
if __name__ == '__main__':
    start_time = time.time()
    start_time_str = time.strftime("%Y/%m/%d %H:%M", time.localtime(start_time))
    print('-------------------------------------')
    print('Running Inference script at', start_time_str, '\n')

    inference = Inference(
        model_path=r'Project/Models/Model4-2/2024-04-22-TF2.5.0-Net-CP005-8.568E-05-3.641E-04.h5',
        location_tbl_path=r'Project/LocationTbl.csv',
        input_dir=r'Project\DataPatches\Data3\Test',
        output_dir=r'Project\InferenceOutput\Model4-2',
        target_method='fuzzy',
        centroid_method='peak',
        window_rng=(-500, 700),
        save_nifty=False, 
    )

    inference.process_test_dir(is_optimized=True)

    print('Inference outputs are saved to ', inference.output_dir)

    end_time = time.time()
    end_time_str = time.strftime("%Y/%m/%d %H:%M", time.localtime(end_time))
    print('\nDone with Inference at', end_time_str, '!')

