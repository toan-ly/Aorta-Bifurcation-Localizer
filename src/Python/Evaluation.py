# -*- coding: utf-8 -*-
"""
Created on Fri Mar 1 2024

@author: Toan Ly

Riverain Tech 2024

"""

import os
import numpy as np
import scipy.io as sio
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_dice(gt_vol, pred_vol):
    """Calculate the Dice score between ground truth and predicted volumes"""
    intersection = np.sum(np.logical_and(gt_vol, pred_vol))
    dice_score = 2 * intersection / (np.sum(gt_vol) + np.sum(pred_vol))
    return dice_score

def calculate_distance(gt_centroid, pred_centroid):
    """Calculate the distance between 2 centroids"""
    if gt_centroid is None or pred_centroid is None:
        return None

    distance = np.linalg.norm(gt_centroid - pred_centroid)
    return distance

def evaluate(input_folder):
    """Evaluate the folder including the MAT files"""
    res = []
    for file_name in tqdm(os.listdir(input_folder), desc='Processing files'):
        if file_name.endswith('.mat'):
            file_path = os.path.join(input_folder, file_name)
            data = sio.loadmat(file_path)
            gt_vol, pred_vol, gt_centroid, pred_centroid = data['gt_vol'], data['pred_vol'], data['gt_centroid'], data['pred_centroid']
    
            dice = calculate_dice(gt_vol, pred_vol)
            distance = calculate_distance(gt_centroid, pred_centroid)
    
            res.append({
                'File Name': file_name,
                'Dice Score': dice,
                'Distance': distance
            })

    return res

def save_table(res, output_file):
    """Save evaluation results to a CSV file"""
    df = pd.DataFrame(res)
    df.to_csv(output_file, index=False)
    print('Evaluation results saved to ', output_file)

def plot_results(res, output_file=None):
    """Plot the evaluation results."""
    df = pd.DataFrame(res)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    
    # Box plot for Dice scores
    sns.boxplot(data=df, y='Dice Score', ax=axes[0], color='skyblue')
    axes[0].set_title('Dice Scores')
    axes[0].set_ylabel('Dice Score')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Box plot for distances
    sns.boxplot(data=df, y='Distance', ax=axes[1], color='lightgreen')
    axes[1].set_title('Distances between Centroids')
    axes[1].set_ylabel('Distance')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, bbox_inches='tight')
        print('Plot saved to ', output_file)
        
    plt.show()
    
if __name__ == '__main__':
    input_folder = r'Project\InferenceOutput\OutputData5'
    evaluation_results = evaluate(input_folder)

    output_file = os.path.join(input_folder, 'EvaluationResults.csv')
    save_table(evaluation_results, output_file)

    output_file = os.path.join(input_folder, 'BoxPlot.png')
    plot_results(evaluation_results, output_file)
    
    print('Done with Evaluation!')

