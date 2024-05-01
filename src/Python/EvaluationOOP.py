# -*- coding: utf-8 -*-
"""
Created on Fri Mar 6 2024

@author: Toan Ly
"""

import os
import numpy as np
import scipy.io as sio
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import ast

class Plotter:
    def __init__(self, models_folder):
        self.models_folder = models_folder
        self.model_names = []
        self.train_losses = []
        self.val_losses = []
        self.mean_dice = []
        self.mean_dist = []

    def load_loss_data(self):
        """Load loss data from result files in each model folder"""
        for model_folder in os.listdir(self.models_folder):
            model_path = os.path.join(self.models_folder, model_folder)
            if os.path.isdir(model_path):
                self.model_names.append(model_folder)

                # Read loss data from the result file
                result_file = glob.glob(os.path.join(model_path, '*results.txt'))
                with open(result_file[0], 'r') as f:
                    data = ast.literal_eval(f.read())
                
                train_loss, val_loss = data[0]['loss'], data[0]['loss']
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                
                eval_file = os.path.join(model_path, 'EvaluationResults.csv')
                if os.path.exists(eval_file):
                    eval_result = pd.read_csv(eval_file)
                    self.mean_dice.append(eval_result['Dice Score'].mean())
                    self.mean_dist.append(eval_result['Distance'].mean())
                else:
                    self.mean_dice.append(np.nan)
                    self.mean_dist.append(np.nan)
                    
    
    def plot_loss_comparison(self, output_file=None):
        """Plot loss comparison for different models"""
        self.load_loss_data()

        plt.figure(figsize=(10, 6))
        for i, (train_loss, val_loss) in enumerate(zip(self.train_losses, self.val_losses)):
            plt.plot(np.log(train_loss), label=f'{self.model_names[i]} - Train')
            # plt.plot(np.log(val_loss), label=f'{self.model_names[i]} - Validation')

        plt.title('Log Loss Comparison for Different Models')
        plt.xlabel('Epochs')
        plt.ylabel('Log Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save or display the plot
        if output_file:
            plt.savefig(output_file)
            print('Log Loss Comparison saved to ', output_file)

        plt.show()
        
    def plot_evaluation_comparison(self, output_file=None):
        """Plot comparison of Dice scores and distances between models"""
        self.load_loss_data()

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

        # Bar plot for Dice scores
        axes[0].bar(self.model_names, self.mean_dice, color='lightcoral')
        axes[0].set_title('Average Dice Scores')
        axes[0].set_ylabel('Mean Dice Score')
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)
        axes[0].tick_params(axis='x', rotation=45)

        # Bar plot for distances
        axes[1].bar(self.model_names, self.mean_dist, color='cornflowerblue')
        axes[1].set_title('Average Distances between Centroids')
        axes[1].set_ylabel('Mean Distance')
        axes[1].grid(axis='y', linestyle='--', alpha=0.7)
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        # Save or display the plot
        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
            print('Evaluation Comparison saved to ', output_file)

        plt.show()

class Evaluation:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.results = []
        self.distances = []

    def calculate_dice(self, gt_vol, pred_vol):
        """Calculate the Dice score between ground truth and predicted volumes"""
        intersection = np.sum(np.logical_and(gt_vol, pred_vol))
        dice_score = 2 * intersection / (np.sum(gt_vol) + np.sum(pred_vol))
        return dice_score

    def calculate_distance(self, gt_centroid, pred_centroid):
        """Calculate the distance between 2 centroids"""
        if gt_centroid is None or pred_centroid is None:
            return None

        distance = np.linalg.norm(gt_centroid - pred_centroid)
        return distance

    def evaluate(self):
        """Evaluate the folder including the MAT files"""
        for file_name in tqdm(os.listdir(self.input_folder),
                              desc='Processing files',
                              leave=True,
                              ncols=100):            
            if file_name.endswith('.mat'):
                file_path = os.path.join(self.input_folder, file_name)
                data = sio.loadmat(file_path)
                gt_vol, pred_vol, gt_centroid, pred_centroid = data['gt_vol'], data['pred_vol'], data['gt_centroid'], data['pred_centroid']

                dice = self.calculate_dice(gt_vol, pred_vol)
                distance = self.calculate_distance(gt_centroid, pred_centroid)

                self.results.append({
                    'File Name': file_name,
                    'Dice Score': dice,
                    'Distance': distance,
                    'Gt Centroid': gt_centroid,
                    'Pred Centroid': pred_centroid
                })
                self.distances.append(distance)

    def save_table(self, output_file):
        """Save evaluation results to a CSV file"""
        df = pd.DataFrame(self.results)
        df.to_csv(output_file, index=False)
        print('Evaluation results saved to ', output_file)

    def plot_results(self, output_file=None, input_folder=None):
        """Plot the evaluation results."""
        df = pd.DataFrame(self.results)
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

        if input_folder:
            plt.suptitle(input_folder)
        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
            print('Box plot saved to ', output_file)

        plt.show()
        

def evaluate_models(model_folders, input_folders):
    all_distances = []
    model_names = []
    for input_folder in os.listdir(input_folders):
        input_file = os.path.join(input_folders, input_folder)
        if os.path.isfile(input_file):
            continue

        print('Processing Model ', input_folder)
        evaluator = Evaluation(input_file)
        evaluator.evaluate()
    
        output_file = os.path.join(model_folders, input_folder, 'EvaluationResults.csv')
        evaluator.save_table(output_file)
    
        output_file = os.path.join(model_folders, input_folder, 'BoxPlot.png')
        evaluator.plot_results(output_file, input_folder)
        all_distances.append(evaluator.distances)
        model_names.append(input_folder[:8])
        print()
    return all_distances, model_names

def plot_box_whisker(all_distances, model_names, output_file=None):
    plt.figure(figsize=(10, 7))
    # plt.style.use('seaborn-darkgrid')
    # plt.boxplot(all_distances, 
    #             labels=[f'Model {i+1}' for i in range(len(all_distances))],
    #             patch_artist=True,
    #             medianprops={"linewidth": 2, "color": "orange"},
    #             boxprops={"facecolor": "lightgray", "edgecolor": "black"},  
    #             flierprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "black"}
    #             )
    
    # plt.title('Box and Whisker Plot of Distances between Centroids', fontsize=16)
    # plt.xlabel('Model', fontsize=14)
    # plt.ylabel('Distance', fontsize=14)
    # plt.grid(True, linestyle='--', alpha=0.7) 
     
    sns.boxplot(data=all_distances,
                palette=sns.color_palette("hls", len(all_distances)),
                showmeans=True,
                )
    plt.title('Model Comparison by Distances between Centroids')
    plt.ylabel('Distances between centroids')
    plt.xlabel('Model')
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    if output_file:
        plt.savefig(output_file, bbox_inches='tight')
        print('Box plot saved to ', output_file)

def create_medians_table(all_distances, model_names, output_file):
    medians = np.median(all_distances, axis=1).round(2).reshape(1, 12)
    table = pd.DataFrame(medians, columns=model_names)
    table.index = ['Median']
    table.to_csv(output_file)
    print(table)


if __name__ == '__main__':
    model_folders = r'Project\Models'
    input_folders = r'Project/InferenceOutput'

    all_distances, model_names = evaluate_models(model_folders, input_folders)
    
    
    output_file = os.path.join(input_folders, 'DistancesComparison.png')
    plot_box_whisker(all_distances, model_names, output_file)
    
    zoom_in_models = [2, 3, 6, 10, 11, 12]
    new_distances = [all_distances[i-1] for i in zoom_in_models]
    new_model_names = [model_names[i-1] for i in zoom_in_models]
    output_file = os.path.join(input_folders, 'DetailedDistancesComparison.png')
    plot_box_whisker(new_distances, new_model_names, output_file)
    
    output_file = os.path.join(input_folders, 'MedianComparison.csv')
    create_medians_table(all_distances, model_names, output_file)
    # plotter = Plotter(model_folders)
    # output_file = os.path.join(input_folders, 'LogLossComparison.png')
    # plotter.plot_loss_comparison(output_file)
    
    # output_file = os.path.join(input_folders, 'MetricsComparison.png')
    # plotter.plot_evaluation_comparison(output_file)

    print('Done with Evaluation!')
    
    
    
  
    
