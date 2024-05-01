# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:46:13 2024

@author: tly
"""

import subprocess
import os
# import pyautogui
# import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def open_itksnap(case_number):
    
    if case_number < 1 or case_number > 1010:
      print(f"Invalid case number: {case_number}.\nPlease enter a number between 1 and 200.")
      return
      
    case_str = f'PAT{case_number:04d}'
    data_dir = r'MiniProjects\CPRCheck\'
    case_path = os.path.join(data_dir, case_str)
    png_path = r'MiniProjects\Images_CPR_Check\'
    


    if not os.path.isdir(case_path):
      print(f"Case {case_path} not found.")
      return
      
    image = os.path.join(case_path, f'imageCpr.nii.gz')
    
    # case_path = os.path.join(data_dir+'_SpineClean', case_str)
    
    gt = os.path.join(case_path, f'vertLabelsCpr.nii.gz')
    # if not os.path.exists(gt):
    #     gt += '.nii'
    
    if not os.path.exists(image) or not os.path.exists(gt):
      print(f"Image or mask not found for case {case_number}.")
      return
      
    im = mpimg.imread(f'{png_path}/{case_str}.png')
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)  # Adjust the figure size and DPI for better quality
    ax.imshow(im)
    ax.axis('off')  # Turn off axis
    plt.title(f'{case_str}')
    plt.show()
    
    command = ['C:/Program Files/ITK-SNAP 3.8/bin/ITK-SNAP.exe', '-g', image, '-s', gt]
    subprocess.run(command)
    

while True:
  case_number = input("Enter case number (1-1010) or 'q' to quit: ")

  # Handle user input
  if case_number == 'q':
    break
  else:
    try:
      case_number = int(case_number)
      open_itksnap(case_number)
    except ValueError:
      print("Invalid input. Please enter a number or 'q'.")

print("Exiting...")
