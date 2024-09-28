from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
import sys
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import random


def sample_paired_images(dataset_path, sample_percentage=0.2, split_ratio=0.8, groundtype='panos'):
    """
    Function to sample a percentage of the dataset and split it into training and validation sets.
    
    Parameters:
        dataset_path (str): Path to the dataset root directory.
        sample_percentage (float): Percentage of the dataset to sample.
        split_ratio (float): Ratio to split the sampled data into training and validation sets.
        
    Returns:
        train_filenames (list): List of training filenames (tuples of panorama and satellite image paths).
        val_filenames (list): List of validation filenames (tuples of panorama and satellite image paths).
    """
    
    if groundtype == 'panos':
        print(groundtype)
        ground_dir = os.path.join(dataset_path, 'streetview', 'panos')
    else:   
        raise ValueError("Invalid groundtype. Choose either 'panos' or 'cutouts'.")
    satellite_dir = os.path.join(dataset_path, 'bingmap')

    paired_filenames = []
    for root, _, files in os.walk(ground_dir):
        for file in files:
            if file.endswith('.jpg'):
                ground_path = os.path.join(root, file)
                image_id = os.path.splitext(file)[0]                
                if image_id is None:
                    continue

                zoom = 18  # Assuming zoom level 18
                sat_path = os.path.join(satellite_dir, f'{zoom}/{image_id}.jpg')
                if os.path.exists(sat_path):
                    paired_filenames.append((ground_path, sat_path))
    
    num_to_select = int(len(paired_filenames) * sample_percentage)
    selected_filenames = random.sample(paired_filenames, num_to_select)
    
    random.shuffle(selected_filenames)
    split_point = int(split_ratio * len(selected_filenames))
    train_filenames = selected_filenames[:split_point]
    val_filenames = selected_filenames[split_point:]

    return train_filenames, val_filenames


def extract_cutout_from_360(image, fov=(90, 180), yaw=180, pitch=90, debug=False):
    h, w = image.shape[:2]
    if debug:
        print(f"Pano Shape: {h}x{w}")
    x_center = (yaw / 360.0) * w
    y_center = (pitch / 180.0) * h
    fov_x = int((fov[0] / 360.0) * w)
    fov_y = int((fov[1] / 180.0) * h)

    if debug:
        print(f"Center coordinates: x={x_center}, y={y_center}")
        print(f"FOV: {fov_x}x{fov_y}")
    
    x1 = int(x_center - fov_x / 2)
    x2 = int(x_center + fov_x / 2)
    y1 = int(y_center - fov_y / 2)
    y2 = int(y_center + fov_y / 2)
    
    # Ensure coordinates are within image bounds
    x1 = max(0, x1)
    x2 = min(w, x2)
    y1 = max(0, y1)
    y2 = min(h, y2)
    
    if debug:
        print(f"Cutout coordinates: x1={x1}, x2={x2}, y1={y1}, y2={y2}, image shape: {image.shape}")
    
    cutout = image[y1:y2, x1:x2]
    
    if debug:
        # Draw the rectangle on the original image
        import matplotlib.patches as patches
        fig, ax = plt.subplots(1, figsize=(10, 5))
        ax.imshow(image)
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.title('Cutout Region')
        plt.show()

    return cutout


class PairedImagesDataset(Dataset):
    def __init__(self, filenames, transform_aerial=None, transform_ground=None, cutout_from_pano=True):
        self.filenames = filenames
        self.transform_aerial = transform_aerial
        self.transform_ground = transform_ground
        self.cutout_from_pano = cutout_from_pano

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx, debug=False):
        ground_img_path, aerial_img_path = self.filenames[idx]

        ground_image = Image.open(ground_img_path).convert('RGB')
        aerial_image = Image.open(aerial_img_path).convert('RGB')

        # Choose Cropping Parameters
        fov = (90, 180)                 # default FOV
        yaw = random.randint(0, 360)    # random yaw between 0 and 360 degrees
        pitch = 90                      # fixed pitch at 90 degrees

        if self.transform_ground:
            if self.cutout_from_pano:
                ground_image_np = np.array(ground_image)                            # Convert PIL Image to NumPy array for the cutout extraction
                ground_image_np = extract_cutout_from_360(ground_image_np, fov, yaw, pitch, debug)
                ground_image = Image.fromarray(ground_image_np.astype('uint8'))     # Convert back to PIL Image
            ground_image = self.transform_ground(ground_image)

        if self.transform_aerial:
            aerial_image = self.transform_aerial(aerial_image)

        return ground_image, aerial_image, fov, yaw, pitch