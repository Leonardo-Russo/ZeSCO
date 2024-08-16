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
        ground_dir = os.path.join(dataset_path, 'streetview', 'panos')
    elif groundtype == 'cutouts':
        ground_dir = os.path.join(dataset_path, 'streetview', 'cutouts')
    else:   
        raise ValueError("Invalid groundtype. Choose either 'panos' or 'cutouts'.")
    satellite_dir = os.path.join(dataset_path, 'streetview_aerial')

    paired_filenames = []
    for root, _, files in os.walk(ground_dir):
        for file in files:
            if file.endswith('.jpg'):
                ground_path = os.path.join(root, file)
                lat, lon = get_metadata(ground_path)
                if lat is None or lon is None:
                    continue
                zoom = 18  # Only consider zoom level 18
                sat_path = get_aerial_path(satellite_dir, lat, lon, zoom)
                if os.path.exists(sat_path):
                    paired_filenames.append((ground_path, sat_path))
    
    num_to_select = int(len(paired_filenames) * sample_percentage)
    selected_filenames = random.sample(paired_filenames, num_to_select)
    
    random.shuffle(selected_filenames)
    split_point = int(split_ratio * len(selected_filenames))
    train_filenames = selected_filenames[:split_point]
    val_filenames = selected_filenames[split_point:]

    return train_filenames, val_filenames


def get_metadata(fname):
    if 'streetview' in fname:
        parts = fname[:-4].rsplit('/', 1)[1].split('_')
        if len(parts) == 2:
            lat, lon = parts
            return lat, lon
        elif len(parts) == 3:
            lat, lon, orientation = parts
            return lat, lon
        else:
            print(f"Unexpected filename format: {fname}")
            return None, None
    return None


def get_aerial_path(root_dir, lat, lon, zoom):
    lat_bin = int(float(lat))
    lon_bin = int(float(lon))
    return os.path.join(root_dir, f'{zoom}/{lat_bin}/{lon_bin}/{lat}_{lon}.jpg')


def extract_cutout_from_360(image, fov=(90, 60), yaw=180, pitch=90, debug=True):
    h, w = image.shape[:2]
    print(f"Pano Shape: {h}x{w}")
    x_center = (yaw / 360.0) * w
    y_center = (pitch / 180.0) * h
    fov_x = int((fov[0] / 360.0) * w)
    fov_y = int((fov[1] / 180.0) * h)

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

    def __getitem__(self, idx):
        ground_img_path, aerial_img_path = self.filenames[idx]

        ground_image = Image.open(ground_img_path).convert('RGB')
        aerial_image = Image.open(aerial_img_path).convert('RGB')

        if self.transform_ground:
            if self.cutout_from_pano:
                # Convert PIL Image to NumPy array for the cutout extraction
                ground_image_np = np.array(ground_image)
                ground_image_np = extract_cutout_from_360(ground_image_np)  # Adjust yaw and pitch as necessary
                # Convert back to PIL Image
                ground_image = Image.fromarray(ground_image_np.astype('uint8'))
            ground_image = self.transform_ground(ground_image)

        if self.transform_aerial:
            aerial_image = self.transform_aerial(aerial_image)

        return ground_image, aerial_image
    

def get_patch_embeddings(model, x):
    x = model.patch_embed(x)
    for blk in model.blocks:
        x = blk(x)
    x = model.norm(x)
    return x