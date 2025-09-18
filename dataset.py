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
import matplotlib.patches as patches
import random
from transformers import ViTImageProcessor, AutoModel


def sample_cvusa_images(dataset_path, sample_percentage=0.2, split_ratio=0.8, groundtype='panos'):
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


def sample_cities_images(dataset_path, sample_percentage=0.2, split_ratio=0.8, ground_subdir=None, satellite_subdir=None):
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

    cities = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

    if ground_subdir is None or satellite_subdir is None:
        if dataset_path == '/home/lrusso/CV-Cities':
            ground_subdir = 'pano_images'
            satellite_subdir = 'sat_images'
        elif dataset_path == '/home/lrusso/CV-GLOBAL':
            ground_subdir = 'streetview2048'
            satellite_subdir = 'satellite'

    n_selected_cities = int(np.ceil(len(cities) * sample_percentage))
    selected_cities = random.sample(cities, n_selected_cities)
        
    paired_filenames = []
    for city in selected_cities:
        ground_dir = os.path.join(dataset_path, city, ground_subdir)
        satellite_dir = os.path.join(dataset_path, city, satellite_subdir)

        for root, _, files in os.walk(ground_dir):
            for file in files:
                if file.endswith('.jpg'):
                    ground_path = os.path.join(root, file)
                    image_id = os.path.splitext(file)[0]                
                    if image_id is None:
                        continue
                    sat_path = os.path.join(satellite_dir, f'{image_id}.jpg')
                    if os.path.exists(sat_path):
                        paired_filenames.append((ground_path, sat_path))
    
    random.shuffle(paired_filenames)
    split_point = int(split_ratio * len(paired_filenames))
    train_filenames = paired_filenames[:split_point]
    val_filenames = paired_filenames[split_point:]

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
        fig, ax = plt.subplots(1, figsize=(10, 5))
        ax.imshow(image)
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.title('Cutout Region')
        plt.show()

    return cutout


def polar_transform(image, target_size):
    """
    Applies a polar transformation to the aerial image to match the dimensions of the ground image.

    Args:
    - image: The input aerial image (PIL Image).
    - target_size: The desired output size (height, width) after the polar transformation.

    Returns:
    - transformed_image: The polar-transformed aerial image (PIL Image).
    """

    # Convert PIL Image to NumPy array
    image_np = np.array(image)

    # Rearrange the shape from (3, height, width) to (height, width, 3)
    if image_np.shape[0] == 3:
        image_np = np.transpose(image_np, (1, 2, 0))

    # Get the original image size and the target size
    Sa = image_np.shape[0]  # Assuming square aerial image
    Hg, Wg = target_size

    # Create the polar transformed image
    transformed_image_np = np.zeros((Hg, Wg, 3), dtype=np.uint8)  # 3 channels for RGB

    for i in range(Hg):
        for j in range(Wg):
            # Calculate the corresponding coordinates in the original aerial image
            xa = int(Sa / 2 - (Sa / 2) * ((Hg - i) / Hg) * np.cos(2 * np.pi * j / Wg))
            ya = int(Sa / 2 + (Sa / 2) * ((Hg - i) / Hg) * np.sin(2 * np.pi * j / Wg))

            # Ensure coordinates are within bounds
            xa = max(0, min(xa, Sa - 1))
            ya = max(0, min(ya, Sa - 1))

            # Copy the pixel value from the original to the transformed image
            transformed_image_np[i, j] = image_np[ya, xa]

    # Make sure all values are within valid range before conversion to PIL Image
    transformed_image_np = np.clip(transformed_image_np, 0, 255).astype(np.uint8)
    
    # Convert NumPy array back to PIL Image
    transformed_image = Image.fromarray(transformed_image_np)

    return transformed_image


class PairedImagesDataset(Dataset):
    def __init__(self, filenames, transform_aerial=None, transform_ground=None, cutout_from_pano=True, apply_polar_transform=False, image_size=224):
        self.filenames = filenames
        self.transform_aerial = transform_aerial
        self.transform_ground = transform_ground
        self.cutout_from_pano = cutout_from_pano
        self.apply_polar_transform = apply_polar_transform
        self.image_size = image_size

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
            if self.apply_polar_transform:
                transform_aerial = transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.CenterCrop((self.image_size, self.image_size))
                ])
                to_tensor = transforms.ToTensor()
                aerial_image = transform_aerial(aerial_image)
                aerial_image = polar_transform(aerial_image, (self.image_size, self.image_size))
                aerial_image = to_tensor(aerial_image)
            else:
                aerial_image = self.transform_aerial(aerial_image)

        return ground_image, aerial_image, fov, yaw, pitch
    

def get_transforms(processor, image_size, aerial_scaling):
        
    if isinstance(processor, tuple):
        processor_ground, processor_aerial = processor
    else:
        processor_ground = processor
        processor_aerial = processor

    transform_ground = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=processor_ground.image_mean,
            std=processor_ground.image_std
        )
    ])

    transform_aerial = transforms.Compose([
        transforms.Resize((image_size*aerial_scaling, image_size*aerial_scaling), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor_aerial.image_mean,
                            std=processor_aerial.image_std)
    ])

    return transform_ground, transform_aerial


def denormalize(img_tensor, processor):
    mean = torch.tensor(processor.image_mean).view(3, 1, 1).to(img_tensor.device)
    std = torch.tensor(processor.image_std).view(3, 1, 1).to(img_tensor.device)
    return img_tensor * std + mean
