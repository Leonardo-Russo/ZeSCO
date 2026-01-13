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
import pickle


def sample_cvusa_images(dataset_path, sample_percentage=0.2, split_ratio=0.8, groundtype='panos', zoom_level=19, shortcut=False):
    """
    Function to sample a percentage of the dataset and split it into training and validation sets.
    
    Parameters:
        dataset_path (str): Path to the dataset root directory.
        sample_percentage (float): Percentage of the dataset to sample.
        split_ratio (float): Ratio to split the sampled data into training and validation sets.
        groundtype (str): Type of ground images ('panos' or 'cutouts').
        zoom_level (int): Zoom level for satellite images (default 19).
        shortcut (bool): If True, limits the dataset to 10000 * sample_percentage images.
        
    Returns:
        train_filenames (list): List of training filenames (tuples of panorama and satellite image paths).
        val_filenames (list): List of validation filenames (tuples of panorama and satellite image paths).
    """

    if 'CVUSA' in dataset_path:
        # Read paired filenames from CSV files
        train_csv = os.path.join(dataset_path, 'splits', 'train-19zl.csv')
        val_csv = os.path.join(dataset_path, 'splits', 'val-19zl.csv')
        
        # Check if CSV files exist
        if not os.path.exists(train_csv) or not os.path.exists(val_csv):
            raise FileNotFoundError(f"CSV files not found. Expected files:\n{train_csv}\n{val_csv}")
        
        # Read training CSV
        train_filenames = []
        with open(train_csv, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) >= 2:
                    sat_path = os.path.join(dataset_path, parts[0])
                    ground_path = os.path.join(dataset_path, parts[1])
                    if os.path.exists(sat_path) and os.path.exists(ground_path):
                        train_filenames.append((ground_path, sat_path))
        
        # Read validation CSV
        val_filenames = []
        with open(val_csv, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) >= 2:
                    sat_path = os.path.join(dataset_path, parts[0])
                    ground_path = os.path.join(dataset_path, parts[1])
                    if os.path.exists(sat_path) and os.path.exists(ground_path):
                        val_filenames.append((ground_path, sat_path))
        
        return train_filenames, val_filenames
    
    if groundtype == 'panos':
        ground_dir = os.path.join(dataset_path, 'streetview', 'panos')
    else:   
        raise ValueError("Invalid groundtype. Choose either 'panos' or 'cutouts'.")
    satellite_dir = os.path.join(dataset_path, 'bingmap')

    if shortcut:
        num_to_select = int(10000 * sample_percentage)

    paired_filenames = []
    for root, _, files in os.walk(ground_dir):
        for file in files:
            if file.endswith('.jpg'):
                ground_path = os.path.join(root, file)
                image_id = os.path.splitext(file)[0]                
                if image_id is None:
                    continue

                sat_path = os.path.join(satellite_dir, f'{zoom_level}/{image_id}.jpg')
                if os.path.exists(sat_path):
                    paired_filenames.append((ground_path, sat_path))
            if shortcut:
                if len(paired_filenames)*sample_percentage >= num_to_select:
                    break

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
    
    x_center = ((-yaw % 360) / 360.0) * w
    y_center = (pitch / 180.0) * h
    fov_x = int((fov[0] / 360.0) * w)
    fov_y = int((fov[1] / 180.0) * h)

    if debug:
        print(f"Center coordinates: x={x_center}, y={y_center}")
        print(f"FOV: {fov_x}x{fov_y}")
    
    # Calculate bounds for horizontal (with wrapping) and vertical (with clamping)
    x1 = int(x_center - fov_x / 2)
    x2 = int(x_center + fov_x / 2)
    y1 = int(y_center - fov_y / 2)
    y2 = int(y_center + fov_y / 2)
    
    # Clamp vertical coordinates (no wrapping for pitch)
    y1_clamped = max(0, y1)
    y2_clamped = min(h, y2)
    
    if debug:
        print(f"Cutout coordinates (before wrapping): x1={x1}, x2={x2}, y1={y1}, y2={y2}")
        print(f"Clamped y coordinates: y1={y1_clamped}, y2={y2_clamped}")
    
    # Handle horizontal wrapping (cylindrical projection)
    if x1 < 0:
        # Wraps around the left edge
        left_part = image[y1_clamped:y2_clamped, x1 % w:]
        right_part = image[y1_clamped:y2_clamped, :x2]
        cutout = np.concatenate([left_part, right_part], axis=1)
    elif x2 > w:
        # Wraps around the right edge
        left_part = image[y1_clamped:y2_clamped, x1:w]
        right_part = image[y1_clamped:y2_clamped, :x2 % w]
        cutout = np.concatenate([left_part, right_part], axis=1)
    else:
        # No wrapping needed
        cutout = image[y1_clamped:y2_clamped, x1:x2]
    
    if debug:
        # Draw the rectangle on the original image (show wrapping if applicable)
        fig, ax = plt.subplots(1, figsize=(10, 5))
        ax.imshow(image)
        
        if x1 < 0 or x2 > w:
            # Wrapping case - draw two rectangles
            if x1 < 0:
                # Left wrap
                rect1 = patches.Rectangle((x1 % w, y1_clamped), w - (x1 % w), y2_clamped-y1_clamped,
                                          linewidth=2, edgecolor='r', facecolor='none')
                rect2 = patches.Rectangle((0, y1_clamped), x2, y2_clamped-y1_clamped,
                                          linewidth=2, edgecolor='r', facecolor='none')
            else:
                # Right wrap
                rect1 = patches.Rectangle((x1, y1_clamped), w - x1, y2_clamped-y1_clamped,
                                          linewidth=2, edgecolor='r', facecolor='none')
                rect2 = patches.Rectangle((0, y1_clamped), x2 % w, y2_clamped-y1_clamped,
                                          linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect1)
            ax.add_patch(rect2)
        else:
            # Normal case
            rect = patches.Rectangle((x1, y1_clamped), x2-x1, y2_clamped-y1_clamped,
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        ax.axis('off')
        
        # plt.title('Cutout Region from Original Image')
        print("Displaying cutout region on original image.")
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
    def __init__(self, filenames, transform_aerial=None, transform_ground=None, cutout_from_pano=True, apply_polar_transform=False, image_size=224, fov_x=90, fov_y=180, debug=False):
        self.filenames = filenames
        self.transform_aerial = transform_aerial
        self.transform_ground = transform_ground
        self.cutout_from_pano = cutout_from_pano
        self.apply_polar_transform = apply_polar_transform
        self.image_size = image_size
        self.fov_x = fov_x
        self.fov_y = fov_y
        self.debug = debug

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx, debug=False):
        ground_img_path, aerial_img_path = self.filenames[idx]

        ground_image = Image.open(ground_img_path).convert('RGB')
        aerial_image = Image.open(aerial_img_path).convert('RGB')

        # Choose Cropping Parameters
        fov = (self.fov_x, self.fov_y)  # set fov
        if debug:
            yaw = int(input("Enter yaw value (or press Enter to use random value): "))
        else:
            yaw = random.randint(0, 360)    # random yaw between 0 and 360 degrees
        pitch = 90                      # fixed pitch at 90 degrees

        if self.transform_ground:
            if self.cutout_from_pano:
                ground_image_np = np.array(ground_image)                            # Convert PIL Image to NumPy array for the cutout extraction
                ground_image_np = extract_cutout_from_360(ground_image_np, fov, yaw, pitch, self.debug)
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
    

def get_transforms(processor, image_size, aerial_scaling, crop_percentage, horizontal_scaling=1.0):

    threshold = 0.05
    if crop_percentage > threshold:
        bottom_crop = threshold
        top_crop = crop_percentage - bottom_crop
    else:
        top_crop = crop_percentage - threshold
        bottom_crop = threshold

    bottom_crop = crop_percentage
    top_crop = crop_percentage
        
    if isinstance(processor, tuple):
        processor_ground, processor_aerial = processor
    else:
        processor_ground = processor
        processor_aerial = processor

    # Check if CLIP-style processor or ViT-style processor
    is_clip_processor = hasattr(processor_ground, 'image_processor')
    
    if is_clip_processor:
        # CLIP processors
        mean_ground = processor_ground.image_processor.image_mean
        std_ground = processor_ground.image_processor.image_std
        mean_aerial = processor_aerial.image_processor.image_mean
        std_aerial = processor_aerial.image_processor.image_std
    else:
        # ViT processors (DinoV2, DinoV3)
        mean_ground = processor_ground.image_mean
        std_ground = processor_ground.image_std
        mean_aerial = processor_aerial.image_mean
        std_aerial = processor_aerial.image_std

    # Ground transform - CLIP needs different resize strategy
    if is_clip_processor:
        # CLIP: resize shortest edge (maintains aspect ratio), then crop
        # Note: We multiply by 255 after ToTensor() to match do_rescale=False behavior
        transform_ground = transforms.Compose([
            transforms.Resize((image_size, image_size*horizontal_scaling), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Lambda(lambda img: transforms.functional.crop(img, int(img.size[1] * top_crop), 0, int(img.size[1] * (1 - top_crop - bottom_crop)), img.size[0])),
            transforms.Resize((image_size, image_size*horizontal_scaling), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_ground, std=std_ground)
        ])
    else:
        # ViT: direct resize to exact dimensions (can distort aspect ratio)
        transform_ground = transforms.Compose([
            transforms.Resize((image_size, image_size*horizontal_scaling), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Lambda(lambda img: transforms.functional.crop(img, int(img.size[1] * top_crop), 0, int(img.size[1] * (1 - top_crop - bottom_crop)), img.size[0])),
            transforms.Resize((image_size, image_size*horizontal_scaling), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_ground, std=std_ground)
        ])

    # Aerial transform
    if is_clip_processor:
        # CLIP: resize shortest edge, then center crop
        # Note: We multiply by 255 after ToTensor() to match do_rescale=False behavior
        transform_aerial = transforms.Compose([
            transforms.Resize((image_size*aerial_scaling, image_size*aerial_scaling), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_aerial, std=std_aerial)
        ])
    else:
        # ViT: direct resize
        transform_aerial = transforms.Compose([
            transforms.Resize((image_size*aerial_scaling, image_size*aerial_scaling), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_aerial, std=std_aerial)
        ])

    return transform_ground, transform_aerial


def denormalize(img_tensor, processor):
    if hasattr(processor, 'image_processor'):
        mean = torch.tensor(processor.image_processor.image_mean).view(3, 1, 1).to(img_tensor.device)
        std = torch.tensor(processor.image_processor.image_std).view(3, 1, 1).to(img_tensor.device)
    else:
        mean = torch.tensor(processor.image_mean).view(3, 1, 1).to(img_tensor.device)
        std = torch.tensor(processor.image_std).view(3, 1, 1).to(img_tensor.device)
    return img_tensor * std + mean
