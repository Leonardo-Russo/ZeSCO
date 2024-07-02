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


class PairedImagesDataset(Dataset):
    def __init__(self, filenames, transform_aerial=None, transform_ground=None):
        self.filenames = filenames
        self.transform_aerial = transform_aerial
        self.transform_ground = transform_ground

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        ground_img_path, aerial_img_path = self.filenames[idx]

        ground_image = Image.open(ground_img_path).convert('RGB')
        aerial_image = Image.open(aerial_img_path).convert('RGB')

        if self.transform_ground:
            ground_image = self.transform_ground(ground_image)

        if self.transform_aerial:
            aerial_image = self.transform_aerial(aerial_image)

        return ground_image, aerial_image



class ModifiedNestedTensorBlock(nn.Module):
    def __init__(self, original_block):
        super(ModifiedNestedTensorBlock, self).__init__()
        self.norm1 = original_block.norm1
        self.attn = original_block.attn
        self.ls1 = original_block.ls1
        self.drop_path1 = original_block.drop_path1
        self.norm2 = original_block.norm2
        self.mlp = original_block.mlp
        self.ls2 = original_block.ls2
        self.drop_path2 = original_block.drop_path2

    def forward(self, x, return_attention=False):
        if return_attention:
            # Modify the attention mechanism to return attention weights
            x_norm = self.norm1(x)
            attn_output, attn_weights = self.attn(x_norm, return_attention=True)
            x = x + self.drop_path1(self.ls1(attn_output))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
            return x, attn_weights
        else:
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
            return x

class croDINO(nn.Module):
    def __init__(self, repo_name="facebookresearch/dinov2", model_name="dinov2_vitb14", pretrained=True):
        super(croDINO, self).__init__()
        original_model = torch.hub.load(repo_name, model_name)
        modified_blocks = nn.ModuleList([ModifiedNestedTensorBlock(blk) for blk in original_model.blocks])
        
        self.patch_embed = original_model.patch_embed
        self.blocks = modified_blocks
        # self.blocks = original_model.blocks
        self.norm = original_model.norm
        self.head = original_model.head
        
        # Positional Encoding
        num_patches = original_model.patch_embed.proj.weight.shape[0]  # original number of patches
        embed_dim = original_model.patch_embed.proj.out_channels
        self.pos_embed = nn.Parameter(torch.cat([original_model.pos_embed[:, :num_patches], original_model.pos_embed[:, :num_patches]], dim=1))
        
        # Ensure the positional embedding matches the concatenated tokens
        self.pos_embed = nn.Parameter(self.pos_embed[:, :512, :])  # 512 = 256 patches per image * 2 images
        
        # Freeze parameters if pretrained is True
        if pretrained:
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            for param in self.blocks.parameters():
                param.requires_grad = False
            for param in self.norm.parameters():
                param.requires_grad = False
            for param in self.head.parameters():
                param.requires_grad = False

    def forward(self, x1, x2, return_attention=False):
        x1 = self.patch_embed(x1)
        x2 = self.patch_embed(x2)
        
        # print(f"x1 shape: {x1.shape}, x2 shape: {x2.shape}")  # Debugging step

        # Concatenate tokens from both images
        x = torch.cat((x1, x2), dim=1)
        
        # print(f"Concatenated x shape: {x.shape}, Positional embedding shape: {self.pos_embed.shape}")  # Debugging step

        # Add positional encoding
        x = x + self.pos_embed
        
        # Collect attention weights if needed
        attention_weights = []
        
        # Process through transformer blocks
        for blk in self.blocks:
            if return_attention:
                x, attn = blk(x, return_attention=True)
                attention_weights.append(attn)
            else:
                x = blk(x)
        
        x = self.norm(x)
        x = self.head(x)
        
        if return_attention:
            return x, attention_weights
        else:
            return x
        


# Sample paired images
dataset_path = '/home/lrusso/cvusa'
train_filenames, val_filenames = sample_paired_images(dataset_path, sample_percentage=0.2, split_ratio=0.8, groundtype='cutouts')

# Settings
image_size = 224
aerial_scaling = 3
provide_paths = False

transform_ground = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.CenterCrop((image_size, image_size)),
    transforms.ToTensor()
])

transform_aerial = transforms.Compose([
    transforms.Resize((int(image_size * aerial_scaling), int(image_size * aerial_scaling))),
    transforms.CenterCrop((image_size, image_size)),
    transforms.ToTensor()
])

# Instantiate the dataset and dataloader
paired_dataset = PairedImagesDataset(train_filenames, transform_aerial=transform_aerial, transform_ground=transform_ground)
data_loader = DataLoader(paired_dataset, batch_size=1, shuffle=True)

# Define the new model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = croDINO(pretrained=True).to(device)

# Load a single pair of images
ground_image, aerial_image = next(iter(data_loader))
ground_image = ground_image.to(device)
aerial_image = aerial_image.to(device)

# Compte the output of the model
output, attention = model(ground_image, aerial_image, return_attention=True)

print("output shape: ", output.shape)