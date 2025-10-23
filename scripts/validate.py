import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm
import pickle
import random
import json

from zesco.dataset import PairedImagesDataset, sample_cvusa_images, sample_cities_images, get_transforms
from zesco.model import CrossviewModel, get_processors
from zesco.validate import validate

import warnings
from transformers import logging
logging.set_verbosity_error()
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate the ZeSCO model.')
    parser.add_argument('--backbone', type=str, default='dinov3', help='Model to use')
    parser.add_argument('--loss', type=str, default='cosine_similarity', help='Loss to use for the Orientation Estimation')
    parser.add_argument('--dataset', type=str, default='cvglobal', help='Dataset to use')
    parser.add_argument('--sample_percentage', type=float, default=0.02, help='Percentage of dataset to sample for testing')
    parser.add_argument('--output_dir', type=str, default='untitled', help='Path to save the model and results')
    parser.add_argument('--threshold', type=float, default=0.4, help='Needed for the middleground weights')
    parser.add_argument('--save_mode', type=str, default='hist', choices=['all', 'hist'], help='Save only the combined 2x2 figure, only the 4 separate figures, or both')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()

    # Settings
    image_size = 224
    aerial_scaling = 2
    BATCH_SIZE = 8
    seed = 42

    # Set seed
    random.seed(seed)
    np.random.seed(seed)

    # Define the Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the Model
    model = CrossviewModel(backbone=args.backbone, frozen=True).to(device)
    grid_size = (image_size // model.patch_size, image_size // model.patch_size)
    print(f"Model patch size: {model.patch_size}, grid size: {grid_size}")
    model.show()

    # Create config dictionary
    config = {
        'output_dir': args.output_dir,
        'backbone': args.backbone,
        'loss': args.loss,
        'dataset': args.dataset,
        'sample_percentage': args.sample_percentage,
        'save_mode': args.save_mode,
        'debug': args.debug,
        'threshold': args.threshold,
        'image_size': image_size,
        'aerial_scaling': aerial_scaling,
        'batch_size': BATCH_SIZE,
        'device': device,
        'grid_size': grid_size,
        'seed': 42
    }

    # Get Dataset Images
    if config['dataset'].lower() == "cvglobal":
        dataset_path = r'D:\cross_view_localization_DSM\Data\CVGlobal'
        train_filenames, _ = sample_cvusa_images(dataset_path, sample_percentage=0.02, split_ratio=1, groundtype='panos')

    # Get the processor and transforms
    processors = get_processors(config['backbone'])
    transform_ground, transform_aerial = get_transforms(processors, image_size, aerial_scaling)

    # Instantiate the dataset and dataloader
    paired_dataset = PairedImagesDataset(train_filenames, transform_aerial=transform_aerial, transform_ground=transform_ground, cutout_from_pano=True)
    data_loader = DataLoader(paired_dataset, batch_size=BATCH_SIZE, shuffle=True)

    validate(
        model=model,
        processors=processors,
        data_loader=data_loader,
        config=config
    )
