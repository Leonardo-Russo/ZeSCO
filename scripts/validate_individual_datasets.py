import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import argparse
import random

from zesco.dataset import PairedImagesDataset, sample_cvusa_images, sample_cities_images, get_transforms
from zesco.model import CrossviewModel, get_processors

from zesco.validate import validate

import warnings
from transformers import logging
logging.set_verbosity_error()
warnings.filterwarnings("ignore")


DATASETS = ['AFR', 'AFU', 'ASR', 'ASU', 'EUR', 'EUU', 'NAR', 'NAU', 'SAR', 'SAU']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the ZeSCO model on individual datasets.')
    parser.add_argument('--backbone', type=str, default='dinov3', help='Model to use')
    parser.add_argument('--loss', type=str, default='cosine_similarity', help='Loss to use for the Orientation Estimation')
    parser.add_argument('--sample_percentage', type=float, default=0.2, help='Percentage of dataset to sample for testing')
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
        'output_dir': None,
        'backbone': args.backbone,
        'loss': args.loss,
        'dataset': None,
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

    # Get the processor and transforms
    processors = get_processors(args.backbone)
    transform_ground, transform_aerial = get_transforms(processors, image_size, aerial_scaling)

    # Loop through each dataset
    for dataset_name in DATASETS:

        # Get Dataset Images
        dataset_path = os.path.join(r'D:\cross_view_localization_DSM\Data', dataset_name)
        train_filenames, _ = sample_cvusa_images(dataset_path, sample_percentage=config['sample_percentage'], split_ratio=1, groundtype='panos')

        # Initialize the dataset and dataloader
        paired_dataset = PairedImagesDataset(train_filenames, transform_aerial=transform_aerial, transform_ground=transform_ground, cutout_from_pano=True)
        data_loader = DataLoader(paired_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # Update config
        config['output_dir'] = dataset_name
        config['dataset'] = dataset_name

        # Validate the model
        validate(
        model=model,
        processors=processors,
        data_loader=data_loader,
        config=config
    )

