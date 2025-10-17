import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
import argparse
from tqdm import tqdm

from zesco.dataset import PairedImagesDataset, sample_cvusa_images, sample_cities_images, get_transforms, denormalize
from zesco.model import CrossviewModel, CosineSimilarityLoss, CosineSimilarityLossCustom, get_processors
from zesco.utils import get_averaged_vertical_tokens, get_averaged_radial_tokens, find_alignment, _next_sample_id, _save_separate_fizesco.gures

from zesco.skyfilter import SkyFilter
from zesco.depther import DepthAnything
from apply_method import test

import warnings
from transformers import logging
logging.set_verbosity_error()
warnings.filterwarnings("ignore")


DATASETS = ['AFR', 'AFU', 'ASR', 'ASU', 'EUR', 'EUU', 'NAR', 'NAU', 'SAR', 'SAU']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test ZeSCO on individual datasets')
    parser.add_argument('--backbone', '-b', type=str, default='dinov3', help='Model to use')
    parser.add_argument('--loss', '-l', type=str, default='cosine_similarity', help='Loss to use for the Orientation Estimation')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--save_mode', '-m', type=str, default='separate', choices=['combined', 'separate', 'both'], help='Save only the combined 2x2 figure, only the 4 separate figures, or both')
    args = parser.parse_args()
    
    # Settings
    image_size = 224
    aerial_scaling = 2
    provide_paths = False
    BATCH_SIZE = 8

    # Get the processor and transforms
    processors = get_processors(args.backbone)
    transform_ground, transform_aerial = get_transforms(processors, image_size, aerial_scaling)

    # Define the Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Define Loss Function
    if args.loss == 'cosine_similarity':
        loss = CosineSimilarityLoss()
    elif args.loss == 'cosine_similarity_custom':
        loss = CosineSimilarityLossCustom()
    else:
        raise ValueError('The loss provided is not implemented.')

    # Load the Model
    model = CrossviewModel(backbone=args.backbone, frozen=True).to(device)
    grid_size = (image_size // model.patch_size, image_size // model.patch_size)
    print(f"Model patch size: {model.patch_size}, grid size: {grid_size}")
    model.show()

    # Loop through each dataset
    for dataset_name in DATASETS:

        # Get Dataset Images
        dataset_path = os.path.join(r'D:\cross_view_localization_DSM\Data', dataset_name)
        train_filenames, _ = sample_cvusa_images(dataset_path, sample_percentage=0.2, split_ratio=1, groundtype='panos')

        # Initialize the dataset and dataloader
        paired_dataset = PairedImagesDataset(train_filenames, transform_aerial=transform_aerial, transform_ground=transform_ground, cutout_from_pano=True)
        data_loader = DataLoader(paired_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # Test the model
        test(
            model=model,
            processors=processors,
            loss=loss,
            data_loader=data_loader,
            grid_size=grid_size,
            device=device,
            savepath=dataset_name,
            debug=args.debug,
            save_mode=args.save_mode
        )


