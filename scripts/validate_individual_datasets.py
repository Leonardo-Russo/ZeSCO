import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import argparse
import random
import pickle
import matplotlib.pyplot as plt
import time

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
    parser.add_argument('--crop_percentage', type=float, default=0.25, help='Percentage of the image to crop')
    parser.add_argument('--loss', type=str, default='cosine_similarity', help='Loss to use for the Orientation Estimation')
    parser.add_argument('--sample_percentage', type=float, default=0.2, help='Percentage of dataset to sample for testing')
    parser.add_argument('--threshold', type=float, default=0.4, help='Needed for the middleground weights')
    parser.add_argument('--main_output_dir', type=str, default=r'..\results\individual_untitled', help='Directory to save output files')
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
        'crop_percentage': args.crop_percentage,
        'sample_percentage': args.sample_percentage,
        'save_mode': args.save_mode,
        'debug': args.debug,
        'threshold': args.threshold,
        'image_size': image_size,
        'aerial_scaling': aerial_scaling,
        'batch_size': BATCH_SIZE,
        'device': device,
        'grid_size': grid_size,
        'seed': 42,
        'main_output_dir': os.path.join(r'..\results', args.main_output_dir)
    }

    # Get the processor and transforms
    processors = get_processors(args.backbone)
    transform_ground, transform_aerial = get_transforms(processors, image_size, aerial_scaling, crop_percentage=args.crop_percentage)

    # Loop through each dataset
    global_delta_yaws = []
    results = {}
    start_time = time.time()
    for dataset_name in DATASETS:

        print(f"\n=== Validating dataset: {dataset_name} ===")

        # Set seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Get Dataset Images
        dataset_path = os.path.join(r'D:\cross_view_localization_DSM\Data', dataset_name)
        train_filenames, _ = sample_cvusa_images(dataset_path, sample_percentage=config['sample_percentage'], split_ratio=1, groundtype='panos')

        # Create generator for reproducible DataLoader
        generator = torch.Generator()
        generator.manual_seed(seed)

        # Initialize the dataset and dataloader
        paired_dataset = PairedImagesDataset(train_filenames, transform_aerial=transform_aerial, transform_ground=transform_ground, cutout_from_pano=True)
        data_loader = DataLoader(
            paired_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False,
            generator=generator,    # use fixed generator
            worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)  # For multi-worker reproducibility
        )

        # Update config
        config['output_dir'] = dataset_name
        config['dataset'] = dataset_name

        # Validate the model
        delta_yaws = validate(
            model=model,
            processors=processors,
            data_loader=data_loader,
            config=config
        )
        global_delta_yaws.extend(delta_yaws)

        # Save median result for each dataset
        results[dataset_name] = np.median(delta_yaws)

    # Stop timer
    total_time = time.time() - start_time
    print(f"\n=== Completed validation on all datasets in {total_time/60:.2f} minutes ===")

    # Print median results
    print("\nMedian Delta Yaw Errors by Dataset:")
    for dataset_name in DATASETS:
        print(f"{dataset_name}: {results[dataset_name]:.2f}°")

    # Plot combined histogram
    global_delta_yaws = np.array(global_delta_yaws)
    error_mean = np.mean(global_delta_yaws)
    error_std = np.std(global_delta_yaws)
    error_median = np.median(global_delta_yaws)

    # Show an histogram of the delta_yaw errors
    plt.figure(figsize=(10, 6))
    plt.hist(global_delta_yaws, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Absolute Orientation Error (degrees)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Orientation Error Distribution - CVGlobal\n' +
            f'Mean: {error_mean:.2f}°, Median: {error_median:.2f}°, Std: {error_std:.2f}°',
            fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config['main_output_dir'], 'delta_yaws_hist.png'), dpi=300, bbox_inches='tight')
    
    # Save delta yaws to pickle file
    with open(os.path.join(config['main_output_dir'], 'delta_yaws.pkl'), 'wb') as f:
        pickle.dump(global_delta_yaws, f)

    # Save statistics to well-formatted info.txt file
    with open(os.path.join(config['main_output_dir'], 'info.txt'), 'w') as f:
        f.write("Delta Yaw Error Statistics\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Total Samples: {len(global_delta_yaws)}\n\n")
        f.write("Error Metrics:\n")
        f.write("-" * 15 + "\n")
        f.write(f"Mean Delta Yaw Error:       {np.mean(global_delta_yaws):.4f}°\n")
        f.write(f"Standard Deviation:         {np.std(global_delta_yaws):.4f}°\n")
        f.write(f"Median Delta Yaw Error:     {np.median(global_delta_yaws):.4f}°\n")
        f.write(f"Minimum Delta Yaw Error:    {np.min(global_delta_yaws):.4f}°\n")
        f.write(f"Maximum Delta Yaw Error:    {np.max(global_delta_yaws):.4f}°\n")
        f.write("\nMedian Delta Yaw Error by Dataset:\n")
        f.write("-" * 35 + "\n")
        for dataset_name in DATASETS:
            f.write(f"{dataset_name}: {results[dataset_name]:.4f}°\n")
        f.write("\n")
        f.write("Timing Information:\n")
        f.write("-" * 15 + "\n")
        f.write(f"Total elapsed time:         {total_time/60:.2f} minutes\n")
