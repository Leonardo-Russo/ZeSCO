import json
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import argparse
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time

from zesco.dataset import PairedImagesDataset, sample_cvusa_images, get_transforms
from zesco.model import CrossviewModel, get_processors
from zesco.validate import validate

import warnings
from transformers import logging
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune hyperparameters for ZeSCO model.')
    parser.add_argument('--dataset', type=str, default='CVGlobal', help='Dataset to use')
    parser.add_argument('--backbone', type=str, default='dinov3', help='Model to use')
    
    # Hyperparameter ranges
    parser.add_argument('--num_layers_values', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6], help='List of num_layers to try')
    parser.add_argument('--crop_percentage_values', type=float, nargs='+', default=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45], help='List of crop_percentages to try')
    
    parser.add_argument('--fov', type=int, default=90, help='Horizontal Field of View for ground images')
    parser.add_argument('--image_size', type=int, default=448, help='Reference square image dimension')
    parser.add_argument('--loss', type=str, default='cosine_similarity', help='Loss to use for the Orientation Estimation')
    parser.add_argument('--sample_percentage', type=float, default=0.1, help='Percentage of dataset to sample for testing')
    parser.add_argument('--recall_k', type=int, default=5, help='K value for Recall@K calculation')
    parser.add_argument('--main_output_dir', type=str, default=r'..\results\finetune_results', help='Directory to save output files')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()

    # Settings
    image_size = args.image_size
    aerial_scaling = 2
    BATCH_SIZE = 8
    seed = args.seed
    horizontal_scaling = round(args.fov / 90.0) if args.fov >= 90 else 1

    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Define the Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the Model (Backbone doesn't change in this loop)
    model = CrossviewModel(backbone=args.backbone, frozen=True).to(device)
    grid_size_ground = (image_size // model.patch_size, image_size // model.patch_size * horizontal_scaling)
    grid_size_aerial = (image_size // model.patch_size, image_size // model.patch_size)
    print(f"Model patch size: {model.patch_size}, grid size ground: {grid_size_ground}, grid size aerial: {grid_size_aerial}")
    model.show()
    
    # Output directory
    os.makedirs(args.main_output_dir, exist_ok=True)

    # Get Processors
    processors = get_processors(args.backbone)

    # Get Dataset Images ONCE to ensure consistency
    print(f"Sampling images from {args.dataset}...")
    dataset_path = os.path.join(r'D:\cross_view_localization_DSM\Data', args.dataset)
    train_filenames, _ = sample_cvusa_images(dataset_path, sample_percentage=args.sample_percentage, split_ratio=1, groundtype='panos')
    print(f"Sampled {len(train_filenames)} images.")

    results = []

    # Loops
    total_combinations = len(args.num_layers_values) * len(args.crop_percentage_values)
    current_iter = 0

    print(f"Starting finetuning with {total_combinations} combinations...")
    print(f"Num Layers: {args.num_layers_values}")
    print(f"Crop Percentages: {args.crop_percentage_values}")

    for nl in args.num_layers_values:
        for cp in args.crop_percentage_values:
            current_iter += 1
            print(f"\n[{current_iter}/{total_combinations}] Testing num_layers={nl}, crop_percentage={cp}")

            # Config for this run
            config = {
                'output_dir': args.dataset, # Just for logging inside validate
                'backbone': args.backbone,
                'num_layers': nl,
                'fov': args.fov,
                'horizontal_scaling': horizontal_scaling,
                'image_size': image_size,
                'loss': args.loss,
                'dataset': args.dataset,
                'crop_percentage': cp,
                'sample_percentage': args.sample_percentage,
                'recall_k': args.recall_k,
                'save_mode': 'none', # Don't save individual plots
                'debug': args.debug,
                'aerial_scaling': aerial_scaling,
                'batch_size': BATCH_SIZE,
                'device': device,
                'grid_size_ground': grid_size_ground,
                'grid_size_aerial': grid_size_aerial,
                'seed': args.seed,
                'main_output_dir': args.main_output_dir
            }
            
            # Transforms depend on crop_percentage
            transform_ground, transform_aerial = get_transforms(processors, image_size, aerial_scaling, crop_percentage=cp, horizontal_scaling=horizontal_scaling)

            # Dataset and DataLoader
            generator = torch.Generator()
            generator.manual_seed(seed)
            
            paired_dataset = PairedImagesDataset(train_filenames, transform_aerial=transform_aerial, transform_ground=transform_ground, cutout_from_pano=True, fov_x=config['fov'])
            data_loader = DataLoader(
                paired_dataset, 
                batch_size=BATCH_SIZE, 
                shuffle=False,
                generator=generator,
                worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)
            )

            # Run Validation
            delta_yaws, delta_yaws_dir = validate(
                model=model,
                processors=processors,
                data_loader=data_loader,
                config=config
            )

            # Calculate Metrics
            delta_yaws = np.array(delta_yaws)
            delta_yaws_dir = np.array(delta_yaws_dir)
            
            error_median_180 = np.median(delta_yaws)
            error_mean_90 = np.mean(delta_yaws_dir)
            recall_val = np.mean(delta_yaws <= args.recall_k) * 100.0

            print(f"  -> Med180: {error_median_180:.2f}, Mean90: {error_mean_90:.2f}, r@{args.recall_k}: {recall_val:.2f}")

            results.append({
                'num_layers': nl,
                'crop_percentage': cp,
                'Median180': error_median_180,
                'Mean90': error_mean_90,
                'RecallAtK': recall_val
            })

    # Save results to file
    results_file = os.path.join(args.main_output_dir, 'finetune_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Plotting
    print("Generating plots...")
    
    # Prepare data for plotting
    X = np.array([r['num_layers'] for r in results])
    Y = np.array([r['crop_percentage'] for r in results])
    
    # Unique sorted values
    nl_vals = sorted(list(set(X)))
    cp_vals = sorted(list(set(Y)))
    X_grid, Y_grid = np.meshgrid(nl_vals, cp_vals)
    
    # Z matrices
    Z_med180 = np.zeros_like(X_grid, dtype=float)
    Z_mean90 = np.zeros_like(X_grid, dtype=float)
    Z_recall = np.zeros_like(X_grid, dtype=float)
    
    # Fill Z matrices
    for r in results:
        i = cp_vals.index(r['crop_percentage'])
        j = nl_vals.index(r['num_layers'])
        Z_med180[i, j] = r['Median180']
        Z_mean90[i, j] = r['Mean90']
        Z_recall[i, j] = r['RecallAtK']

    metrics = [
        ('Median180', Z_med180, 'Median Error 180° (Lower is better)'),
        ('Mean90', Z_mean90, 'Mean Error 90° (Lower is better)'),
        (f'r@{args.recall_k}', Z_recall, f'Recall @ {args.recall_k}° (Higher is better)')
    ]

    for name, Z, title in metrics:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(X_grid, Y_grid, Z, cmap=cm.viridis, linewidth=0, antialiased=False)
        
        ax.set_xlabel('Num Layers')
        ax.set_ylabel('Crop Percentage')
        ax.set_zlabel(name)
        ax.set_title(title)
        
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        plt.savefig(os.path.join(args.main_output_dir, f'surface_plot_{name}.png'), dpi=300)
        
        # Save as pickle for interactive viewing
        with open(os.path.join(args.main_output_dir, f'surface_plot_{name}.pkl'), 'wb') as f:
            pickle.dump(fig, f)

        plt.close()

    print(f"Done. Results saved to {args.main_output_dir}")
