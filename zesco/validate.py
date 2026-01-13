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

from zesco.dataset import PairedImagesDataset, sample_cvusa_images, get_transforms, denormalize
from zesco.model import CrossviewModel, CosineSimilarityLoss, CosineSimilarityLossCustom, get_processors
from zesco.utils import get_averaged_vertical_tokens, get_averaged_radial_tokens, find_alignment, _next_sample_id, _save_separate_figures
from zesco.skyfilter import SkyFilter
from zesco.depther import DepthAnything

import warnings
from transformers import logging
logging.set_verbosity_error()
warnings.filterwarnings("ignore")


def validate(model, processors, data_loader, config):

    # Retrieve settings from config
    device = config['device']
    grid_size_ground = config['grid_size_ground']
    grid_size_aerial = config['grid_size_aerial']
    output_dir = config['output_dir']
    debug = config['debug']
    save_mode = config['save_mode']
    num_layers = config['num_layers']

    # Create results directory and retrieve batch size
    results_dir = os.path.join(config['main_output_dir'], output_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Store config into into json file
    with open(os.path.join(results_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # Define Loss Function
    if config['loss'] == 'cosine_similarity':
        loss = CosineSimilarityLoss()
    elif config['loss'] == 'cosine_similarity_custom':
        loss = CosineSimilarityLossCustom()
    else:
        raise ValueError('The loss provided is not implemented.')

    # Initialize the Sky Filter and DepthAnything
    sky_filter = SkyFilter(grid_size=grid_size_ground)
    depth_anything = DepthAnything(grid_size=grid_size_ground)

    # Create aerial depth map grid using numpy
    radial_coords_x = np.arange(grid_size_aerial[0]).astype(np.float32)
    radial_coords_y = np.arange(grid_size_aerial[1]).astype(np.float32)
    x_grid, y_grid = np.meshgrid(radial_coords_x, radial_coords_y, indexing='ij')
    center_x = (grid_size_aerial[0] - 1) / 2
    center_y = (grid_size_aerial[1] - 1) / 2
    radial_dist = np.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)
    depth_map_grid_aerial = 1 - (radial_dist / radial_dist.max())

    # Core Processing Loop
    delta_yaws = []
    delta_yaws_dir = []
    with tqdm(total=len(data_loader.dataset), desc="Processing Images") as pbar:
        for batch_idx, (ground_images, aerial_images, fovs, yaws, pitchs) in enumerate(data_loader):
            ground_images = ground_images.to(device)
            aerial_images = aerial_images.to(device)
            batch_size = ground_images.size(0)  # Get actual batch size (might be smaller for last batch)

            if debug:
                print(f"Batch {batch_idx}: fovs", fovs)
                print(f"Batch {batch_idx}: yaws", yaws)
                print(f"Batch {batch_idx}: pitchs", pitchs)

            # Forward pass through the model
            with torch.no_grad():
                ground_tokens, aerial_tokens = model(ground_images, aerial_images, debug=False)
            fov_x, fov_y = fovs

            # Process each image in the batch individually
            for i in range(batch_size):  # Iterate over batch size
                ground_image = ground_images[i:i+1]
                aerial_image = aerial_images[i:i+1]
                fov = (fov_x[i].item(), fov_y[i].item())
                yaw = yaws[i].item()
                heading = yaw - 90
                pitch = pitchs[i].item()
                
                # Extract features for the i-th image in the batch
                ground_features = ground_tokens[i:i+1].squeeze().detach().cpu().numpy()
                aerial_features = aerial_tokens[i:i+1].squeeze().detach().cpu().numpy()
                
                if debug:
                    print("fov", fov)
                    print("yaw", yaw)
                    print("pitch", pitch)
                    print("ground_features.shape:", ground_features.shape)
                    print("aerial_features.shape:", aerial_features.shape)

                # Convert images to numpy for visualization
                if processors is not None:
                    ground_image_denorm = denormalize(ground_image.squeeze(), processors[0])
                    aerial_image_denorm = denormalize(aerial_image.squeeze(), processors[1])
                    ground_image_np = ground_image_denorm.permute(1, 2, 0).detach().cpu().numpy()
                    aerial_image_np = aerial_image_denorm.permute(1, 2, 0).detach().cpu().numpy()
                else:
                    raise ValueError("Processors must be provided for image denormalization.")

                # For the visualization with sky filter, convert to uint8
                ground_image_vis = ground_image_np * 255
                aerial_image_vis = aerial_image_np * 255
                ground_image_vis = ground_image_vis.astype(np.uint8)
                aerial_image_vis = aerial_image_vis.astype(np.uint8)

                # Apply sky filter
                ground_image_no_sky, sky_mask, sky_grid = sky_filter(ground_image_vis, debug=debug)

                # Apply depth estimation
                depth_map_ground, depth_map_grid_ground = depth_anything(ground_image_no_sky, debug=debug)

                fov_x_i = fov_x[i].item()   # horizontal fov in degrees
                angle_step = fov_x_i / grid_size_ground[1]

                # Compute Averaged Tokens using the weight vector, excluding sky tokens
                vertical_averaged_tokens = get_averaged_vertical_tokens(
                    angle_step=angle_step,
                    image_tokens=ground_features,
                    grid_size=grid_size_ground,
                    sky_grid=sky_grid,
                    depth_map_grid=depth_map_grid_ground,
                    num_layers=num_layers,
                    debug=debug
                )
                radial_averaged_tokens = get_averaged_radial_tokens(
                    angle_step=angle_step,
                    image_tokens=aerial_features,
                    grid_size=grid_size_aerial,
                    sky_grid=sky_grid,
                    depth_map_grid=depth_map_grid_aerial,
                    num_layers=num_layers,
                    debug=debug
                )

                if debug:
                    print("averaged vertical tokens: ", vertical_averaged_tokens.shape)
                    print("averaged radial tokens: ", radial_averaged_tokens.shape)

                if debug:
                    return 0

                # Find the best alignment
                best_orientation, distances, min_distance, confidence = find_alignment(loss, vertical_averaged_tokens, radial_averaged_tokens, grid_size_ground, fov_x_i, debug=False)

                delta_yaw = np.abs(((best_orientation - heading) + 180) % 360 - 180)
                delta_yaw_dir = np.abs((delta_yaw + 90) % 180 - 90)
                if delta_yaw < 0:
                    delta_yaw += 180

                delta_yaws.append(delta_yaw)
                delta_yaws_dir.append(delta_yaw_dir)

                if save_mode == 'all':

                    sample_id = _next_sample_id(results_dir)
                    _save_separate_figures(
                        results_dir=results_dir,
                        sample_id=sample_id,
                        ground_image_np=ground_image_np,
                        aerial_image_np=aerial_image_np,
                        best_orientation=best_orientation,
                        yaw=yaw,
                        angle_step=angle_step,
                        distances=distances
                    )

                # Update progress bar with current results
                pbar.set_postfix({
                    'Med180': f"{np.median(delta_yaws):.2f}°",
                    'Mean90': f"{np.mean(delta_yaws_dir):.2f}°"
                })
                pbar.update(1)  # Increment by 1 for each image processed

    # Store Data
    delta_yaws = np.array(delta_yaws)
    delta_yaws_dir = np.array(delta_yaws_dir)
    with open(os.path.join(results_dir, 'data.pkl'), 'wb') as f:
        pickle.dump(
            {
                'delta_yaws': delta_yaws,
                'delta_yaws_dir': delta_yaws_dir
            },
            f
        )

    # Store Metrics
    error_mean = np.mean(delta_yaws)
    error_median = np.median(delta_yaws)
    recall_at_k = np.mean(delta_yaws <= config['recall_k']) * 100.0
    error_mean_dir = np.mean(delta_yaws_dir)
    error_median_dir = np.median(delta_yaws_dir)
    recall_at_k_dir = np.mean(delta_yaws_dir <= config['recall_k']) * 100.0
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(
            {
                "mean": float(error_mean),
                "median": float(error_median),
                f"recall_at_{config['recall_k']}": float(recall_at_k),
                "tau_mean": float(error_mean_dir),
                "tau_median": float(error_median_dir),
                f"tau_recall_at_{config['recall_k']}": float(recall_at_k_dir)
            },
            f,
            indent=4
        )
    print(f"Med@180: {error_median:.2f}°")
    print(f"Mean@90: {error_mean_dir:.2f}°")
    print(f"r@{config['recall_k']}: {recall_at_k:.2f}%")
    print(f'Med@90: {error_median_dir:.2f}°')

    # Store Histograms
    if save_mode in ['hist', 'all']:
        plt.figure(figsize=(10, 6))
        plt.hist(delta_yaws, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Delta Yaw (degrees)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Orientation Errors - {config["dataset"]}\n' +
                f'Med@180: {error_median:.2f}°, Mean@90: {error_mean_dir:.2f}°, r@{config['recall_k']}: {recall_at_k:.2f}%',
                fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'orientation_errors.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Plot histogram for directional errors
        plt.figure(figsize=(10, 6))
        plt.hist(delta_yaws_dir, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Orientation Directional Error (degrees)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Directional Errors - {config["dataset"]}\n' +
                f'Med@180: {error_median:.2f}°, Mean@90: {error_mean_dir:.2f}°, r@{config['recall_k']}: {recall_at_k:.2f}%',
                fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'orientation_directional_errors.png'), dpi=300, bbox_inches='tight')
        plt.close()
    

    return delta_yaws, delta_yaws_dir