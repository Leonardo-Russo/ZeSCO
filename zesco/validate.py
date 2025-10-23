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

from zesco.dataset import PairedImagesDataset, sample_cvusa_images, sample_cities_images, get_transforms, denormalize
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
    grid_size = config['grid_size']
    output_dir = config['output_dir']
    debug = config['debug']
    save_mode = config['save_mode']
    threshold = config['threshold']

    # Create results directory and retrieve batch size
    results_dir = os.path.join(r'..\results', output_dir)
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
    sky_filter = SkyFilter(grid_size=grid_size)
    depth_anything = DepthAnything(grid_size=grid_size)

    # Core Processing Loop
    delta_yaws = []
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
            # Note: batch_size is already defined above
            for i in range(batch_size):  # Iterate over batch size
                ground_image = ground_images[i:i+1]
                aerial_image = aerial_images[i:i+1]
                fov = (fov_x[i].item(), fov_y[i].item())
                yaw = yaws[i].item()
                pitch = pitchs[i].item()
                
                # Extract features for the i-th image in the batch
                ground_features = ground_tokens[i:i+1].squeeze().detach().cpu().numpy()
                aerial_features = aerial_tokens[i:i+1].squeeze().detach().cpu().numpy()

                # Calculate grid size from actual token dimensions
                grid_dim = int(np.sqrt(ground_features.shape[0]))  # assuming square grid
                
                if debug:
                    print("fov", fov)
                    print("yaw", yaw)
                    print("pitch", pitch)
                    print("normalized_features1.shape:", ground_features.shape)
                    print("normalized_features2.shape:", aerial_features.shape)
                    print("grid_size:", grid_dim)

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
                depth_map, depth_map_grid = depth_anything(ground_image_no_sky, debug=debug)

                fov_x_i = fov_x[i].item()                          # horizontal fov in degrees
                angle_step = fov_x_i / grid_dim

                # Compute Averaged Tokens using the weight vector, excluding sky tokens
                fore_vert_avg_tokens, midd_vert_avg_tokens, back_vert_avg_tokens = get_averaged_vertical_tokens(angle_step, ground_features, grid_dim, sky_grid, depth_map_grid, threshold=threshold)
                fore_rad_avg_tokens, midd_rad_avg_tokens, back_rad_avg_tokens = get_averaged_radial_tokens(angle_step, aerial_features, grid_dim, sky_grid, depth_map_grid)
                
                if debug:
                    print("averaged vertical tokens: ", fore_vert_avg_tokens.shape)
                    print("averaged radial tokens: ", fore_rad_avg_tokens.shape)   

                # Find the best alignment
                best_orientation, distances, min_distance, confidence = find_alignment(loss, fore_vert_avg_tokens, midd_vert_avg_tokens, back_vert_avg_tokens, fore_rad_avg_tokens, midd_rad_avg_tokens, back_rad_avg_tokens, grid_dim, fov_x_i, debug=False)

                delta_yaw = np.abs(((90 - (yaw - 180)) - best_orientation + 180) % 360 - 180)
                if delta_yaw < 0:
                    delta_yaw += 180
                delta_yaws.append(delta_yaw)

                if save_mode == 'all':

                    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))

                    # ax1.imshow(ground_image_np)
                    # ax1.set_title("Ground Image - Yaw: {:.1f}°".format(yaw))
                    # ax1.axis('off')

                    # ax2.imshow(aerial_image_np)
                    # radius = aerial_image_np.shape[0] // 2
                    # center = (aerial_image_np.shape[1] // 2, aerial_image_np.shape[0] // 2)
                    # end_x = int(center[0] + radius * np.cos(np.deg2rad(best_orientation)))
                    # end_y = int(center[1] - radius * np.sin(np.deg2rad(best_orientation)))
                    # end_x_GT = int(center[0] + radius * np.cos(np.deg2rad(90 - (yaw - 180))))
                    # end_y_GT = int(center[1] - radius * np.sin(np.deg2rad(90 - (yaw - 180))))
                    # line_pred = ax2.plot([center[0], end_x], [center[1], end_y], color='red', linestyle='--', label='Prediction')
                    # line_gt = ax2.plot([center[0], end_x_GT], [center[1], end_y_GT], color='orange', linestyle='--', label='Ground Truth')

                    # ax2.set_title("Aerial Image Orientation - Delta: {:.4f}°".format(delta_yaw))
                    # ax2.legend(loc='upper right')
                    # ax2.axis('off')

                    # ax3.plot(np.arange(0, 360, angle_step), distances)
                    # ax3.set_title("Distance vs Orientation")
                    # ax3.grid(True)
                    # ax3.set_xlabel('Orientation')
                    # ax3.set_ylabel('Distance')
                    # ax3.set_xlim(0, 360)
                    # ax3.set_ylim(min(distances), max(distances))

                    # ax4.imshow(aerial_image_np)
                    # radius = aerial_image_np.shape[0] // 2
                    # center = (aerial_image_np.shape[1] // 2, aerial_image_np.shape[0] // 2)
                    # min_dist = min(distances)
                    # max_dist = max(distances)
                    # for j, beta in enumerate(np.arange(0, 360, angle_step)):
                    #     end_x = int(center[0] + radius * np.cos(np.deg2rad(beta)))
                    #     end_y = int(center[1] - radius * np.sin(np.deg2rad(beta)))
                    #     # Normalize distances for color map and ensure they're in [0, 1]
                    #     normalized_dist = (distances[j] - min_dist) / (max_dist - min_dist) if max_dist > min_dist else 0.0
                    #     normalized_dist = np.clip(normalized_dist, 0.0, 1.0)
                    #     color = plt.cm.plasma(normalized_dist)
                    #     ax4.plot([center[0], end_x], [center[1], end_y], color=color)
                    # ax4.set_title("Aerial Image with Distances")
                    # ax4.axis('off')

                    # norm = plt.Normalize(min_dist, max_dist)
                    # sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
                    # sm.set_array([])
                    # cbar = plt.colorbar(sm, ax=ax4)

                    # Determine the next available sample id (per group of images)
                    sample_id = _next_sample_id(results_dir)

                    if save_mode == 'all':
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
                    'Delta Yaw Median': f"{np.median(delta_yaws):.2f}°" 
                    # 'Batch': f"{batch_idx+1}/{len(data_loader)}"
                })
                pbar.update(1)  # Increment by 1 for each image processed

    # Output the delta_yaw errors
    delta_yaws = np.array(delta_yaws)
    error_mean = np.mean(delta_yaws)
    error_std = np.std(delta_yaws)
    error_median = np.median(delta_yaws)
    
    print(f"Mean Delta Yaw Error: {error_mean}")
    print(f"Standard Deviation of Delta Yaw Error: {error_std}")
    print(f"Median Delta Yaw Error: {error_median}")

    # Show an histogram of the delta_yaw errors
    plt.figure(figsize=(10, 6))
    plt.hist(delta_yaws, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Absolute Orientation Error (degrees)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Orientation Error Distribution - {config['dataset']}\n' +
             f'Mean: {error_mean:.2f}°, Median: {error_median:.2f}°, Std: {error_std:.2f}°',
             fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'delta_yaws_hist.png'), dpi=300, bbox_inches='tight')
    
    # Save delta yaws to pickle file
    with open(os.path.join(results_dir, 'delta_yaws.pkl'), 'wb') as f:
        pickle.dump(delta_yaws, f)
    
    # Save statistics to well-formatted info.txt file
    with open(os.path.join(results_dir, 'info.txt'), 'w') as f:
        f.write("Delta Yaw Error Statistics\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Total Samples: {len(delta_yaws)}\n\n")
        f.write("Error Metrics:\n")
        f.write("-" * 15 + "\n")
        f.write(f"Mean Delta Yaw Error:       {np.mean(delta_yaws):.4f}°\n")
        f.write(f"Standard Deviation:         {np.std(delta_yaws):.4f}°\n")
        f.write(f"Median Delta Yaw Error:     {np.median(delta_yaws):.4f}°\n")
        f.write(f"Minimum Delta Yaw Error:    {np.min(delta_yaws):.4f}°\n")
        f.write(f"Maximum Delta Yaw Error:    {np.max(delta_yaws):.4f}°\n")