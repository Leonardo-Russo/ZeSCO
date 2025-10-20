import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm
import pickle

from zesco.dataset import PairedImagesDataset, sample_cvusa_images, get_transforms, denormalize
from zesco.model import CosineSimilarityLoss, CosineSimilarityLossCustom, get_processors
from zesco.utils import get_averaged_vertical_tokens, get_averaged_radial_tokens, find_alignment, _next_sample_id, _save_separate_figures
from zesco.skyfilter import SkyFilter
from zesco.depther import DepthAnything

# import warnings
# from transformers import logging
# logging.set_verbosity_error()
# warnings.filterwarnings("ignore")

# matplotlib.use('TkAgg')  # or 'Agg' for non-GUI


def test(processors, loss, data_loader, grid_size, device, savepath='untitled', threshold=0.4, debug=False, save_mode='combined'):

    # Create results directory and retrieve batch size
    results_dir = os.path.join(r'..\results', savepath)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Initialize the Sky Filter and DepthAnything
    sky_filter = SkyFilter(grid_size=grid_size)
    depth_anything = DepthAnything(grid_size=grid_size).to(device)

    # Core Processing Loop
    delta_yaws = []
    with tqdm(total=len(data_loader.dataset), desc="Processing Images") as pbar:
        for batch_idx, (ground_images, aerial_images, fovs, yaws, pitchs) in enumerate(data_loader):
            ground_images = ground_images.to(device)
            aerial_images = aerial_images.to(device)
            fov_x, fov_y = fovs

            # Denormalize images
            ground_images_denorm = denormalize(ground_images, processors[0])
            aerial_images_denorm = denormalize(aerial_images, processors[1])

            # Apply sky filter
            # NOTE: it's working but I need to check that it is working correctly
            with torch.no_grad():
                ground_images_no_sky, sky_masks, sky_grids = sky_filter(ground_images_denorm.permute(0, 2, 3, 1), debug=debug)

            if debug:
                # Visualize the original image, mask, and the sky-removed image
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(18, 6))
                ax1.imshow(ground_images_denorm[0].permute(1, 2, 0).cpu().numpy())
                ax1.set_title("Ground Image", fontsize=12, fontweight='bold')
                ax1.axis('off')
                ax2.imshow(sky_masks[0, 0].cpu().numpy(), cmap='gray')
                ax2.set_title("Sky Mask", fontsize=12, fontweight='bold')
                ax2.axis('off')
                ax3.imshow(ground_images_no_sky[0].permute(1, 2, 0).cpu().numpy())
                ax3.set_title("Ground Image without Sky", fontsize=12, fontweight='bold')
                ax3.axis('off')
                ax4.imshow(sky_grids[0, 0].cpu().numpy(), cmap='gray')
                ax4.set_title("Sky Grid Mask", fontsize=12, fontweight='bold')
                ax4.axis('off')
                plt.show()

            # Apply depth estimation
            with torch.no_grad():
                depth_maps, depth_maps_grid = depth_anything(ground_images_no_sky.permute(0, 2, 3, 1), debug=debug)    # (B, 1, H, W), (B, 1, grid_h, grid_w)
                # depth_maps, depth_maps_grid = depth_anything(ground_images_denorm.permute(0, 2, 3, 1), debug=debug)    # (B, 1, H, W), (B, 1, grid_h, grid_w)

            # Plot one of the depth maps and one of the depth maps grid for debugging
            if debug:
                fig, ax = plt.subplots(1, 3, figsize=(10, 5))
                ax[0].imshow(ground_images_no_sky[0].permute(1, 2, 0).cpu().numpy())
                ax[0].set_title('Ground Image without Sky')
                ax[0].axis('off')
                ax[1].imshow(depth_maps[0, 0].cpu().numpy(), cmap='plasma')
                ax[1].set_title('Depth Map')
                ax[1].axis('off')
                ax[2].imshow(depth_maps_grid[0, 0].cpu().numpy(), cmap='plasma')
                ax[2].set_title('Depth Map Grid')
                ax[2].axis('off')
                plt.show()


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


                fov_x_i = fov_x[i].item()                          # horizontal fov in degrees
                angle_step = fov_x_i / grid_dim

                # Compute Averaged Tokens using the weight vector, excluding sky tokens
                fore_vert_avg_tokens, midd_vert_avg_tokens, back_vert_avg_tokens = get_averaged_vertical_tokens(angle_step, ground_features, grid_dim, sky_grids, depth_map_grid, threshold=threshold)
                fore_rad_avg_tokens, midd_rad_avg_tokens, back_rad_avg_tokens = get_averaged_radial_tokens(angle_step, aerial_features, grid_dim, sky_grids, depth_map_grid)
                
                if debug:
                    print("averaged vertical tokens: ", fore_vert_avg_tokens.shape)
                    print("averaged radial tokens: ", fore_rad_avg_tokens.shape)   

                # Find the best alignment
                best_orientation, distances, min_distance, confidence = find_alignment(loss, fore_vert_avg_tokens, midd_vert_avg_tokens, back_vert_avg_tokens, fore_rad_avg_tokens, midd_rad_avg_tokens, back_rad_avg_tokens, grid_dim, fov_x_i, debug=False)

                delta_yaw = np.abs(((90 - (yaw - 180)) - best_orientation + 180) % 360 - 180)
                if delta_yaw < 0:
                    delta_yaw += 180
                delta_yaws.append(delta_yaw)

                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))

                ax1.imshow(ground_image_np)
                ax1.set_title("Ground Image - Yaw: {:.1f}°".format(yaw))
                ax1.axis('off')

                ax2.imshow(aerial_image_np)
                radius = aerial_image_np.shape[0] // 2
                center = (aerial_image_np.shape[1] // 2, aerial_image_np.shape[0] // 2)
                end_x = int(center[0] + radius * np.cos(np.deg2rad(best_orientation)))
                end_y = int(center[1] - radius * np.sin(np.deg2rad(best_orientation)))
                end_x_GT = int(center[0] + radius * np.cos(np.deg2rad(90 - (yaw - 180))))
                end_y_GT = int(center[1] - radius * np.sin(np.deg2rad(90 - (yaw - 180))))
                line_pred = ax2.plot([center[0], end_x], [center[1], end_y], color='red', linestyle='--', label='Prediction')
                line_gt = ax2.plot([center[0], end_x_GT], [center[1], end_y_GT], color='orange', linestyle='--', label='Ground Truth')

                ax2.set_title("Aerial Image Orientation - Delta: {:.4f}°".format(delta_yaw))
                ax2.legend(loc='upper right')
                ax2.axis('off')

                ax3.plot(np.arange(0, 360, angle_step), distances)
                ax3.set_title("Distance vs Orientation")
                ax3.grid(True)
                ax3.set_xlabel('Orientation')
                ax3.set_ylabel('Distance')
                ax3.set_xlim(0, 360)
                ax3.set_ylim(min(distances), max(distances))

                ax4.imshow(aerial_image_np)
                radius = aerial_image_np.shape[0] // 2
                center = (aerial_image_np.shape[1] // 2, aerial_image_np.shape[0] // 2)
                min_dist = min(distances)
                max_dist = max(distances)
                for j, beta in enumerate(np.arange(0, 360, angle_step)):
                    end_x = int(center[0] + radius * np.cos(np.deg2rad(beta)))
                    end_y = int(center[1] - radius * np.sin(np.deg2rad(beta)))
                    # Normalize distances for color map and ensure they're in [0, 1]
                    normalized_dist = (distances[j] - min_dist) / (max_dist - min_dist) if max_dist > min_dist else 0.0
                    normalized_dist = np.clip(normalized_dist, 0.0, 1.0)
                    color = plt.cm.plasma(normalized_dist)
                    ax4.plot([center[0], end_x], [center[1], end_y], color=color)
                ax4.set_title("Aerial Image with Distances")
                ax4.axis('off')

                norm = plt.Normalize(min_dist, max_dist)
                sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax4)

                # Determine the next available sample id (per group of images)
                sample_id = _next_sample_id(results_dir)

                # Save combined and/or separate figures depending on save_mode
                if save_mode in ("combined", "both"):
                    combined_path = os.path.join(results_dir, f"sample_{sample_id}_combined.png")
                    plt.savefig(combined_path, dpi=300, bbox_inches='tight')

                if save_mode in ("separate", "both"):
                    _save_separate_figures(results_dir, sample_id,
                                            ground_image_np, aerial_image_np,
                                            best_orientation, yaw,
                                            angle_step, distances)
                if debug:
                    plt.show()

                plt.close(fig)

                # Update progress bar with current results
                pbar.set_postfix({
                    'Delta Yaw': f"{np.mean(delta_yaws):.2f}°", 
                    'Batch': f"{batch_idx+1}/{len(data_loader)}"
                })
                pbar.update(1)  # Increment by 1 for each image processed

    # Output the delta_yaw errors
    delta_yaws = np.array(delta_yaws)
    print(f"Mean Delta Yaw Error: {np.mean(delta_yaws)}")
    print(f"Standard Deviation of Delta Yaw Error: {np.std(delta_yaws)}")
    print(f"Median Delta Yaw Error: {np.median(delta_yaws)}")

    # Show an histogram of the delta_yaw errors
    plt.figure()
    plt.hist(delta_yaws, bins=20, color='skyblue', edgecolor='black', linewidth=1.2)
    plt.title('Histogram of Delta Yaw Errors')
    plt.xlabel('Delta Yaw Error (degrees)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test CRODINO')
    parser.add_argument('--name', '-n', type=str, default='untitled', help='Path to save the model and results')
    parser.add_argument('--backbone', '-b', type=str, default='dinov3', help='Model to use')
    parser.add_argument('--loss', '-l', type=str, default='cosine_similarity', help='Loss to use for the Orientation Estimation')
    parser.add_argument('--dataset', '-d', type=str, default='cvglobal', help='Dataset to use')
    parser.add_argument('--create_figs', '-s', type=str, default='true', help='Create figures')
    parser.add_argument('--save_mode', '-m', type=str, default='separate', choices=['combined', 'separate', 'both'],
                        help='Save only the combined 2x2 figure, only the 4 separate figures, or both')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with visualizations')
    args = parser.parse_args()
    
    # Get Dataset Images
    dataset_name = args.dataset
    if dataset_name.lower() == "cvglobal":
        dataset_path = r'D:\cross_view_localization_DSM\Data\CVGlobal'
        train_filenames, _ = sample_cvusa_images(dataset_path, sample_percentage=0.002, split_ratio=1, groundtype='panos', shortcut=True)

    # Settings
    image_size = 224
    aerial_scaling = 2
    provide_paths = False
    BATCH_SIZE = 8

    # Get the processor and transforms
    processors = get_processors(args.backbone)
    transform_ground, transform_aerial = get_transforms(processors, image_size, aerial_scaling)

    # Instantiate the dataset and dataloader
    paired_dataset = PairedImagesDataset(train_filenames, transform_aerial=transform_aerial, transform_ground=transform_ground, cutout_from_pano=True)
    data_loader = DataLoader(paired_dataset, batch_size=BATCH_SIZE, shuffle=True)

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

    grid_size = (14, 14)

    test(processors,
         loss,
         data_loader,
         grid_size,
         device,
         savepath=args.name,
         debug=args.debug,
         save_mode=args.save_mode)
