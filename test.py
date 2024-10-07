from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import sys
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from matplotlib.patches import ConnectionPatch
from torch.utils.data import Dataset, DataLoader
import os
import random
import argparse
from tqdm import tqdm

from dataset import PairedImagesDataset, sample_paired_images
from model import CroDINO, Dinov2Matcher, CosineSimilarityLoss, get_combined_embedding_visualization_all
from skyfilter import SkyFilter

from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import requests

import matplotlib
matplotlib.use('TkAgg')  # or 'Agg' for non-GUI


def apply_depth_estimation(model, image_processor, image, grid_size=16, debug=False):
    """
    Applies depth estimation to the image, and returns the depth map along with
    a downsampled version of the depth map on a 16x16 grid where each grid cell
    contains the average depth value of the pixels in that cell.
    
    Parameters:
    - model: The depth estimation model.
    - image_processor: The processor for the depth estimation model.
    - image: The image to be processed for depth estimation.
    - grid_size: The size of the token grid (default is 16).
    - debug: Enable visualization of intermediate steps (default is False).

    Returns:
    - depth_map: The estimated depth map for the image.
    - depth_map_grid: The downsampled 16x16 depth map, containing average depth values.
    """
    # Prepare image for the model
    inputs = image_processor(images=image, return_tensors="pt")

    # Dimensions of the image
    height, width = image.shape[:2]

    # Get the predicted depth
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Interpolate to the original image size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.shape[:2][::-1],  # [width, height]
        mode="bicubic",
        align_corners=False,
    )

    # Convert the tensor to a NumPy array and remove extra dimensions
    depth_map = prediction.squeeze().cpu().numpy()

    # Normalize the depth map to the range [0, 1]
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

    # Calculate the size of each grid cell
    cell_height = height // grid_size
    cell_width = width // grid_size

    # Create the downsampled depth map grid
    depth_map_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    for i in range(grid_size):
        for j in range(grid_size):
            start_x = j * cell_width
            start_y = i * cell_height
            end_x = (j + 1) * cell_width if j < grid_size - 1 else width
            end_y = (i + 1) * cell_height if i < grid_size - 1 else height
            
            # Calculate the average depth value in the cell
            cell_depth = depth_map[start_y:end_y, start_x:end_x]
            depth_map_grid[i, j] = np.mean(cell_depth)

    # Visualize the depth map and downsampled depth map grid if in debug mode
    if debug:
        plt.figure(figsize=(18, 8))
        plt.subplot(131)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(depth_map, cmap='plasma')
        plt.colorbar()
        plt.title('Depth Map')
        plt.axis('off')
        plt.subplot(133)
        plt.imshow(depth_map_grid, cmap='plasma')
        plt.colorbar()
        plt.title('Downsampled Depth Map (16x16 Grid)')
        plt.axis('off')
        plt.show()

    return depth_map, depth_map_grid

def apply_sky_filter(sky_filter, ground_image_vis, grid_size, debug=False):
    """
    Applies a sky filter to remove the sky from an image.
    Parameters:
    - sky_filter: The sky filter object used to process the image.
    - ground_image_vis: The original image with the sky.
    - grid_size: The size of the grid used for dividing the image.
    - debug: Optional parameter to enable visualization of intermediate steps. Default is False.
    Returns:
    - ground_image_no_sky: The image with the sky removed.
    - sky_mask: The binary mask indicating the sky regions in the image.
    - grid_mask: The binary mask indicating the ground regions in the image grid.
    The function applies the sky filter to the ground_image_vis to remove the sky from the image. It then divides the image into a grid of cells and determines whether each cell is sky or ground based on the majority voting of the corresponding region in the sky mask. The resulting image with the sky removed, the sky mask, and the grid mask are returned.
    If debug is set to True, the function also visualizes the original image, the sky mask, the image without sky, and the grid mask.
    """

    # Process the image array directly
    ground_image_no_sky, sky_mask = sky_filter.run_img_array(ground_image_vis)

    # Dimensions of the image
    height, width = ground_image_no_sky.shape[:2]

    # Calculate the size of each grid cell
    cell_height = height // grid_size
    cell_width = width // grid_size

    # Initialize the grid mask
    grid_mask = np.zeros((grid_size, grid_size), dtype=np.uint8)

    # Loop over each cell in the grid
    for i in range(grid_size):
        for j in range(grid_size):
            # Define the region of interest (ROI) for this cell
            start_x = j * cell_width
            start_y = i * cell_height
            end_x = (j + 1) * cell_width if j < grid_size - 1 else width
            end_y = (i + 1) * cell_height if i < grid_size - 1 else height
            
            # Extract the cell from the sky mask
            cell = sky_mask[start_y:end_y, start_x:end_x]
            
            # Apply majority voting: if more than half of the cell is sky, mark it as sky
            if np.mean(cell) > 127:  # Since the mask is binary, 127 is the midpoint
                grid_mask[i, j] = 1  # Mark as ground
            else:
                grid_mask[i, j] = 0  # Mark as ground

    # Visualize the original image, mask, sky-removed image and grid mask
    if debug:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 18))
        ax1.imshow(ground_image_vis)
        ax1.set_title("Original Image")
        ax1.axis('off')
        ax2.imshow(sky_mask, cmap='gray')
        ax2.set_title("Sky Mask")
        ax2.axis('off')
        ax3.imshow(ground_image_no_sky)
        ax3.set_title("Image Without Sky")
        ax3.axis('off')
        ax4.imshow(grid_mask, cmap='gray')
        ax4.set_title("Grid Mask")
        ax4.axis('off')
        plt.show()

    return ground_image_no_sky, sky_mask, grid_mask

def get_direction_tokens(tokens, angle=None, vertical_idx=None, grid_size=16):
    """
    Retrieves direction tokens and their corresponding indices based on the given angle or vertical index.
    Parameters:
    - tokens (ndarray): The array of tokens.
    - angle (float, optional): The angle in degrees for radial direction. Defaults to None.
    - vertical_idx (int, optional): The vertical index for vertical line. Defaults to None.
    - grid_size (int, optional): The size of the grid. Defaults to 16.
    Returns:
    - direction_tokens (ndarray): The array of direction tokens.
    - indices (list): The list of indices corresponding to the direction tokens.
    Notes:
    - If angle is provided, the function retrieves direction tokens in a radial direction.
    - If vertical_idx is provided, the function retrieves direction tokens in a vertical line.
    - The function returns an empty array if neither angle nor vertical_idx is provided.
    - The function stops retrieving tokens if they are out of bounds.
    """
    
    if angle is not None:  # Radial direction
        center = (grid_size // 2, grid_size // 2)
        direction_tokens = []
        indices = []
        for r in range(grid_size):
            delta = 0
            x = int(center[0] + (r+delta) * np.cos(np.deg2rad(angle)))
            y = int(center[1] - (r+delta) * np.sin(np.deg2rad(angle)))
            if 0 <= x < grid_size and 0 <= y < grid_size:
                idx = y * grid_size + x
                if idx < tokens.shape[0]:  # Ensure index is within bounds
                    direction_tokens.append(tokens[idx])
                    indices.append((y, x))
                else:
                    break  # Stop if out of bounds
            else:
                break  # Stop if out of bounds
        return np.array(direction_tokens), indices
    elif vertical_idx is not None:  # Vertical line
        direction_tokens = tokens[vertical_idx::grid_size]  # extract each vertical line
        return direction_tokens, [(i, vertical_idx) for i in range(grid_size)]
        
def find_alignment(fore_vert_avg_tokens, midd_vert_avg_tokens, back_vert_avg_tokens, fore_rad_avg_tokens, midd_rad_avg_tokens, back_rad_avg_tokens, grid_size, image_span, debug=False):
    """
    Finds the alignment between averaged vertical tokens and averaged radial tokens.
    Parameters:
    - averaged_vertical_tokens (ndarray): A numpy array containing the averaged vertical tokens.
    - averaged_radial_tokens (ndarray): A numpy array containing the averaged radial tokens.
    - grid_size (int): The size of the grid.
    - image_span (float): The span of the image.
    Returns:
    - best_orientation (float): The best orientation in degrees.
    - distances (list): A list of distances for each orientation.
    - min_distance (float): The minimum distance.
    - confidence (float): The confidence score.
    """

    angle_step = image_span / grid_size
    min_distance = float('inf')
    distances = []

    for j, beta in enumerate(np.arange(0, 360, angle_step)):
        cone_distance = 0
        for i in range(grid_size+1):

            fore_rad_token = fore_rad_avg_tokens[int(j + i - grid_size/2) % fore_rad_avg_tokens.shape[0]]
            midd_rad_token = midd_rad_avg_tokens[int(j + i - grid_size/2) % midd_rad_avg_tokens.shape[0]]
            back_rad_token = back_rad_avg_tokens[int(j + i - grid_size/2) % back_rad_avg_tokens.shape[0]]
            # print(f"beta: {beta:.2f} \tangle: {(j + i - grid_size/2)*angle_step} \tindex: {int(j + i - grid_size/2) % averaged_radial_tokens.shape[0]}")       

            vert_avg_tokens = np.vstack((fore_vert_avg_tokens[(grid_size-1)-i], midd_vert_avg_tokens[(grid_size-1)-i], back_vert_avg_tokens[(grid_size-1)-i]))
            # vert_avg_tokens = np.vstack((fore_vert_avg_tokens[i-1], midd_vert_avg_tokens[i-1], back_vert_avg_tokens[i-1]))
            rad_tokens = np.vstack((fore_rad_token, midd_rad_token, back_rad_token))

            cone_distance += np.linalg.norm((1 - np.dot(vert_avg_tokens, np.transpose(rad_tokens))))       # cosine distance
            # cone_distance += np.linalg.norm((1 - np.dot(fore_rad_token, np.transpose(fore_vert_avg_tokens[(grid_size-1)-i]))))       # cosine distance

        cone_distance /= grid_size
        if cone_distance < min_distance:
            min_distance = cone_distance
            best_orientation = beta
            if debug:
                print(f"Min Distance: {min_distance:.4f} \tBest Orientation: {best_orientation}°")
        distances.append(cone_distance)

    # Compute confidence
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    confidence = (mean_distance - min_distance) / std_distance  # Z-score

    return best_orientation, distances, min_distance, confidence

def get_averaged_vertical_tokens(angle_step, normalized_features1, grid_size, sky_grid, depth_map_grid, threshold=0.5):
    
    averaged_foreground_tokens = []
    averaged_middleground_tokens = []
    averaged_background_tokens = []
    for i in range(grid_size):
        vertical_tokens, indices = get_direction_tokens(normalized_features1, vertical_idx=i, grid_size=grid_size)
        valid_tokens = []
        foreground_weights = []
        middleground_weights = []
        background_weights = []
        for token, (y, x) in zip(vertical_tokens, indices):
            if sky_grid[y, x] == 1:  # 1 indicates ground, 0 indicates sky
                valid_tokens.append(token)
                foreground_weights.append(depth_map_grid[y, x])
                if depth_map_grid[y, x] <= 0.5:
                    middleground_weights.append((1 / threshold) * depth_map_grid[y, x])
                else:
                    middleground_weights.append((1 - depth_map_grid[y, x]) / depth_map_grid[y, x])
                background_weights.append(1 - depth_map_grid[y, x])
        
        if valid_tokens:
            valid_tokens = np.array(valid_tokens)
            foreground_weights = np.array(foreground_weights)
            middleground_weights = np.array(middleground_weights)
            background_weights = np.array(background_weights)
            foreground_weights /= np.sum(foreground_weights)  # Normalize the weights
            middleground_weights /= np.sum(middleground_weights)
            background_weights /= np.sum(background_weights)
            
            # Calculate weighted average only on valid (non-sky) tokens
            foreground_avg = np.average(valid_tokens, axis=0, weights=foreground_weights)
            middleground_avg = np.average(valid_tokens, axis=0, weights=middleground_weights)
            background_avg = np.average(valid_tokens, axis=0, weights=background_weights)
            averaged_foreground_tokens.append(foreground_avg)
            averaged_middleground_tokens.append(middleground_avg)
            averaged_background_tokens.append(background_avg)
        else:
            # If no valid tokens are found (i.e., entire column is sky), append a zero vector or any placeholder
            averaged_foreground_tokens.append(np.zeros_like(vertical_tokens[0]))
            averaged_middleground_tokens.append(np.zeros_like(vertical_tokens[0]))
            averaged_background_tokens.append(np.zeros_like(vertical_tokens[0]))
    
    averaged_foreground_tokens = np.array(averaged_foreground_tokens)
    averaged_middleground_tokens = np.array(averaged_middleground_tokens)
    averaged_background_tokens = np.array(averaged_background_tokens)

    return averaged_foreground_tokens, averaged_middleground_tokens, averaged_background_tokens

def get_averaged_radial_tokens(angle_step, normalized_features2, grid_size, sky_grid, depth_map_grid):
    
    averaged_fore_radial_tokens = []
    averaged_middle_radial_tokens = []
    averaged_back_radial_tokens = []
    for beta in np.arange(0, 360, angle_step):
        radial_tokens, _ = get_direction_tokens(normalized_features2, angle=beta, grid_size=grid_size)
        increasing_weights = np.linspace(0, 1, len(radial_tokens))
        decreasing_weights = np.linspace(1, 0, len(radial_tokens))

        # Ensure middle_weights has the same length as radial_tokens
        half_len = len(radial_tokens) // 2
        middle_weights = np.hstack((np.linspace(0, 1, half_len, endpoint=False), np.linspace(1, 0, len(radial_tokens) - half_len)))

        increasing_weights /= np.sum(increasing_weights)
        decreasing_weights /= np.sum(decreasing_weights)
        middle_weights /= np.sum(middle_weights)

        # print("radial tokens:", radial_tokens.shape)
        # print("middle weights:", middle_weights.shape)

        foreground_avg = np.average(radial_tokens, axis=0, weights=decreasing_weights)
        middleground_avg = np.average(radial_tokens, axis=0, weights=middle_weights)
        background_avg = np.average(radial_tokens, axis=0, weights=increasing_weights)

        averaged_fore_radial_tokens.append(foreground_avg)
        averaged_middle_radial_tokens.append(middleground_avg)
        averaged_back_radial_tokens.append(background_avg)

    averaged_fore_radial_tokens = np.array(averaged_fore_radial_tokens)         # 64x768
    averaged_middle_radial_tokens = np.array(averaged_middle_radial_tokens)
    averaged_back_radial_tokens = np.array(averaged_back_radial_tokens)

    return averaged_fore_radial_tokens, averaged_middle_radial_tokens, averaged_back_radial_tokens

def test(model, data_loader, device, savepath='results', create_figs=False, debug=False):

    # Create results directory if it doesn't exist
    results_dir = savepath
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Initialize the sky filter
    sky_filter = SkyFilter() 

    # Initialize the depth estimation model
    image_processor_depth = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
    depth_model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")

    delta_yaws = []
    delta_yaws_combined = []

    threshold = 0.4

    with tqdm(enumerate(data_loader), total=len(data_loader), desc="Processing Batches") as pbar:
        for batch_idx, (ground_images, aerial_images, fovs, yaws, pitchs) in pbar:
            ground_images = ground_images.to(device)
            aerial_images = aerial_images.to(device)

            batch_size = ground_images.size(0)

            if debug:
                print(f"Batch {batch_idx}: fovs", fovs)
                print(f"Batch {batch_idx}: yaws", yaws)
                print(f"Batch {batch_idx}: pitchs", pitchs)

            # Assuming you want to handle each image in the batch individually
            for i in range(batch_size):  # Iterate over batch size
                ground_image = ground_images[i:i+1]
                aerial_image = aerial_images[i:i+1]
                fov_x, fov_y = fovs
                fov = (fov_x[i].item(), fov_y[i].item())
                yaw = yaws[i].item()
                pitch = pitchs[i].item()
                
                # Compute the output of the model
                ground_tokens, aerial_tokens, _ = model(ground_image, aerial_image, debug=False)

                if debug:
                    print("fov", fov)
                    print("yaw", yaw)
                    print("pitch", pitch)

                # Calculate the number of patches for ground and aerial images
                num_patches_ground = (ground_image.shape[-1] // model.patch_size) * (ground_image.shape[-2] // model.patch_size)
                num_patches_aerial = (aerial_image.shape[-1] // model.patch_size) * (aerial_image.shape[-2] // model.patch_size)

                # Convert images to numpy for visualization
                ground_image_np = ground_image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
                aerial_image_np = aerial_image.squeeze().permute(1, 2, 0).detach().cpu().numpy()

                ground_image_vis = ground_image.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255
                aerial_image_vis = aerial_image.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255
                ground_image_vis = ground_image_vis.astype(np.uint8)
                aerial_image_vis = aerial_image_vis.astype(np.uint8)

                # Apply sky filter
                ground_image_no_sky, sky_mask, sky_grid = apply_sky_filter(sky_filter, ground_image_vis, grid_size=16, debug=False)

                # Apply depth estimation
                depth_map, depth_map_grid = apply_depth_estimation(depth_model, image_processor_depth, ground_image_no_sky, debug=False)

                # Normalize the features
                normalized_features1 = normalize(ground_tokens.squeeze().detach().cpu().numpy(), axis=1)
                normalized_features2 = normalize(aerial_tokens.squeeze().detach().cpu().numpy(), axis=1)
                if debug:
                    print("normalized_features1.shape:", normalized_features1.shape)
                    print("normalized_features2.shape:", normalized_features2.shape)

                grid_size = int(np.sqrt(normalized_features1.shape[0]))  # assuming square grid
                if debug:
                    print("grid_size:", grid_size)

                fov_x = 90                          # horizontal fov in degrees
                angle_step = fov_x / grid_size
            
                # Compute Averaged Tokens using the weight vector, excluding sky tokens
                fore_vert_avg_tokens, midd_vert_avg_tokens, back_vert_avg_tokens = get_averaged_vertical_tokens(angle_step, normalized_features1, grid_size, sky_grid, depth_map_grid, threshold=threshold)
                fore_rad_avg_tokens, midd_rad_avg_tokens, back_rad_avg_tokens = get_averaged_radial_tokens(angle_step, normalized_features2, grid_size, sky_grid, depth_map_grid)
                
                if debug:
                    print("averaged vertical tokens: ", fore_vert_avg_tokens.shape)
                    print("averaged radial tokens: ", fore_rad_avg_tokens.shape)   

                # Find the best alignment
                best_orientation, distances, min_distance, confidence = find_alignment(fore_vert_avg_tokens, midd_vert_avg_tokens, back_vert_avg_tokens, fore_rad_avg_tokens, midd_rad_avg_tokens, back_rad_avg_tokens, grid_size, fov_x, debug=False)

                delta_yaw = np.abs(((90 - (yaw - 180)) - best_orientation + 180) % 360 - 180)
                delta_yaws.append(delta_yaw)

                if create_figs or debug:

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
                    ax3.set_title("Distance over Orientations - Confidence: {:.4f}".format(confidence))
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
                        color = plt.cm.plasma((distances[j] - min_dist) / (max_dist - min_dist))  # Normalize distances for color map
                        ax4.plot([center[0], end_x], [center[1], end_y], color=color)
                    ax4.set_title("Aerial Image with Distances")
                    ax4.axis('off')

                    norm = plt.Normalize(min_dist, max_dist)
                    sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=ax4)

                    # Determine the next available file number
                    file_count = len([name for name in os.listdir(results_dir) if name.startswith("sample") and name.endswith(".png")])
                    file_path = os.path.join(results_dir, f"sample_{file_count}.png")

                    if create_figs:
                        plt.savefig(file_path, dpi=300, bbox_inches='tight')

                    if debug:
                        plt.show()

                    if create_figs:
                        plt.close(fig)

                # Update pbar to show current delta_yaws average
                pbar.set_postfix({'Delta Yaw': np.mean(delta_yaws)})
                pbar.update()

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

    # Save results to a text file
    np.savetxt(os.path.join(results_dir, 'delta_yaws.txt'), delta_yaws, header="Delta Yaws (degrees)")
    with open(os.path.join(results_dir, 'delta_yaws.txt'), 'a') as f:
        f.write(f"\n\nMean Delta Yaw Error: {np.mean(delta_yaws)}\n")
        f.write(f"Standard Deviation of Delta Yaw Error: {np.std(delta_yaws)}\n")
        f.write(f"Median Delta Yaw Error: {np.median(delta_yaws)}\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test CRODINO')
    parser.add_argument('--save_path', '-p', type=str, default='results', help='Path to save the model and results')
    parser.add_argument('--debug', '-d', type=str, default='False', help='Debug mode')
    parser.add_argument('--create_figs', '-s', type=str, default='False', help='Create figures')
    args = parser.parse_args()
    
    # Sample paired images
    dataset_path = '/home/lrusso/cvusa/CVPR_subset'
    train_filenames, val_filenames = sample_paired_images(dataset_path, sample_percentage=0.005, split_ratio=0.8, groundtype='panos')

    # Settings
    image_size = 224
    aerial_scaling = 2
    provide_paths = False
    BATCH_SIZE = 8

    transform_ground = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop((image_size, image_size)),
        transforms.ToTensor()
    ])

    transform_aerial = transforms.Compose([
        transforms.Resize((image_size*aerial_scaling, image_size*aerial_scaling)),
        transforms.CenterCrop((image_size, image_size)),
        transforms.ToTensor()
    ])

    # Instantiate the dataset and dataloader
    paired_dataset = PairedImagesDataset(train_filenames, transform_aerial=transform_aerial, transform_ground=transform_ground, cutout_from_pano=True)
    data_loader = DataLoader(paired_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Define the Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the Model
    repo_name="facebookresearch/dinov2"
    model_name="dinov2_vitb14"
    model = CroDINO(repo_name, model_name, pretrained=True).to(device)

    test(model, data_loader, device, savepath=args.save_path, create_figs=args.create_figs.lower() == 'true', debug=args.debug.lower() == 'true')
