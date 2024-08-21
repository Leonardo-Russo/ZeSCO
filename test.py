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

import matplotlib
matplotlib.use('TkAgg')  # or 'Agg' for non-GUI


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

    # Visualize the original image, mask, and the sky-removed image
    if debug:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        ax1.imshow(ground_image_vis)
        ax1.set_title("Original Image")
        ax1.axis('off')
        ax2.imshow(sky_mask, cmap='gray')
        ax2.set_title("Sky Mask")
        ax2.axis('off')
        ax3.imshow(ground_image_no_sky)
        ax3.set_title("Image Without Sky")
        ax3.axis('off')
        plt.show()

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

    # Visualize the grid mask
    if debug:
        plt.figure(figsize=(6, 6))
        plt.imshow(grid_mask, cmap='gray')
        plt.title("Grid Mask")
        plt.axis('off')
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
            delta = 4
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
        

def find_alignment(averaged_vertical_tokens, averaged_radial_tokens, grid_size, image_span, debug=False):
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

            vertical_token = averaged_vertical_tokens[(grid_size-1)-i]
            radial_token = averaged_radial_tokens[int(j + i - grid_size/2) % averaged_radial_tokens.shape[0]]
            # print(f"beta: {beta:.2f} \tangle: {(j + i - grid_size/2)*angle_step} \tindex: {int(j + i - grid_size/2) % averaged_radial_tokens.shape[0]}")       

            cone_distance += (1 - np.dot(vertical_token, radial_token))  # Cosine distance
            # cone_distance += np.linalg.norm(vertical_token - radial_token)  # Euclidean distance

        cone_distance /= grid_size
        if cone_distance < min_distance:
            min_distance = cone_distance
            best_orientation = beta
            if debug:
                print(f"Min Distance: {min_distance:.4f} \tBest Orientation: {best_orientation}°")
        distances.append(cone_distance)

    # Compute Combined Alignment
    num_angles = len(distances)
    combined_distances = []

    # Sum opposite directions' distances
    for i in range(num_angles // 2):
        combined_distance = distances[i] + distances[(i + num_angles // 2) % num_angles]
        combined_distances.append(combined_distance)

    # Find the line with the least combined distance
    min_combined_distance = min(combined_distances)
    min_combined_index = combined_distances.index(min_combined_distance)

    # Compare the two possible directions on that line and select the best one
    if distances[min_combined_index] < distances[(min_combined_index + num_angles // 2) % num_angles]:
        best_combined_orientation = min_combined_index * angle_step
        combined_min_distance = distances[min_combined_index]
    else:
        best_combined_orientation = (min_combined_index + num_angles // 2) * angle_step
        combined_min_distance = distances[(min_combined_index + num_angles // 2) % num_angles]

    # Compute confidence
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    confidence = (mean_distance - min_distance) / std_distance  # Z-score

    return best_orientation, distances, min_distance, confidence, best_combined_orientation, combined_min_distance


def test(model, data_loader, device, savepath='results', debug=False):

    # Create results directory if it doesn't exist
    results_dir = savepath
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Initialize the sky filter
    sky_filter = SkyFilter()

    delta_yaws = []
    delta_yaws_combined = []

    for batch_idx, (ground_images, aerial_images, fovs, yaws, pitchs) in tqdm(enumerate(data_loader), total=len(data_loader), desc="Processing Batches"):
        ground_images = ground_images.to(device)
        aerial_images = aerial_images.to(device)

        BATCH_SIZE = ground_images.size(0)

        if debug:
            print(f"Batch {batch_idx}: fovs", fovs)
            print(f"Batch {batch_idx}: yaws", yaws)
            print(f"Batch {batch_idx}: pitchs", pitchs)

        # Assuming you want to handle each image in the batch individually
        for i in range(BATCH_SIZE):  # Iterate over batch size
            ground_image = ground_images[i:i+1]
            aerial_image = aerial_images[i:i+1]
            fov_x, fov_y = fovs
            fov = (fov_x[i].item(), fov_y[i].item())
            yaw = yaws[i].item()
            pitch = pitchs[i].item()
            
            # Compute the output of the model
            ground_tokens, aerial_tokens, attention = model(ground_image, aerial_image, debug=False)

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


            ground_image_no_sky, sky_mask, grid_mask = apply_sky_filter(sky_filter, ground_image_vis, grid_size=16, debug=False)

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
            averaged_vertical_tokens = []
            for i in range(grid_size):
                vertical_tokens, indices = get_direction_tokens(normalized_features1, vertical_idx=i, grid_size=grid_size)
                valid_tokens = []
                valid_weights = []
                for token, (y, x) in zip(vertical_tokens, indices):
                    if grid_mask[y, x] == 1:  # 1 indicates ground, 0 indicates sky
                        valid_tokens.append(token)
                        valid_weights.append(1.0)  # You can adjust the weights if needed
                
                if valid_tokens:
                    valid_tokens = np.array(valid_tokens)
                    valid_weights = np.array(valid_weights)
                    valid_weights /= np.sum(valid_weights)  # Normalize the weights
                    
                    # Calculate weighted average only on valid (non-sky) tokens
                    weighted_avg = np.average(valid_tokens, axis=0, weights=valid_weights)
                    averaged_vertical_tokens.append(weighted_avg)
                else:
                    # If no valid tokens are found (i.e., entire column is sky), append a zero vector or any placeholder
                    averaged_vertical_tokens.append(np.zeros_like(vertical_tokens[0]))
            averaged_vertical_tokens = np.array(averaged_vertical_tokens)


            averaged_radial_tokens = []
            for beta in np.arange(0, 360, angle_step):
                radial_tokens, _ = get_direction_tokens(normalized_features2, angle=beta, grid_size=grid_size)
                # increasing_weights = np.linspace(0.1, 1, len(radial_tokens))
                increasing_weights = np.linspace(1, 1, len(radial_tokens))
                increasing_weights /= np.sum(increasing_weights)
                weighted_avg = np.average(radial_tokens, axis=0, weights=increasing_weights)
                averaged_radial_tokens.append(weighted_avg)
            averaged_radial_tokens = np.array(averaged_radial_tokens)

            if debug:
                print("averaged_vertical_tokens.shape:", averaged_vertical_tokens.shape)
                print("averaged_radial_tokens.shape:", averaged_radial_tokens.shape)   

            # Find the best alignment
            best_orientation, distances, min_distance, confidence, best_combined_orientation, combined_min_distance = find_alignment(averaged_vertical_tokens, averaged_radial_tokens, grid_size, fov_x)

            delta_yaw = np.abs(((90 - (yaw - 180)) - best_orientation + 180) % 360 - 180)
            delta_yaw_combined = np.abs(((90 - (yaw - 180)) - best_combined_orientation + 180) % 360 - 180)
            delta_yaws.append(delta_yaw)
            delta_yaws_combined.append(delta_yaw_combined)

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))

            ax1.imshow(ground_image_np)
            ax1.set_title("Ground Image - Yaw: {:.1f}°".format(yaw))
            ax1.axis('off')

            ax2.imshow(aerial_image_np)
            radius = aerial_image_np.shape[0] // 2
            center = (aerial_image_np.shape[1] // 2, aerial_image_np.shape[0] // 2)
            end_x = int(center[0] + radius * np.cos(np.deg2rad(best_orientation)))
            end_y = int(center[1] - radius * np.sin(np.deg2rad(best_orientation)))
            end_x_combined = int(center[0] + radius * np.cos(np.deg2rad(best_combined_orientation)))
            end_y_combined = int(center[1] - radius * np.sin(np.deg2rad(best_combined_orientation)))
            end_x_GT = int(center[0] + radius * np.cos(np.deg2rad(90 - (yaw - 180))))
            end_y_GT = int(center[1] - radius * np.sin(np.deg2rad(90 - (yaw - 180))))
            line_pred = ax2.plot([center[0], end_x], [center[1], end_y], color='red', linestyle='--', label='Prediction')
            line_pred_combined = ax2.plot([center[0], end_x_combined], [center[1], end_y_combined], color='blue', linestyle='--', label='Combined Prediction')
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

            # Save the figure
            plt.savefig(file_path, dpi=300, bbox_inches='tight')

            if debug:
                plt.show()

            plt.close(fig)

    # Output the delta_yaw errors
    delta_yaws = np.array(delta_yaws)
    print("Delta Yaw Errors (degrees):")
    print(delta_yaws)
    print(f"Mean Delta Yaw Error: {np.mean(delta_yaws)}")
    print(f"Standard Deviation of Delta Yaw Error: {np.std(delta_yaws)}")

    # Optionally save errors to a file
    np.savetxt(os.path.join(results_dir, 'delta_yaws.txt'), delta_yaws, header="Delta Yaws (degrees)")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test CRODINO')
    parser.add_argument('--save_path', '-p', type=str, default='results', help='Path to save the model and results')
    parser.add_argument('--debug', '-d', type=str, default='False', help='Debug mode')
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

    test(model, data_loader, device, savepath=args.save_path, debug=args.debug.lower() == 'true')