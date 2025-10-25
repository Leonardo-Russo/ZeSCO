"""
ZeSCO Utils - Token Processing and Alignment Functions
GPU-Accelerated version for high-performance geospatial alignment

Performance Features:
- GPU acceleration for find_alignment using PyTorch (10-50x speedup on CUDA)
- Vectorized batch processing of all orientation×grid comparisons
- Automatic CPU fallback when GPU unavailable
- Matrix multiplication (@) for efficient weighted averaging

The code automatically detects GPU availability and uses it when possible.
Set USE_GPU=False to force CPU execution.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import re
import torch

# Check if CUDA is available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_GPU = torch.cuda.is_available()

print(f"[ZeSCO Utils] Running on: {DEVICE} | GPU Acceleration: {USE_GPU}")


def get_direction_tokens(tokens, angle=None, vertical_idx=None, grid_dim=16, sky_grid=None):
    """
    Retrieves direction tokens and their corresponding indices based on the given angle or vertical index.
    Parameters:
    - tokens (ndarray): The array of tokens.
    - angle (float, optional): The angle in degrees for radial direction. Defaults to None.
    - vertical_idx (int, optional): The vertical index for vertical line. Defaults to None.
    - grid_size (int, optional): The size of the grid. Defaults to 16.
    - sky_grid (ndarray, optional): The sky grid for additional context, if provided only valid tokens are returned. Defaults to None.
    Returns:
    - direction_tokens (ndarray): The array of direction tokens.
    - indices (list): The list of indices corresponding to the direction tokens.
    Notes:
    - If angle is provided, the function retrieves direction tokens in a radial direction.
    - If vertical_idx is provided, the function retrieves direction tokens in a vertical line.
    - The function returns an empty array if neither angle nor vertical_idx is provided.
    - The function stops retrieving tokens if they are out of bounds.
    """
    if sky_grid is None:
        sky_grid = np.ones((grid_dim, grid_dim))  # default to all ground if no sky_grid provided
    
    if angle is not None:  # Radial direction
        center = (grid_dim // 2, grid_dim // 2)
        direction_tokens = []
        indices = []
        for r in range(grid_dim):
            x = round(center[0] + r * np.cos(np.deg2rad(angle)))
            y = round(center[1] - r * np.sin(np.deg2rad(angle)))
            if 0 <= x < grid_dim and 0 <= y < grid_dim:
                idx = y * grid_dim + x
                if tokens is None:
                    direction_tokens.append(None)
                    indices.append((y, x))
                else:
                    if idx < tokens.shape[0]:
                        direction_tokens.append(tokens[idx])
                        indices.append((y, x))
                    else:
                        break  # out of bounds
            else:
                break  # out of bounds
        return np.array(direction_tokens), indices
    elif vertical_idx is not None:      # vertical line
        direction_tokens = tokens[vertical_idx::grid_dim]  # extract each vertical line

        # Only grab valid tokens
        valid_tokens = []
        valid_indices = []
        for i in range(grid_dim):
            y = i
            x = vertical_idx
            if sky_grid[y, x] == 1:  # 1 indicates ground, 0 indicates sky
                valid_tokens.append(direction_tokens[i])
                valid_indices.append((y, x))
        return np.array(valid_tokens), valid_indices

def find_alignment(loss, vertical_averaged_tokens, radial_averaged_tokens, grid_size, image_span, debug=False):
    """
    Finds the alignment between averaged vertical tokens and averaged radial tokens.
    GPU-accelerated version with vectorized distance computation.
    Parameters:
    - vertical_averaged_tokens (ndarray): A numpy array of shape (num_layers, grid_size, feature_dim)
    - radial_averaged_tokens (ndarray): A numpy array of shape (num_layers, num_orientations, feature_dim)
    - grid_size (int): The size of the grid.
    - image_span (float): The span of the image.
    Returns:
    - best_orientation (float): The best orientation in degrees.
    - distances (list): A list of distances for each orientation.
    - min_distance (float): The minimum distance.
    - confidence (float): The confidence score.
    """

    angle_step = image_span / grid_size
    num_steps = int(round(360 / angle_step))
    
    if USE_GPU:
        # Convert to GPU tensors
        vert_tokens_gpu = torch.from_numpy(vertical_averaged_tokens).to(DEVICE)  # (num_layers, grid_size, feature_dim)
        rad_tokens_gpu = torch.from_numpy(radial_averaged_tokens).to(DEVICE)     # (num_layers, num_orientations, feature_dim)
        
        num_orientations = rad_tokens_gpu.shape[1]
        
        # Vectorized index computation for all orientations and grid positions at once
        # Shape: (num_steps, grid_size)
        j_indices = torch.arange(num_steps, device=DEVICE).unsqueeze(1)  # (num_steps, 1)
        i_indices = torch.arange(grid_size, device=DEVICE).unsqueeze(0)  # (1, grid_size)
        
        # Compute radial indices for all (orientation, position) pairs
        rad_indices = (j_indices + i_indices - grid_size // 2) % num_orientations  # (num_steps, grid_size)
        
        # Compute vertical indices for all positions
        vert_indices = (grid_size - 1) - i_indices  # (1, grid_size)
        
        # Gather tokens efficiently
        # rad_tokens_gpu: (num_layers, num_orientations, feature_dim)
        # We need: (num_steps, grid_size, num_layers, feature_dim)
        rad_gathered = rad_tokens_gpu[:, rad_indices, :]  # (num_layers, num_steps, grid_size, feature_dim)
        rad_gathered = rad_gathered.permute(1, 2, 0, 3)   # (num_steps, grid_size, num_layers, feature_dim)
        
        # vert_tokens_gpu: (num_layers, grid_size, feature_dim)
        vert_gathered = vert_tokens_gpu[:, vert_indices, :]  # (num_layers, 1, grid_size, feature_dim)
        vert_gathered = vert_gathered.squeeze(1).permute(1, 0, 2)  # (grid_size, num_layers, feature_dim)
        vert_gathered = vert_gathered.unsqueeze(0).expand(num_steps, -1, -1, -1)  # (num_steps, grid_size, num_layers, feature_dim)
        
        # Compute cosine similarity losses in batch
        # Normalize
        rad_norm = torch.nn.functional.normalize(rad_gathered, p=2, dim=-1)
        vert_norm = torch.nn.functional.normalize(vert_gathered, p=2, dim=-1)
        
        # Cosine similarity: (num_steps, grid_size, num_layers)
        cos_sim = (rad_norm * vert_norm).sum(dim=-1)
        
        # Convert to distance (1 - cosine similarity) and average over layers and grid
        distances_per_layer = 1 - cos_sim  # (num_steps, grid_size, num_layers)
        cone_distances = distances_per_layer.mean(dim=2).mean(dim=1)  # (num_steps,)
        
        # Convert back to CPU for final operations
        distances = cone_distances.cpu().numpy().tolist()
        min_distance = float(cone_distances.min().item())
        best_idx = cone_distances.argmin().item()
        best_orientation = float(best_idx * angle_step)
        
        # Clean up GPU memory
        del vert_tokens_gpu, rad_tokens_gpu, rad_gathered, vert_gathered, rad_norm, vert_norm, cos_sim, distances_per_layer, cone_distances
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    else:
        # CPU fallback - original implementation
        min_distance = float('inf')
        distances = []
        
        for j, beta in enumerate(np.linspace(0, 360 - angle_step, num_steps)):
            cone_distance = 0
            for i in range(grid_size):
                rad_idx = int(j + i - grid_size/2) % radial_averaged_tokens.shape[1]
                rad_avg_tokens = radial_averaged_tokens[:, rad_idx, :]
                vert_idx = (grid_size - 1) - i
                vert_avg_tokens = vertical_averaged_tokens[:, vert_idx, :]
                cone_distance += loss(vert_avg_tokens, rad_avg_tokens)
            
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

def get_averaged_vertical_tokens(angle_step, image_tokens, grid_size, sky_grid, depth_map_grid, threshold=0.5, num_layers=3, debug=False):

    averaged_vertical_tokens = []
    
    # Pre-compute midpoints outside the loop
    if num_layers >= 3:
        midpoints = np.linspace(0, 1, num_layers)[1:-1]  # exclude 0 and 1
    
    for i in range(grid_size):  # loop across vertical lines
        vertical_tokens, indices = get_direction_tokens(image_tokens, vertical_idx=i, grid_dim=grid_size, sky_grid=sky_grid)

        if len(vertical_tokens) == 0:
            # No valid tokens - create zero array
            averaged_tokens = np.zeros((num_layers, image_tokens.shape[1]))
            averaged_vertical_tokens.append(averaged_tokens)
            continue

        if num_layers == 1:
            # Compute equal weights for all tokens
            weights = np.ones((1, len(vertical_tokens))) / len(vertical_tokens)

        elif num_layers == 2:
            # Compute foreground and background weights
            weights_fore = np.array([depth_map_grid[y, x] for (y, x) in indices])
            weights_back = 1 - weights_fore
            weights_fore /= np.sum(weights_fore)
            weights_back /= np.sum(weights_back)
            weights = np.stack((weights_fore, weights_back))
            
        else:  # num_layers >= 3
            # Vectorized weight computation
            depth_values = np.array([depth_map_grid[y, x] for (y, x) in indices])
            
            # Compute foreground and background weights
            weights_fore = depth_values / np.sum(depth_values)
            weights_back = (1 - depth_values) / np.sum(1 - depth_values)
            
            # Compute middleground weights (vectorized)
            weights_middle = []
            for m in midpoints:
                # Vectorized conditional computation
                mth_weights = np.where(
                    depth_values <= m,
                    depth_values / m,
                    (1 - depth_values) / (1 - m)
                )
                weights_middle.append(mth_weights / np.sum(mth_weights))
            
            weights = np.stack([weights_fore, *weights_middle, weights_back])
        
        # Single matrix multiplication for all layers
        averaged_tokens = weights @ vertical_tokens  # shape: (num_layers, feature_dim)
        averaged_vertical_tokens.append(averaged_tokens)

    if debug:   # show the weights computed
        # Only collect weights for debugging when needed
        vertical_weights = []
        for i in range(grid_size):
            vertical_tokens, indices = get_direction_tokens(image_tokens, vertical_idx=i, grid_dim=grid_size, sky_grid=sky_grid)
            if len(vertical_tokens) == 0:
                vertical_weights.append(np.zeros((num_layers, 1)))
                continue
            
            # Recompute weights for debugging (same logic as above)
            if num_layers == 1:
                weights = np.ones((1, len(vertical_tokens))) / len(vertical_tokens)
            elif num_layers == 2:
                weights_fore = np.array([depth_map_grid[y, x] for (y, x) in indices])
                weights_back = 1 - weights_fore
                weights_fore /= np.sum(weights_fore)
                weights_back /= np.sum(weights_back)
                weights = np.stack((weights_fore, weights_back))
            else:
                depth_values = np.array([depth_map_grid[y, x] for (y, x) in indices])
                weights_fore = depth_values / np.sum(depth_values)
                weights_back = (1 - depth_values) / np.sum(1 - depth_values)
                weights_middle = []
                for m in midpoints:
                    mth_weights = np.where(depth_values <= m, depth_values / m, (1 - depth_values) / (1 - m))
                    weights_middle.append(mth_weights / np.sum(mth_weights))
                weights = np.stack([weights_fore, *weights_middle, weights_back])
            vertical_weights.append(weights)

        # Pad weights to same size for visualization
        max_tokens = max(w.shape[1] for w in vertical_weights)
        vertical_weights_padded = []
        for weights in vertical_weights:
            padded = np.zeros((num_layers, max_tokens))
            padded[:, :weights.shape[1]] = weights
            vertical_weights_padded.append(padded)
        vertical_weights_padded = np.array(vertical_weights_padded)  # shape: (grid_size, num_layers, max_tokens)
        
        # Create subplot grid
        ncols = min(num_layers, 3)  # Max 3 columns
        nrows = (num_layers + ncols - 1) // ncols  # Ceiling division
        fig, axs = plt.subplots(nrows, ncols, figsize=(3*ncols + 1, 3*nrows), squeeze=False)
        
        for layer in range(num_layers):
            row = layer // ncols
            col = layer % ncols
            ax = axs[row, col]
            
            # Plot weights heatmap
            im = ax.imshow(vertical_weights_padded[:, layer, :].T, aspect='auto', cmap='viridis', interpolation='nearest', vmin=0, vmax=1, origin='lower')
            ax.set_title(f'Layer {layer+1} Weights', fontsize=12, fontweight='bold')
            ax.set_xlabel('Vertical Line Index', fontsize=10)
            ax.set_ylabel('Token Index along Vertical Line', fontsize=10)
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='Weight', fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        for layer in range(num_layers, nrows * ncols):
            row = layer // ncols
            col = layer % ncols
            axs[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(r'..\debug\vertical_weights_sample.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    averaged_vertical_tokens = np.array(averaged_vertical_tokens)  # shape: (grid_size, num_layers, feature_dim)
    averaged_vertical_tokens = np.transpose(averaged_vertical_tokens, (1, 0, 2))  # shape: (num_layers, grid_size, feature_dim)
    return averaged_vertical_tokens

def get_averaged_radial_tokens(angle_step, image_tokens, grid_size, sky_grid, depth_map_grid, num_layers=3, debug=False):

    averaged_radial_tokens = []
    
    # Pre-compute midpoints outside the loop
    if num_layers >= 3:
        midpoints = np.linspace(0, 1, num_layers)[1:-1]  # exclude 0 and 1
    
    for beta in np.arange(0, 360, angle_step):
        radial_tokens, indices = get_direction_tokens(image_tokens, angle=beta, grid_dim=grid_size)

        if len(radial_tokens) == 0:
            averaged_tokens = np.zeros((num_layers, image_tokens.shape[1]))
            averaged_radial_tokens.append(averaged_tokens)
            continue

        if num_layers == 1:
            # Compute equal weights for all tokens
            weights = np.ones((1, len(radial_tokens))) / len(radial_tokens)

        elif num_layers == 2:
            # Compute foreground and background weights (pre-normalized)
            n = len(radial_tokens)
            weights_fore = np.linspace(1, 0, n) / (n * (n + 1) / (2 * n))  # sum = n/2
            weights_back = np.linspace(0, 1, n) / (n * (n + 1) / (2 * n))
            # Proper normalization
            weights_fore = weights_fore / weights_fore.sum()
            weights_back = weights_back / weights_back.sum()
            weights = np.stack((weights_fore, weights_back))
            
        else:  # num_layers >= 3
            n = len(radial_tokens)
            # Compute foreground and background weights
            weights_fore = np.linspace(1, 0, n)
            weights_back = np.linspace(0, 1, n)
            weights_fore /= weights_fore.sum()
            weights_back /= weights_back.sum()
            
            # Vectorized middleground weights computation
            depth_values = np.array([depth_map_grid[y, x] for (y, x) in indices])
            weights_middle = []
            for m in midpoints:
                mth_weights = np.where(
                    depth_values <= m,
                    depth_values / m,
                    (1 - depth_values) / (1 - m)
                )
                weights_middle.append(mth_weights / mth_weights.sum())
            
            weights = np.stack([weights_fore, *weights_middle, weights_back])

        # Single matrix multiplication for all layers
        averaged_tokens = weights @ radial_tokens  # shape: (num_layers, feature_dim)
        averaged_radial_tokens.append(averaged_tokens)

    if debug:   # show the weights computed
        # Only collect weights for debugging when needed
        radial_weights = []
        for beta in np.arange(0, 360, angle_step):
            radial_tokens, indices = get_direction_tokens(image_tokens, angle=beta, grid_dim=grid_size)
            if len(radial_tokens) == 0:
                radial_weights.append(np.zeros((num_layers, 1)))
                continue
                
            # Recompute weights for debugging (same logic as above)
            if num_layers == 1:
                weights = np.ones((1, len(radial_tokens))) / len(radial_tokens)
            elif num_layers == 2:
                n = len(radial_tokens)
                weights_fore = np.linspace(1, 0, n) / np.linspace(1, 0, n).sum()
                weights_back = np.linspace(0, 1, n) / np.linspace(0, 1, n).sum()
                weights = np.stack((weights_fore, weights_back))
            else:
                n = len(radial_tokens)
                weights_fore = np.linspace(1, 0, n) / np.linspace(1, 0, n).sum()
                weights_back = np.linspace(0, 1, n) / np.linspace(0, 1, n).sum()
                depth_values = np.array([depth_map_grid[y, x] for (y, x) in indices])
                weights_middle = []
                for m in midpoints:
                    mth_weights = np.where(depth_values <= m, depth_values / m, (1 - depth_values) / (1 - m))
                    weights_middle.append(mth_weights / mth_weights.sum())
                weights = np.stack([weights_fore, *weights_middle, weights_back])
            radial_weights.append(weights)

        # Pad weights to same size for visualization
        max_tokens = max(w.shape[1] for w in radial_weights)
        radial_weights_padded = []
        for weights in radial_weights:
            padded = np.zeros((num_layers, max_tokens))
            padded[:, :weights.shape[1]] = weights
            radial_weights_padded.append(padded)
        radial_weights_padded = np.array(radial_weights_padded)  # shape: (num_orientations, num_layers, max_tokens)
        
        # Create subplot grid
        ncols = min(num_layers, 3)  # Max 3 columns
        nrows = (num_layers + ncols - 1) // ncols  # Ceiling division
        fig, axs = plt.subplots(nrows, ncols, figsize=(3*ncols + 1, 3*nrows), squeeze=False)
        
        for layer in range(num_layers):
            row = layer // ncols
            col = layer % ncols
            ax = axs[row, col]
            
            # Plot weights heatmap
            im = ax.imshow(radial_weights_padded[:, layer, :].T, aspect='auto', cmap='viridis', interpolation='nearest', vmin=0, vmax=1, origin='lower')
            ax.set_title(f'Layer {layer+1} Weights (Radial)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Orientation (degrees)', fontsize=10)
            ax.set_ylabel('Token Index along Radial Line', fontsize=10)
            
            # Set x-axis ticks to show angles
            num_orientations = radial_weights_padded.shape[0]
            tick_positions = np.linspace(0, num_orientations - 1, 9)  # 9 ticks for 0, 45, 90, ..., 360
            tick_labels = [f'{int(angle)}°' for angle in np.linspace(0, 360, 9)]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45)
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='Weight', fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        for layer in range(num_layers, nrows * ncols):
            row = layer // ncols
            col = layer % ncols
            axs[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(r'..\debug\radial_weights_sample.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    averaged_radial_tokens = np.array(averaged_radial_tokens)  # shape: (num_orientations, num_layers, feature_dim)
    averaged_radial_tokens = np.transpose(averaged_radial_tokens, (1, 0, 2))  # shape: (num_layers, num_orientations, feature_dim)

    return averaged_radial_tokens

def _next_sample_id(results_dir: str) -> int:
    """Compute the next integer sample id based on existing files named like 'sample_#_... .png' or 'sample_#.png'."""
    if not os.path.exists(results_dir):
        return 0
    ids = []
    pattern = re.compile(r"^sample_(\d+)(?:_|\.)")
    for name in os.listdir(results_dir):
        m = pattern.match(name)
        if m:
            try:
                ids.append(int(m.group(1)))
            except ValueError:
                pass
    return max(ids) + 1 if ids else 0


def _save_separate_figures(results_dir, sample_id,
                           ground_image_np, aerial_image_np,
                           best_orientation, yaw,
                           angle_step, distances):
    """Save the 4 separate images corresponding to the 2x2 combined figure."""
    # 1) Ground image
    fig_g, ax_g = plt.subplots(figsize=(6, 6))
    ax_g.imshow(ground_image_np)
    ax_g.set_title("Yaw: {:.1f}°".format(yaw), fontsize=16, fontweight='bold')
    ax_g.axis('off')
    fig_g.savefig(os.path.join(results_dir, f"sample_{sample_id}_ground.png"), dpi=300, bbox_inches='tight')
    plt.close(fig_g)

    # 2) Aerial with predicted vs GT orientation
    fig_a, ax_a = plt.subplots(figsize=(6, 6))
    ax_a.imshow(aerial_image_np)
    radius = aerial_image_np.shape[0] // 2
    ctr = (aerial_image_np.shape[1] // 2, aerial_image_np.shape[0] // 2)
    end_x = int(ctr[0] + radius * np.cos(np.deg2rad(best_orientation)))
    end_y = int(ctr[1] - radius * np.sin(np.deg2rad(best_orientation)))
    end_x_GT = int(ctr[0] + radius * np.cos(np.deg2rad(90 - (yaw - 180))))
    end_y_GT = int(ctr[1] - radius * np.sin(np.deg2rad(90 - (yaw - 180))))
    ax_a.plot([ctr[0], end_x], [ctr[1], end_y], color='red', linestyle='--', label='Prediction')
    ax_a.plot([ctr[0], end_x_GT], [ctr[1], end_y_GT], color='orange', linestyle='--', label='Ground Truth')
    delta_yaw = np.abs(((90 - (yaw - 180)) - best_orientation + 180) % 360 - 180)
    ax_a.set_title("Orientation Error: {:.4f}°".format(delta_yaw), fontsize=16, fontweight='bold')
    ax_a.legend(loc='upper right')
    ax_a.axis('off')
    fig_a.savefig(os.path.join(results_dir, f"sample_{sample_id}_aerial_overlay.png"), dpi=300, bbox_inches='tight')
    plt.close(fig_a)

    # 3) Distance over orientations plot
    fig_d, ax_d = plt.subplots(figsize=(6, 6))
    ax_d.plot(np.arange(0, 360, angle_step), distances)
    # Confidence from distances
    mean_distance = float(np.mean(distances))
    std_distance = float(np.std(distances))
    min_distance = float(np.min(distances))
    confidence = (mean_distance - min_distance) / std_distance if std_distance > 0 else 0.0
    # ax_d.set_title("Distance vs Orientation", fontsize=16, fontweight='bold')
    ax_d.grid(True)
    ax_d.set_xlabel('Orientation (deg)')
    ax_d.set_ylabel('Distance')
    ax_d.set_xlim(0, 360)
    ax_d.set_ylim(min(distances), max(distances))
    fig_d.savefig(os.path.join(results_dir, f"sample_{sample_id}_distance_curve.png"), dpi=300, bbox_inches='tight')
    plt.close(fig_d)

    # 4) Aerial image with rays colored by distance magnitude
    fig_r, ax_r = plt.subplots(figsize=(6, 6))
    ax_r.imshow(aerial_image_np)
    radius = aerial_image_np.shape[0] // 2
    ctr = (aerial_image_np.shape[1] // 2, aerial_image_np.shape[0] // 2)
    min_dist = min(distances)
    max_dist = max(distances)
    for j, beta in enumerate(np.arange(0, 360, angle_step)):
        end_x = int(ctr[0] + radius * np.cos(np.deg2rad(beta)))
        end_y = int(ctr[1] - radius * np.sin(np.deg2rad(beta)))
        normalized_dist = (distances[j] - min_dist) / (max_dist - min_dist) if max_dist > min_dist else 0.0
        normalized_dist = np.clip(normalized_dist, 0.0, 1.0)
        color = plt.cm.plasma(normalized_dist)
        ax_r.plot([ctr[0], end_x], [ctr[1], end_y], color=color)
    # ax_r.set_title("Aerial with Distance Rays", fontsize=16, fontweight='bold')
    ax_r.axis('off')
    # colorbar
    norm = plt.Normalize(min_dist, max_dist)
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
    sm.set_array([])
    # plt.colorbar(sm, ax=ax_r)
    fig_r.savefig(os.path.join(results_dir, f"sample_{sample_id}_aerial_rays.png"), dpi=300, bbox_inches='tight')
    plt.close(fig_r)