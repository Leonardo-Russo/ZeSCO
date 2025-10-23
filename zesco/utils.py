import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import re
        
def find_alignment(loss, fore_vert_avg_tokens, midd_vert_avg_tokens, back_vert_avg_tokens, fore_rad_avg_tokens, midd_rad_avg_tokens, back_rad_avg_tokens, grid_size, angle_step, debug=False):
    """
    Finds the alignment between averaged vertical tokens and averaged radial tokens using PyTorch.
    
    Parameters:
    - fore/midd/back_vert_avg_tokens: torch.Tensors of shape (batch_size, grid_size, feature_dim) or (grid_size, feature_dim)
    - fore/midd/back_rad_avg_tokens: torch.Tensors of shape (batch_size, num_angles, feature_dim) or (num_angles, feature_dim)
    - grid_size (int): The size of the grid
    - image_span (float or torch.Tensor): The span of the image (can be batched)
    
    Returns:
    - best_orientations: torch.Tensor of shape (batch_size,) or float
    - distances: list of lists or list
    - min_distances: torch.Tensor of shape (batch_size,) or float
    - confidences: torch.Tensor of shape (batch_size,) or float
    """
    device = fore_vert_avg_tokens.device
    batch_size = fore_vert_avg_tokens.shape[0]

    # Compute angles using linspace to avoid floating point precision issues
    num_steps = int(round(360 / angle_step))
    angles = torch.linspace(0, 360 - angle_step, num_steps, device=device)
    num_angles = len(angles)
    
    # Initialize batch outputs
    best_orientations = torch.zeros(batch_size, device=device)
    all_distances = []
    min_distances = torch.zeros(batch_size, device=device)
    confidences = torch.zeros(batch_size, device=device)
    
    # Process each sample in the batch
    for b in range(batch_size):
        distances = []
        
        for j, angle in enumerate(angles):
            beta = angle.item()
            cone_distance = 0
            
            for k in range(grid_size):
                # Get radial tokens with modulo indexing
                rad_idx = int(j + k - grid_size/2) % fore_rad_avg_tokens.shape[1]
                vert_idx = (grid_size - 1) - k
                
                # Stack tokens for this batch element
                vert_avg_tokens = torch.stack([
                    fore_vert_avg_tokens[b, vert_idx],
                    midd_vert_avg_tokens[b, vert_idx],
                    back_vert_avg_tokens[b, vert_idx]
                ])
                
                rad_avg_tokens = torch.stack([
                    fore_rad_avg_tokens[b, rad_idx],
                    midd_rad_avg_tokens[b, rad_idx],
                    back_rad_avg_tokens[b, rad_idx]
                ])
                
                # Calculate distance using loss function
                dist = loss(vert_avg_tokens.cpu().numpy(), rad_avg_tokens.cpu().numpy())
                cone_distance += dist
            
            cone_distance /= grid_size
            distances.append(cone_distance)
        
        # Find minimum distance and best orientation for this sample
        distances_tensor = torch.tensor(distances, device=device)
        min_distances[b] = distances_tensor.min()
        best_idx = int(distances_tensor.argmin())
        best_orientations[b] = best_idx * angle_step
        
        # Compute confidence
        mean_distance = float(distances_tensor.mean())
        std_distance = float(distances_tensor.std())
        confidences[b] = (mean_distance - min_distances[b].item()) / std_distance if std_distance > 0 else 0.0
        
        all_distances.append(distances)
        
        if debug:
            print(f"Batch {b}: Min Distance: {min_distances[b]:.4f} \tBest Orientation: {best_orientations[b]:.1f}°")
    
    return best_orientations, all_distances, min_distances, confidences

def get_averaged_vertical_tokens(angle_step, image_tokens_grid, grid_size, sky_grid, depth_map_grid, threshold=0.5):
    """
    Compute averaged vertical tokens using PyTorch for GPU acceleration (fully vectorized).
    
    Parameters:
    - image_tokens: torch.Tensor of shape (batch_size, grid_size, grid_size, feature_dim)
    - sky_grid: torch.Tensor of shape (batch_size, 1, grid_size, grid_size)
    - depth_map_grid: torch.Tensor of shape (batch_size, 1, grid_size, grid_size)
    
    Returns:
    - Three tensors of averaged tokens (foreground, middleground, background) each of shape (grid_size, feature_dim)
    """

    # Extract vertical slices: vertical_tokens[batch, i] will be the i-th column (as tokens_grid[batch, :, i, :])
    vertical_tokens = image_tokens_grid.permute(0, 2, 1, 3)  # (batch_size, grid_size_cols, grid_size_rows, feature_dim)

    # Get sky and depth masks for each vertical column
    vertical_sky_grid = sky_grid.permute(0, 3, 2, 1)    # (batch_size, grid_size_cols, grid_size_rows, 1)
    vertical_depth_map_grid = depth_map_grid.permute(0, 3, 2, 1)  # (batch_size, grid_size_cols, grid_size_rows, 1)

    # Compute foreground weights
    foreground_weights = vertical_depth_map_grid * vertical_sky_grid  # (batch_size, grid_size_cols, grid_size_rows, 1)
    foreground_weights_sum = foreground_weights.sum(dim=2, keepdim=True).clamp(min=1e-8)  # (batch_size, grid_size_cols, 1, 1)
    foreground_weights_norm = foreground_weights / foreground_weights_sum  # (batch_size, grid_size_cols, grid_size_rows, 1)
    
    # Compute middleground weights
    middleground_weights = torch.where(
        vertical_depth_map_grid <= 0.5,
        (1 / threshold) * vertical_depth_map_grid,
        (1 - vertical_depth_map_grid) / vertical_depth_map_grid
    ) * vertical_sky_grid
    middleground_weights_sum = middleground_weights.sum(dim=2, keepdim=True).clamp(min=1e-8)
    middleground_weights_norm = middleground_weights / middleground_weights_sum
    
    # Compute background weights
    background_weights = (1 - vertical_depth_map_grid) * vertical_sky_grid
    background_weights_sum = background_weights.sum(dim=2, keepdim=True).clamp(min=1e-8)
    background_weights_norm = background_weights / background_weights_sum
    
    # Compute weighted averages using einsum for efficiency
    # Shape: (batch_size, grid_size_cols, grid_size_rows, 1) @ (batch_size, grid_size_cols, grid_size_rows, feature_dim)
    #     -> (batch_size, grid_size_cols, feature_dim)
    # Then sum over grid_size_rows dimension
    foreground_avg = torch.einsum('bijk,bijk->bik', foreground_weights_norm, vertical_tokens).squeeze(0)
    middleground_avg = torch.einsum('bijk,bijk->bik', middleground_weights_norm, vertical_tokens).squeeze(0)
    background_avg = torch.einsum('bijk,bijk->bik', background_weights_norm, vertical_tokens).squeeze(0)
    
    return foreground_avg, middleground_avg, background_avg

def get_averaged_radial_tokens(angle_step, image_tokens_grid, grid_size, sky_grid, depth_map_grid):
    """
    Compute averaged radial tokens using PyTorch for GPU acceleration.
    
    Parameters:
    - image_tokens_grid: torch.Tensor of shape (batch_size, grid_size, grid_size, feature_dim)
    - sky_grid: torch.Tensor of shape (batch_size, 1, grid_size, grid_size)
    - depth_map_grid: torch.Tensor of shape (batch_size, 1, grid_size, grid_size)
    - angle_step: float, the angle step in degrees
    
    Returns:
    - Three tensors of averaged radial tokens (foreground, middleground, background)
      each of shape (batch_size, num_angles, feature_dim)
    """
    device = image_tokens_grid.device
    batch_size, _, _, feature_dim = image_tokens_grid.shape
    
    # Remove channel dimension from sky and depth grids
    sky_grid = sky_grid.squeeze(1)  # (batch_size, grid_size, grid_size)
    depth_map_grid = depth_map_grid.squeeze(1)  # (batch_size, grid_size, grid_size)

    # Compute angles using linspace to avoid floating point precision issues
    num_steps = int(round(360 / angle_step))
    angles = torch.linspace(0, 360 - angle_step, num_steps, device=device)
    num_angles = len(angles)
    
    # Pre-allocate output tensors for the batch
    averaged_fore_radial_tokens = torch.zeros(batch_size, num_angles, feature_dim, device=device)
    averaged_middle_radial_tokens = torch.zeros(batch_size, num_angles, feature_dim, device=device)
    averaged_back_radial_tokens = torch.zeros(batch_size, num_angles, feature_dim, device=device)
    
    # Precompute radial sampling indices for all angles
    center = (grid_size // 2, grid_size // 2)
    angles_rad = torch.deg2rad(angles)
    
    # Process each batch element
    for b in range(batch_size):
        # Extract current batch's tensors
        batch_tokens = image_tokens_grid[b]  # (grid_size, grid_size, feature_dim)
        batch_sky = sky_grid[b]  # (grid_size, grid_size)
        batch_depth = depth_map_grid.squeeze(0)  # (grid_size, grid_size)
        
        # For each angle, compute the radial path from center outward
        for idx, angle_rad in enumerate(angles_rad):
            # Generate radial coordinates
            radii = torch.arange(grid_size, device=device, dtype=torch.float32)
            x_coords = center[0] + radii * torch.cos(angle_rad)
            y_coords = center[1] - radii * torch.sin(angle_rad)
            
            # Round to nearest integer and clamp to valid range
            x_int = torch.clamp(torch.round(x_coords).long(), 0, grid_size - 1)
            y_int = torch.clamp(torch.round(y_coords).long(), 0, grid_size - 1)

            # Retrieve radial tokens, sky, and depth values
            radial_tokens = batch_tokens[y_int, x_int]  # (valid_mask.shape[0], feature_dim)
            radial_sky = batch_sky[y_int, x_int]  # (valid_mask.shape[0])
            radial_depth = batch_depth[y_int, x_int]  # (valid_mask.shape[0])
            
            # Check which indices are within bounds (before rounding)
            inbounds_mask = ((x_coords >= 0) & (x_coords < grid_size) & (y_coords >= 0) & (y_coords < grid_size)).float()

            # Compute foreground weights
            foreground_weights = radial_depth * radial_sky * inbounds_mask
            num_tokens = int(foreground_weights.sum().item())
            
            # Middleground: prioritize middle distances, excluding sky
            if num_tokens > 0:
                t = torch.linspace(0, 1, grid_size, device=device)
                # Compute the center as the midpoint of the inbounds_mask indices
                mid_center = ((inbounds_mask.nonzero(as_tuple=True)[0].min() + inbounds_mask.nonzero(as_tuple=True)[0].max()) / 2) / grid_size
                sigma = 0.25  # controls spread of the bell shape (tweakable)
                middle_weights_pattern = torch.exp(-0.5 * ((t - mid_center) / sigma) ** 2)
                middleground_weights = middle_weights_pattern * radial_sky * inbounds_mask
            else:
                middleground_weights = torch.zeros_like(foreground_weights, device=device)
            
            # Background: prioritize far (low depth), excluding sky
            background_weights = (1 - radial_depth) * radial_sky * inbounds_mask
            
            # Normalize weights
            foreground_weights = foreground_weights / (foreground_weights.sum() + 1e-8)
            middleground_weights = middleground_weights / (middleground_weights.sum() + 1e-8)
            background_weights = background_weights / (background_weights.sum() + 1e-8)
            
            # Compute weighted averages
            averaged_fore_radial_tokens[b, idx] = torch.mv(radial_tokens.t(), foreground_weights)
            averaged_middle_radial_tokens[b, idx] = torch.mv(radial_tokens.t(), middleground_weights)
            averaged_back_radial_tokens[b, idx] = torch.mv(radial_tokens.t(), background_weights)

    return averaged_fore_radial_tokens, averaged_middle_radial_tokens, averaged_back_radial_tokens

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