import numpy as np
import matplotlib.pyplot as plt
import os
import re


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
            x = round(center[0] + r * np.cos(np.deg2rad(angle)))
            y = round(center[1] - r * np.sin(np.deg2rad(angle)))
            if 0 <= x < grid_size and 0 <= y < grid_size:
                idx = y * grid_size + x
                if tokens is None:
                    direction_tokens.append(None)
                    indices.append((y, x))
                else:
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
        
def find_alignment(loss, fore_vert_avg_tokens, midd_vert_avg_tokens, back_vert_avg_tokens, fore_rad_avg_tokens, midd_rad_avg_tokens, back_rad_avg_tokens, grid_size, image_span, debug=False):
    """
    Finds the alignment between averaged vertical tokens and averaged radial tokens.
    Parameters:
    - averaged_vertical_tokens (ndarray): A numpy array containing the averaged vertical tokens.
    - averaged_radial_tokens (ndarray): A numpy array containing the averaged radial tokens.
    - grid_size (int): The size of the grid.
    - image_span (float): The span of the image.
    Returns:
    - best_orientation (float): The best orientation in degrees.
    - distances (list): A list of distances for each orienta'ption.
    - min_distance (float): The minimum distance.
    - confidence (float): The confidence score.
    """

    angle_step = image_span / grid_size
    min_distance = float('inf')
    distances = []

    num_steps = int(round(360 / angle_step.item()))
    for j, beta in enumerate(np.linspace(0, 360 - angle_step, num_steps)):
        cone_distance = 0
        for i in range(grid_size+1):

            fore_rad_avg_token = fore_rad_avg_tokens[int(j + i - grid_size/2) % fore_rad_avg_tokens.shape[0]]
            midd_rad_avg_token = midd_rad_avg_tokens[int(j + i - grid_size/2) % midd_rad_avg_tokens.shape[0]]
            back_rad_avg_token = back_rad_avg_tokens[int(j + i - grid_size/2) % back_rad_avg_tokens.shape[0]]
            # print(f"beta: {beta:.2f} \tangle: {(j + i - grid_size/2)*angle_step} \tindex: {int(j + i - grid_size/2) % averaged_radial_tokens.shape[0]}")       

            vert_avg_tokens = np.vstack((fore_vert_avg_tokens[(grid_size-1)-i], midd_vert_avg_tokens[(grid_size-1)-i], back_vert_avg_tokens[(grid_size-1)-i]))            
            rad_avg_tokens = np.vstack((fore_rad_avg_token, midd_rad_avg_token, back_rad_avg_token))

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