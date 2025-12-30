# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from zesco.dataset import PairedImagesDataset, sample_cvusa_images, get_transforms, denormalize
from zesco.model import CrossviewModel, CosineSimilarityLoss, get_processors
from zesco.inoutdoor import IndoorOutdoorClassifier
from zesco.depther import DepthAnything, get_radial_depth_map
from zesco.skyfilter import SkyFilter
from zesco.utils import get_averaged_vertical_tokens, get_averaged_radial_tokens, find_alignment

# %%
# Sample paired images
dataset_path = r"D:\cross_view_localization_DSM\Data\CVGlobal"
# dataset_path = r"D:\cross_view_localization_DSM\Data\CVUSA"
train_filenames, val_filenames = sample_cvusa_images(dataset_path, sample_percentage=0.2, split_ratio=0.8, groundtype='panos')

# Settings
image_size = 224
aerial_scaling = 2
provide_paths = False
BATCH_SIZE = 1
fov_x = 90                   # horizontal fov in degrees
num_layers = 5
crop_percentage = 0.15
horizontal_scaling = round(fov_x / 90.0) if fov_x >= 90 else 1.0

backbone = 'dinov3'

# Get the processor and transforms
processors = get_processors(backbone)
transform_ground, transform_aerial = get_transforms(processors, image_size, aerial_scaling, crop_percentage=crop_percentage, horizontal_scaling=horizontal_scaling)

# Instantiate the dataset and dataloader
paired_dataset = PairedImagesDataset(train_filenames, transform_aerial=transform_aerial, transform_ground=transform_ground, cutout_from_pano=True, fov_x=fov_x, debug=False)
data_loader = DataLoader(paired_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Define the Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load IndoorOutdoorClassifier
indoor_outdoor_classifier = IndoorOutdoorClassifier().to(device)

# Load the Model
model = CrossviewModel(backbone=backbone, frozen=True, fov=fov_x).to(device)
grid_size_ground = (image_size // model.patch_size, image_size // model.patch_size * horizontal_scaling)
grid_size_aerial = (image_size // model.patch_size, image_size // model.patch_size)
# grid_dim = grid_size[0] if grid_size[0] == grid_size[1] else ValueError("Grid size should be square.")
print(f"Model patch size: {model.patch_size}, grid size ground: {grid_size_ground}, grid size aerial: {grid_size_aerial}")
model.show()

# %%
n_runs = 100
for i in range(n_runs):

    visualizations_dir = r'..\utils\visualizations'

    # Create intermediate folder based on FOVs
    fov_dir = os.path.join(visualizations_dir, str(fov_x))

    # Ensure the base directory exists
    if not os.path.exists(fov_dir):
        os.makedirs(fov_dir)

    # Find the next available numbered subfolder
    existing_folders = [
        d for d in os.listdir(fov_dir)
        if os.path.isdir(os.path.join(fov_dir, d)) and d.isdigit()
    ]

    if existing_folders:
        last_number = max(int(d) for d in existing_folders)
        next_number = last_number + 1
    else:
        next_number = 0

    # Create the new subfolder
    visualizations_path = os.path.join(fov_dir, f"{next_number:04d}")
    os.makedirs(visualizations_path, exist_ok=True)

    print(f"Visualizations will be saved to: {visualizations_path}")

    # %%
    # Load a single pair of images
    ground_image, aerial_image, fov, yaw, pitch = next(iter(data_loader))
    ground_image = ground_image.to(device)
    aerial_image = aerial_image.to(device)

    print("fov:", (fov[0].item(), fov[1].item()))
    print("yaw:", yaw.item())
    print("pitch:", pitch.item())

    # Compute the output of the model
    ground_tokens, aerial_tokens = model(ground_image, aerial_image, debug=True)

    embed_dim = ground_tokens.shape[2]
    ground_tokens_grid = ground_tokens.view(BATCH_SIZE, embed_dim, grid_size_ground[0], grid_size_ground[1])
    aerial_tokens_grid = aerial_tokens.view(BATCH_SIZE, embed_dim, grid_size_aerial[0], grid_size_aerial[1])

    print("ground tokens shape:", ground_tokens.shape)
    print("aerial tokens shape:", aerial_tokens.shape)

    # Calculate the number of patches for ground and aerial images
    num_patches_ground = (ground_image.shape[-1] // model.patch_size) * (ground_image.shape[-2] // model.patch_size)
    num_patches_aerial = (aerial_image.shape[-1] // model.patch_size) * (aerial_image.shape[-2] // model.patch_size)
    print("num_patches_ground: ", num_patches_ground)
    print("num_patches_aerial: ", num_patches_aerial)

    ground_image_denorm = denormalize(ground_image.squeeze(), processors[0])
    aerial_image_denorm = denormalize(aerial_image.squeeze(), processors[1])

    # Convert images to numpy for visualization
    ground_image_np = ground_image_denorm.permute(1, 2, 0).detach().cpu().numpy()
    aerial_image_np = aerial_image_denorm.permute(1, 2, 0).detach().cpu().numpy()
    ground_image_vis = ground_image_np * 255
    aerial_image_vis = aerial_image_np * 255
    ground_image_vis = ground_image_vis.astype(np.uint8)
    aerial_image_vis = aerial_image_vis.astype(np.uint8)

    print("Ground image pixel values (min, max):", ground_image_np.min(), ground_image_np.max())
    print("Aerial image pixel values (min, max):", aerial_image_np.min(), aerial_image_np.max())

    # # Plot 2: Aerial Image with prediction and ground truth lines
    img_size = aerial_image_np.shape[0]
    Y, X = np.ogrid[:img_size, :img_size]
    center_y, center_x = img_size // 2, img_size // 2

    # Calculate angle for each pixel to match the convention: angle=0 is +x axis, angle increases counterclockwise
    # Using the same convention as get_direction_tokens: x = center_x + r*cos(angle), y = center_y - r*sin(angle)
    dx = X - center_x
    dy = center_y - Y  # Note: inverted because y increases downward in image coordinates
    angles_grid = np.arctan2(dy, dx) * 180 / np.pi  # arctan2(dy, dx) gives angle from +x axis
    angles_grid = angles_grid % 360  # Convert to [0, 360)

    radius = aerial_image_np.shape[0] // 2
    center = (aerial_image_np.shape[1] // 2, aerial_image_np.shape[0] // 2)
    heading = yaw.item() - 90
    end_x_GT = int(center[0] + radius * np.cos(np.deg2rad(heading)))
    end_y_GT = int(center[1] - radius * np.sin(np.deg2rad(heading)))

    # Use colors that stand out from coolwarm (blue-red) spectrum
    # fig, ax2 = plt.subplots(1, 1, figsize=(5, 5))
    # ax2.imshow(aerial_image_np)
    # line_gt = ax2.plot([center[0], end_x_GT], [center[1], end_y_GT], color='#ADFF2F', linestyle='--', label='Ground Truth', linewidth=3)
    # ax2.legend(loc='best', fontsize=14)
    # ax2.axis('off')
    # plt.show()

    # Check for Indoor or Outdoor Image
    with torch.no_grad():
        # Convert ground image to PIL Image for the classifier
        ground_image_pil = transforms.ToPILImage()(ground_image_denorm.cpu())
        ground_preds = indoor_outdoor_classifier(ground_image_pil)

    print(f"Outdoor Probability: {ground_preds['Outdoor']*100}%")

    # Helper function to save resized images
    def save_resized_image(path, img, target_size=(224, 224), cmap=None):
        if img.shape[:2] != target_size:
            # Use nearest neighbor for small feature maps to preserve grid look
            interpolation = cv2.INTER_NEAREST if img.shape[0] < target_size[0] else cv2.INTER_LINEAR
            img_resized = cv2.resize(img, target_size, interpolation=interpolation)
        else:
            img_resized = img
        
        plt.imsave(path, img_resized, cmap=cmap)

    save_resized_image(os.path.join(visualizations_path, "ground_image.png"), ground_image_vis)
    # plt.imshow(ground_image_vis)
    # plt.axis('off')
    # plt.show()

    save_resized_image(os.path.join(visualizations_path, "aerial_image.png"), aerial_image_vis)
    # plt.imshow(aerial_image_vis)
    # plt.axis('off')
    # plt.show()

    ground_vis_tokens = model.show_tokens(ground_tokens, grid_size=grid_size_ground, mode='save', results_path=os.path.join(visualizations_path, "ground_tokens.png"), return_tokens=True, target_size=(224, 224))
    aerial_vis_tokens = model.show_tokens(aerial_tokens, grid_size=grid_size_aerial, mode='save', results_path=os.path.join(visualizations_path, "aerial_tokens.png"), return_tokens=True, target_size=(224, 224))

    # %%
    # Visualization - Common Baselined Tokens Combined
    ground_features = ground_tokens.squeeze(0).detach().cpu().numpy()
    aerial_features = aerial_tokens.squeeze(0).detach().cpu().numpy()
    # ground_features_combined, aerial_features_combined = model.get_combined_embedding_visualization(ground_features, aerial_features, grid_size_ground, grid_size_aerial)

    # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # ax.imshow(ground_features_combined)
    # ax.axis('off')
    # plt.savefig(os.path.join(visualizations_path, "ground_features_combined.png"), bbox_inches='tight', pad_inches=0, dpi=300)
    # plt.show()

    # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # ax.imshow(aerial_features_combined)
    # ax.axis('off')
    # plt.savefig(os.path.join(visualizations_path, "aerial_features_combined.png"), bbox_inches='tight', pad_inches=0, dpi=300)
    # plt.show()

    # %% [markdown]
    # #### Apply the Sky Filter

    # %%
    # Initialize the sky filter
    sky_filter = SkyFilter(grid_size=grid_size_ground)

    # Apply the sky filter
    ground_image_no_sky, sky_mask, sky_grid = sky_filter(ground_image_vis, debug=False)

    # Visualize the original image, mask, and the sky-removed image
    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(18, 6))
    # ax1.imshow(ground_image_vis)
    # ax1.set_title("Original Image", fontsize=12, fontweight='bold')
    # ax1.axis('off')
        
    # ax2.imshow(sky_mask, cmap='gray')
    # ax2.set_title("Sky Mask", fontsize=12, fontweight='bold')
    # ax2.axis('off')

    # ax3.imshow(ground_image_no_sky)
    # ax3.set_title("Image Without Sky", fontsize=12, fontweight='bold')
    # ax3.axis('off')

    # ax4.imshow(sky_grid, cmap='gray')
    # ax4.set_title("Sky Grid Mask", fontsize=12, fontweight='bold')
    # ax4.axis('off')

    # plt.show()

    # %%
    print("ground_vis_tokens.shape: ", ground_vis_tokens.shape)
    print("sky_grid.shape: ", sky_grid.shape)
    print("ground_image_vis.shape: ", ground_image_vis.shape)
    print("model.patch_size: ", model.patch_size)

    # Recompute combined features with sky grid
    ground_features_combined, aerial_features_combined = model.get_combined_embedding_visualization(ground_features, aerial_features, grid_size_ground, grid_size_aerial, sky_grid=sky_grid)
    ground_features_combined_with_sky, aerial_features_combined_with_sky = model.get_combined_embedding_visualization(ground_features, aerial_features, grid_size_ground, grid_size_aerial)

    # Show Ground Tokens without Sky - replace sky (black) with white
    ground_tokens_no_sky = ground_vis_tokens * sky_grid.reshape(1, sky_grid.shape[0], sky_grid.shape[1], 1) + (1 - sky_grid).reshape(1, sky_grid.shape[0], sky_grid.shape[1], 1) * 255
    ground_tokens_combined_no_sky = ground_features_combined * sky_grid.reshape(sky_grid.shape[0], sky_grid.shape[1], 1) + (1 - sky_grid).reshape(sky_grid.shape[0], sky_grid.shape[1], 1) * np.array([1.0, 1.0, 1.0])

    # Create a figure in which the grid mask is applied to the original image - replace sky with white
    ground_image_tokenized_no_sky = ground_image_vis * sky_grid.reshape(sky_grid.shape[0], sky_grid.shape[1], 1).repeat(16, axis=0).repeat(16, axis=1) + (1 - sky_grid).reshape(sky_grid.shape[0], sky_grid.shape[1], 1).repeat(16, axis=0).repeat(16, axis=1) * 255
    print("ground_tokens_no_sky.shape: ", ground_tokens_no_sky.shape)

    save_resized_image(os.path.join(visualizations_path, "ground_image_no_sky.png"), ground_image_tokenized_no_sky.astype(np.uint8))
    # plt.imshow(ground_image_tokenized_no_sky.astype(np.uint8))
    # plt.axis('off')
    # plt.show()

    save_resized_image(os.path.join(visualizations_path, "ground_features_combined_with_sky.png"), ground_features_combined_with_sky)
    # plt.imshow(ground_features_combined_with_sky)
    # plt.axis('off')
    # plt.show()

    save_resized_image(os.path.join(visualizations_path, "aerial_features_combined_with_sky.png"), aerial_features_combined_with_sky)
    # plt.imshow(aerial_features_combined_with_sky)
    # plt.axis('off')
    # plt.show()

    save_resized_image(os.path.join(visualizations_path, "ground_tokens_combined_no_sky.png"), ground_tokens_combined_no_sky)
    # plt.imshow(ground_tokens_combined_no_sky)
    # plt.axis('off')
    # plt.show()

    save_resized_image(os.path.join(visualizations_path, "aerial_features_combined_no_sky.png"), aerial_features_combined)
    # plt.imshow(aerial_features_combined)
    # plt.axis('off')
    # plt.show()

    # %%
    # ground_features_combined, aerial_features_combined = model.get_combined_embedding_visualization(ground_features, aerial_features, grid_size_ground, grid_size_aerial, sky_grid=sky_grid)

    # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # ax.imshow(ground_features_combined)
    # ax.axis('off')
    # plt.savefig(os.path.join(visualizations_path, "ground_features_combined.png"), bbox_inches='tight', pad_inches=0, dpi=300)
    # plt.show()

    # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # ax.imshow(aerial_features_combined)
    # ax.axis('off')
    # plt.savefig(os.path.join(visualizations_path, "aerial_features_combined.png"), bbox_inches='tight', pad_inches=0, dpi=300)
    # plt.show()

    # %%
    # Compute Depth Maps
    depth_anything = DepthAnything(grid_size=grid_size_ground)

    depth_map_ground, depth_map_grid_ground = depth_anything(ground_image_no_sky, debug=False)
    depth_map_grid_aerial = get_radial_depth_map(grid_size_aerial)

    save_resized_image(os.path.join(visualizations_path, "depth_map.png"), depth_map_ground, cmap='plasma')
    # plt.imshow(depth_map_ground, cmap='plasma')
    # plt.axis('off')
    # plt.show()

    save_resized_image(os.path.join(visualizations_path, "depth_map_ground.png"), depth_map_grid_ground, cmap='plasma')
    # plt.imshow(depth_map_grid_ground, cmap='plasma')
    # plt.axis('off')
    # plt.show()

    save_resized_image(os.path.join(visualizations_path, "depth_map_aerial.png"), depth_map_grid_aerial, cmap='plasma')
    # plt.imshow(depth_map_grid_aerial, cmap='plasma')
    # plt.axis('off')
    # plt.show()

    # %%
    # Define Angle Step
    angle_step = fov_x / grid_size_ground[1]    # should be always 6.4285 degrees for 224x224
    print("angle_step: ", angle_step)

    loss = CosineSimilarityLoss()

    # ground_features = ground_features_combined.reshape(grid_size_ground[0], grid_size_ground[1], -1)
    # aerial_features = aerial_features_combined.reshape(grid_size_aerial[0], grid_size_aerial[1], -1)
    ground_features_combined_flat = ground_features_combined.reshape(-1, ground_features_combined.shape[2])
    aerial_features_combined_flat = aerial_features_combined.reshape(-1, aerial_features_combined.shape[2])

    # Compute Averaged Tokens using the weight vector, excluding sky tokens
    vertical_averaged_tokens = get_averaged_vertical_tokens(
        angle_step=angle_step,
        image_tokens=ground_features,
        # image_tokens=ground_features_combined_flat,
        grid_size=grid_size_ground,
        sky_grid=sky_grid,
        depth_map_grid=depth_map_grid_ground,
        num_layers=num_layers,
        debug=False
    )
    radial_averaged_tokens = get_averaged_radial_tokens(
        angle_step=angle_step,
        image_tokens=aerial_features,
        # image_tokens=aerial_features_combined_flat,
        grid_size=grid_size_aerial,     # use height for square aerial grid
        sky_grid=sky_grid,
        depth_map_grid=depth_map_grid_aerial,
        num_layers=num_layers,
        debug=False
    )
    print("averaged vertical tokens: ", vertical_averaged_tokens.shape)
    print("averaged radial tokens: ", radial_averaged_tokens.shape)

    # Find the best alignment - pass grid_size_ground tuple (will use width for alignment)
    best_orientation, distances, min_distance, confidence = find_alignment(loss, vertical_averaged_tokens, radial_averaged_tokens, grid_size_ground, fov_x, debug=False)
    # delta_yaw = np.abs(((90 - (yaw.item() - 180)) - best_orientation + 180) % 360 - 180)
    # delta_yaw = np.abs(best_orientation - heading)
    delta_yaw = np.abs(((best_orientation - heading) + 180) % 360 - 180)
    delta_yaw_tau = np.abs((delta_yaw + 90) % 180 - 90)
    if delta_yaw < 0:
        delta_yaw += 180

    print("Delta yaw: ", delta_yaw)
    print("Delta Yaw Tau: ", delta_yaw_tau)
    print(f"Confidence: {confidence:.4f}")

    num_steps = int(round(360 / angle_step))
    angles = np.linspace(0, 360 - angle_step, num_steps)
    print("len(angles): ", len(angles))
    print("len(distances): ", len(distances))

    # Interpolate for smoother curves
    from scipy.interpolate import interp1d
    angles_fine = np.linspace(0, 360, 1000)
    interpolator = interp1d(np.append(angles, 360), np.append(distances, distances[0]), kind='cubic')
    distances_fine = interpolator(angles_fine)

    # Use coolwarm colormap for better contrast (reversed so blue = low distance = good match)
    cmap = plt.cm.coolwarm_r

    # # Plot 2: Aerial Image with prediction and ground truth lines
    img_size = aerial_image_np.shape[0]
    Y, X = np.ogrid[:img_size, :img_size]
    center_y, center_x = img_size // 2, img_size // 2
    min_dist = min(distances)
    max_dist = max(distances)

    # Calculate angle for each pixel to match the convention: angle=0 is +x axis, angle increases counterclockwise
    # Using the same convention as get_direction_tokens: x = center_x + r*cos(angle), y = center_y - r*sin(angle)
    dx = X - center_x
    dy = center_y - Y  # Note: inverted because y increases downward in image coordinates
    angles_grid = np.arctan2(dy, dx) * 180 / np.pi  # arctan2(dy, dx) gives angle from +x axis
    angles_grid = angles_grid % 360  # Convert to [0, 360)

    radius = aerial_image_np.shape[0] // 2
    center = (aerial_image_np.shape[1] // 2, aerial_image_np.shape[0] // 2)
    end_x = int(center[0] + radius * np.cos(np.deg2rad(best_orientation)))
    end_y = int(center[1] - radius * np.sin(np.deg2rad(best_orientation)))
    end_x_GT = int(center[0] + radius * np.cos(np.deg2rad(heading)))
    end_y_GT = int(center[1] - radius * np.sin(np.deg2rad(heading)))

    # Use interpolation to get smooth distance values
    interpolator_heatmap = interp1d(np.append(angles, 360), np.append(distances, distances[0]), 
                                    kind='cubic', fill_value='extrapolate')

    # Map each pixel to its corresponding distance value using interpolation
    heatmap = np.zeros((img_size, img_size))
    for i in range(img_size):
        for j in range(img_size):
            pixel_angle = angles_grid[i, j]
            heatmap[i, j] = interpolator_heatmap(pixel_angle)

    # Normalize heatmap (now blue=low distance=good match, red=high distance=bad match)
    heatmap_normalized = (heatmap - min_dist) / (max_dist - min_dist)

    # Plot 4: Aerial image with heatmap AND prediction/ground truth lines
    # Use colors that stand out from coolwarm (blue-red) spectrum
    fig = plt.figure(figsize=(5, 5))
    ax2 = plt.Axes(fig, [0., 0., 1., 1.])
    ax2.set_axis_off()
    fig.add_axes(ax2)
    ax2.imshow(aerial_image_np)

    # Overlay the heatmap
    ax2.imshow(heatmap_normalized, cmap=cmap, alpha=0.7, interpolation='bilinear')
    # Add prediction and ground truth lines with colors distinct from coolwarm
    # Cyan for prediction and yellow-green for ground truth
    line_pred = ax2.plot([center[0], end_x], [center[1], end_y], 
                     color='#00FFFF', linestyle='--', label='Prediction', linewidth=5)

    line_gt = ax2.plot([center[0], end_x_GT], [center[1], end_y_GT], 
                   color='#FFD700', linestyle='--', label='Ground Truth', linewidth=5)
    ax2.legend(loc='best', fontsize=30)
    plt.savefig(os.path.join(visualizations_path, "plot_aerial_heatmap_with_lines.png"), dpi=300)
    # plt.show()
    plt.close(fig)

    # Plot 3: Distance curve with color-coded points
    fig, ax3 = plt.subplots(1, 1, figsize=(8, 5))

    # Create smooth colored line
    colors_fine = [cmap((d - min_dist) / (max_dist - min_dist)) for d in distances_fine]
    for i in range(len(angles_fine) - 1):
        ax3.plot(angles_fine[i:i+2], distances_fine[i:i+2], color=colors_fine[i], linewidth=2)

    # set the ticks values font size
    ax3.tick_params(axis='x', labelsize=26)
    ax3.tick_params(axis='y', labelsize=26)
    ax3.grid(True, alpha=0.6)
    ax3.set_xlabel('$Orientation \, [deg]$', fontsize=30)
    ax3.set_ylabel('$Distance$', fontsize=30)
    ax3.set_xlim(0, 360)
    # Add some padding to y-axis to ensure curve fits
    y_range = max_dist - min_dist
    ax3.set_ylim(min_dist - 0.05 * y_range, max_dist + 0.05 * y_range)
    plt.tight_layout()
    plt.savefig(os.path.join(visualizations_path, "plot_distance_curve.png"), dpi=300, bbox_inches='tight', pad_inches=0)
    # plt.show()
    plt.close(fig)

    # Determine the next available file number for summary
    file_count = len([name for name in os.listdir(visualizations_path) if name.startswith("summary") and name.endswith(".png")])
    file_path = os.path.join(visualizations_path, f"summary_{file_count}.png")

    print(f"Plots saved to {visualizations_path}")
