import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


class DepthAnything(nn.Module):

    def __init__(self, grid_size: tuple = (16, 16)):
        super(DepthAnything, self).__init__()
        self.image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf", use_fast=True)
        self.model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")
        self.grid_size = grid_size

    def forward(self, images, debug=False):
        """
        Applies depth estimation to a batch of images, and returns the depth maps along with
        downsampled versions of the depth maps on a grid where each grid cell
        contains the average depth value of the pixels in that cell.
        
        Parameters:
        - images: Batch of images (B, H, W, C) or list of images.
        - debug: Enable visualization of intermediate steps (default is False).

        Returns:
        - depth_maps: The estimated depth maps. Shape: (B, H, W)
        - depth_map_grids: The downsampled depth map grids. Shape: (B, grid_h, grid_w)
        """
        # Prepare images for the model
        inputs = self.image_processor(images=images, return_tensors="pt")

        # Get the predicted depth
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate to the original image size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=images.shape[1:3] if isinstance(images, np.ndarray) else images[0].shape[:2],
            mode="bicubic",
            align_corners=False,
        )

        # Extract non-normalized depth maps
        depth_maps = prediction  # (B, 1, H, W)

        # Normalize each depth map to the range [0, 1]
        depth_min = depth_maps.view(depth_maps.shape[0], -1).min(dim=1, keepdim=True)[0].unsqueeze(-1)
        depth_max = depth_maps.view(depth_maps.shape[0], -1).max(dim=1, keepdim=True)[0].unsqueeze(-1)
        depth_maps = (depth_maps.squeeze(1) - depth_min) / (depth_max - depth_min)
        depth_maps = torch.clamp(depth_maps, 0.0, 1.0).unsqueeze(1)

        # Use adaptive average pooling to create the downsampled depth map grids
        depth_map_grids = torch.nn.functional.adaptive_avg_pool2d(depth_maps, output_size=self.grid_size)
        depth_map_grids = torch.clamp(depth_map_grids, 0.0, 1.0)    # ensure values are in [0, 1] range

        # Visualize the depth map and downsampled depth map grid if in debug mode (only first image)
        if debug:
            depth_maps_np = depth_maps.cpu().numpy()
            depth_map_grids_np = depth_map_grids.cpu().numpy()
            images_np = images if isinstance(images, np.ndarray) else np.array(images)
            
            plt.figure(figsize=(18, 8))
            plt.subplot(131)
            plt.imshow(images_np[0])
            plt.title('Original Image (First in Batch)')
            plt.axis('off')
            plt.subplot(132)
            plt.imshow(depth_maps_np[0], cmap='plasma')
            plt.colorbar()
            plt.title('Depth Map (First in Batch)')
            plt.axis('off')
            plt.subplot(133)
            plt.imshow(depth_map_grids_np[0], cmap='plasma')
            plt.colorbar()
            plt.title(f'Downsampled Depth Map ({self.grid_size[0]}x{self.grid_size[1]} Grid)')
            plt.axis('off')
            plt.show()

        return depth_maps, depth_map_grids