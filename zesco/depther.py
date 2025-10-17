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

    def forward(self, image, debug=False):
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
        inputs = self.image_processor(images=image, return_tensors="pt")

        # Dimensions of the image
        height, width = image.shape[:2]

        # Get the predicted depth
        with torch.no_grad():
            outputs = self.model(**inputs)
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
        
        # Ensure values are exactly within [0, 1] range
        depth_map = np.clip(depth_map, 0.0, 1.0)

        # Calculate the size of each grid cell
        cell_height = height // self.grid_size[0]
        cell_width = width // self.grid_size[1]

        # Create the downsampled depth map grid
        depth_map_grid = np.zeros((self.grid_size[0], self.grid_size[1]), dtype=np.float32)
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                start_x = j * cell_width
                start_y = i * cell_height
                end_x = (j + 1) * cell_width if j < self.grid_size[1] - 1 else width
                end_y = (i + 1) * cell_height if i < self.grid_size[0] - 1 else height

                # Calculate the average depth value in the cell
                cell_depth = depth_map[start_y:end_y, start_x:end_x]
                # Ensure the mean value is also properly clipped
                depth_map_grid[i, j] = np.clip(np.mean(cell_depth), 0.0, 1.0)

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