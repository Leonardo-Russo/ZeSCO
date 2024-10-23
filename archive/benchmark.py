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
import math

from dataset import sample_paired_images

import matplotlib
matplotlib.use('TkAgg')  # or 'Agg' for non-GUI


def polar_transform(image, target_size):
    """
    Applies a polar transformation to the aerial image to match the dimensions of the ground image.

    Args:
    - image: The input aerial image (PIL Image).
    - target_size: The desired output size (height, width) after the polar transformation.

    Returns:
    - transformed_image: The polar-transformed aerial image (PIL Image).
    """

    # Convert PIL Image to NumPy array
    image_np = np.array(image)

    # Get the original image size and the target size
    Sa = image_np.shape[0]  # Assuming square aerial image
    Hg, Wg = target_size

    # Create the polar transformed image
    transformed_image_np = np.zeros((Hg, Wg, 3), dtype=np.uint8)  # 3 channels for RGB

    for i in range(Hg):
        for j in range(Wg):
            # Calculate the corresponding coordinates in the original aerial image
            xa = int(Sa / 2 - (Sa / 2) * ((Hg - i) / Hg) * np.cos(2 * np.pi * j / Wg))
            ya = int(Sa / 2 + (Sa / 2) * ((Hg - i) / Hg) * np.sin(2 * np.pi * j / Wg))

            # Ensure coordinates are within bounds
            xa = max(0, min(xa, Sa - 1))
            ya = max(0, min(ya, Sa - 1))

            # Copy the pixel value from the original to the transformed image
            transformed_image_np[i, j] = image_np[ya, xa]

    # Convert NumPy array back to PIL Image
    transformed_image = Image.fromarray(transformed_image_np)

    return transformed_image


class Paired360ImagesDataset(Dataset):
    def __init__(self, filenames, transform_aerial=None, transform_ground=None):
        self.filenames = filenames
        self.transform_aerial = transform_aerial
        self.transform_ground = transform_ground

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx, debug=False):
        ground_img_path, aerial_img_path = self.filenames[idx]

        ground_image = Image.open(ground_img_path).convert('RGB')
        aerial_image = Image.open(aerial_img_path).convert('RGB')

        ground_image_width, ground_image_height = ground_image.size
        if debug:
            print(f"Ground Image Size: {ground_image_width} x {ground_image_height}")

        if self.transform_ground:
            ground_image = self.transform_ground(ground_image)

        if self.transform_aerial:
            to_tensor = transforms.ToTensor()
            aerial_image = self.transform_aerial(aerial_image)
            aerial_image = polar_transform(aerial_image, (ground_image_height, ground_image_width))
            aerial_image = to_tensor(aerial_image)

        return ground_image, aerial_image


def compute_misalignment(ground_image, aerial_image, debug=False):
    """
    Calculates the misalignment between a ground image and an aerial image using ORB feature matching and homography estimation.
    
    Parameters:
    - ground_image: The ground image (tensor)
    - aerial_image: The aerial image (tensor)
    - debug: If True, intermediate images and keypoints will be displayed

    Returns:
    - misalignment_angle: The estimated misalignment angle in degrees
    """

    # Convert tensors to numpy arrays and then to grayscale
    img1_color = (ground_image.squeeze().permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    img2_color = (aerial_image.squeeze().permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
    height, width = img2.shape

    # Create ORB detector  
    orb_detector = cv2.ORB_create(5000)

    # Find keypoints and descriptors
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)  


    if debug:
        # Display keypoints on the images 
        img1_keypoints = cv2.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)
        img2_keypoints = cv2.drawKeypoints(img2, kp2, None, color=(0,255,0), flags=0)
        cv2.imshow('Ground Image Keypoints', img1_keypoints)
        cv2.imshow('Aerial Image Keypoints', img2_keypoints)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Match features
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(d1, d2)

    # Convert matches to a list
    matches = list(matches) 
    
    if debug:
        print(matches)
        print("matches len: ", len(matches))

    matches.sort(key=lambda x: x.distance)
    matches = matches[:int(len(matches) * 0.9)]

    # Extract matched keypoints
    p1 = np.zeros((len(matches), 2))
    p2 = np.zeros((len(matches), 2))
    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt  


    # Find homography
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

    # Check if homography was found
    if homography is None:
        print("Error: Homography not found")
    else:
        # Handle 2D or 3D cases
        if homography.shape == (3, 3):
            u, _, vh = np.linalg.svd(homography)
            R = u @ vh
        else:  # Assuming 2D
            u, _, vh = np.linalg.svd(homography[0:2, 0:2])
            R = u @ vh

        misalignment_angle = np.rad2deg(math.atan2(R[1,0], R[0,0]))


    if debug:
        # Warp the ground image using the homography for visualization
        transformed_img = cv2.warpPerspective(img1_color, homography, (width, height))
        cv2.imshow('Transformed Ground Image', transformed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return misalignment_angle


def benchmark(data_loader, device, savepath='results', debug=False):

    # Create results directory if it doesn't exist
    results_dir = savepath
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    delta_yaws = []
    delta_yaws_combined = []

    for batch_idx, (ground_images, aerial_images) in tqdm(enumerate(data_loader), total=len(data_loader), desc="Processing Batches"):
        ground_images = ground_images.to(device)
        aerial_images = aerial_images.to(device)

        n_batches = ground_images.size(0)

        # Assuming you want to handle each image in the batch individually
        for i in range(n_batches):  # Iterate over batch size
            ground_image = ground_images[i:i+1]
            aerial_image = aerial_images[i:i+1]
            misalignment = compute_misalignment(ground_image, aerial_image)
            delta_yaws.append(misalignment)
            if debug:
                print(f"Misalignment for image pair {i}: {misalignment} degrees")

            # Convert images to numpy for visualization
            ground_image_np = ground_image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            aerial_image_np = aerial_image.squeeze().permute(1, 2, 0).detach().cpu().numpy()

            ground_image_vis = ground_image.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255
            aerial_image_vis = aerial_image.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255
            ground_image_vis = ground_image_vis.astype(np.uint8)
            aerial_image_vis = aerial_image_vis.astype(np.uint8)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

            ax1.imshow(ground_image_np)
            ax1.set_title("Ground Image")
            ax1.axis('off')

            ax2.imshow(aerial_image_np)
            ax2.set_title("Aerial Image : Delta = {:.4f}°".format(misalignment))
            ax2.axis('off')

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

    transform_ground = transforms.ToTensor()

    transform_aerial = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop((image_size, image_size))
    ])

    # Instantiate the dataset and dataloader
    paired_dataset = Paired360ImagesDataset(train_filenames, transform_aerial=transform_aerial, transform_ground=transform_ground)
    data_loader = DataLoader(paired_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Define the Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    benchmark(data_loader, device, savepath=args.save_path, debug=args.debug.lower() == 'true')
