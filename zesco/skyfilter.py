
import time
import numpy as np
import cv2
import os
import urllib.request 
import zipfile
import onnx
import onnxruntime as ort

import glob
import unicodedata
import re

import matplotlib.pyplot as plt

import torch.nn as nn


# This version should match the tag in the repository
version = "v1.0.6"
default_model_url = "https://github.com/OpenDroneMap/SkyRemoval/releases/download/%s/model.zip" % version 
default_model_folder = "model"
url_file = os.path.join(default_model_folder, 'url.txt')
guided_filter_radius, guided_filter_eps = 20, 0.01


# Based on Fast Guided Filter
# Kaiming He, Jian Sun
# https://arxiv.org/abs/1505.00996

def box(img, radius):
    dst = np.zeros_like(img)
    (r, c) = img.shape

    s = [radius, 1]
    c_sum = np.cumsum(img, 0)
    dst[0:radius+1, :, ...] = c_sum[radius:2*radius+1, :, ...]
    dst[radius+1:r-radius, :, ...] = c_sum[2*radius+1:r, :, ...] - c_sum[0:r-2*radius-1, :, ...]
    dst[r-radius:r, :, ...] = np.tile(c_sum[r-1:r, :, ...], s) - c_sum[r-2*radius-1:r-radius-1, :, ...]

    s = [1, radius]
    c_sum = np.cumsum(dst, 1)
    dst[:, 0:radius+1, ...] = c_sum[:, radius:2*radius+1, ...]
    dst[:, radius+1:c-radius, ...] = c_sum[:, 2*radius+1 : c, ...] - c_sum[:, 0 : c-2*radius-1, ...]
    dst[:, c-radius: c, ...] = np.tile(c_sum[:, c-1:c, ...], s) - c_sum[:, c-2*radius-1 : c-radius-1, ...]

    return dst


#--- utils.py ---

def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')
   

def get_cached_url():            
    if not os.path.exists(url_file):
        return None

    with open(url_file, 'r') as f:
        return f.read()
        

def save_cached_url(url):
    with open(url_file, 'w') as f:
        f.write(url)


def find_model_file():

    # Get first file with .onnx extension, pretty naive way
    candidates = glob.glob(os.path.join(default_model_folder, '*.onnx'))
    if len(candidates) == 0:
        raise Exception('No model found (expected at least one file with .onnx extension')
    
    return candidates[0]



def guided_filter(img, guide, radius, eps):
    (r, c) = img.shape

    CNT = box(np.ones([r, c]), radius)

    mean_img = box(img, radius) / CNT
    mean_guide = box(guide, radius) / CNT

    a = ((box(img * guide, radius) / CNT) - mean_img * mean_guide) / (((box(img * img, radius) / CNT) - mean_img * mean_img) + eps)
    b = mean_guide - a * mean_img

    return (box(a, radius) / CNT) * img + (box(b, radius) / CNT)


# Use GPU if it is available, otherwise CPU
provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in ort.get_available_providers() else "CPUExecutionProvider"

class SkyFilter(nn.Module):

    def __init__(self, model=default_model_url, ignore_cache=False, width=384, height=384, grid_size: tuple = (16, 16)):
        super(SkyFilter, self).__init__()
        self.model = model
        self.ignore_cache = ignore_cache
        self.width, self.height = width, height
        self.grid_size = grid_size

        print(' ?> Using provider %s' % provider)
        self.load_model()

    def forward(self, ground_image_vis, debug=False):
        """
        Applies a sky filter to remove the sky from an image.
        Parameters:
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
        ground_image_no_sky, sky_mask = self.run_img_array(ground_image_vis)

        # Dimensions of the image
        height, width = ground_image_no_sky.shape[:2]

        # Calculate the size of each grid cell
        cell_height = height // self.grid_size[0]
        cell_width = width // self.grid_size[1]

        # Initialize the grid mask
        grid_mask = np.zeros((self.grid_size[0], self.grid_size[1]), dtype=np.uint8)

        # Loop over each cell in the grid
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                # Define the region of interest (ROI) for this cell
                start_x = j * cell_width
                start_y = i * cell_height
                end_x = (j + 1) * cell_width if j < self.grid_size[1] - 1 else width
                end_y = (i + 1) * cell_height if i < self.grid_size[0] - 1 else height
                
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

    
    def load_model(self):
        
        # Check if model is path or url
        if not os.path.exists(self.model):
          
            if not os.path.exists(default_model_folder):
                os.mkdir(default_model_folder)

            if self.ignore_cache:

                print(" ?> We are ignoring the cache")
                self.model = self.get_model(self.model)

            else:

                cached_url = get_cached_url()

                if cached_url is None:

                    url = self.model
                    self.model = self.get_model(self.model)
                    save_cached_url(url)

                else:
                    
                    if cached_url != self.model:
                        url = self.model
                        self.model = self.get_model(self.model)
                        save_cached_url(url)
                    else:
                        self.model = find_model_file()

        print(' -> Loading the model')
        onnx_model = onnx.load(self.model)

        # Check the model
        try:
            onnx.checker.check_model(onnx_model)
        except onnx.checker.ValidationError as e:
            print(' !> The model is invalid: %s' % e)
            raise
        else:
            print(' ?> The model is valid!')

        self.session = ort.InferenceSession(self.model, providers=[provider])     


    def get_model(self, url):

        print(' -> Downloading model from: %s' % url)

        dest_file = os.path.join(default_model_folder, slugify(os.path.basename(url)))

        urllib.request.urlretrieve(url, dest_file)

        # Check if model is a zip file
        if os.path.splitext(url)[1].lower() == '.zip':
            print(' -> Extracting model')
            with zipfile.ZipFile(dest_file, 'r') as zip_ref:
                zip_ref.extractall(default_model_folder)
            os.remove(dest_file)

            return find_model_file()

        else:
            return dest_file


    def get_mask(self, img):

        height, width, c = img.shape

        # Resize image to fit the model input
        new_img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        new_img = np.array(new_img, dtype=np.float32)

        # Input vector for onnx model
        input = np.expand_dims(new_img.transpose((2, 0, 1)), axis=0)
        ort_inputs = {self.session.get_inputs()[0].name: input}

        # Run the model
        ort_outs = self.session.run(None, ort_inputs)

        # Get the output
        output = np.array(ort_outs)
        output = output[0][0].transpose((1, 2, 0))
        output = cv2.resize(output, (width, height), interpolation=cv2.INTER_LANCZOS4)
        output = np.array([output, output, output]).transpose((1, 2, 0))
        output = np.clip(output, a_max=1.0, a_min=0.0)

        return self.refine(output, img)        


    def refine(self, pred, img):

        refined = guided_filter(img[:,:,2], pred[:,:,0], guided_filter_radius, guided_filter_eps)

        res = np.clip(refined, a_min=0, a_max=1)
        
        # Convert res to CV_8UC1
        res = np.array(res * 255., dtype=np.uint8)
        
        # Thresholding
        # res = cv2.threshold(res, 127, 255, cv2.THRESH_BINARY_INV)[1]
        res = cv2.threshold(res, 127, 255, cv2.THRESH_BINARY)[1]
        
        return res
        

    def run_folder(self, folder, dest):

        print(' -> Processing folder ' + folder)

        # Remove trailing slash if present
        if folder[-1] == '/':
            folder = folder[:-1]

        if os.path.exists(dest) is False:
            os.mkdir(dest)

        img_names = os.listdir(folder)

        start = time.time()

        # Filter files to only include images
        img_names = [name for name in img_names if os.path.splitext(name)[1].lower() in ['.jpg', '.png', '.jpeg', '.bmp', '.tif', '.tiff']]

        for idx in range(len(img_names)):
            img_name = img_names[idx]
            print(' -> [%d / %d] processing %s' % (idx+1, len(img_names), img_name))
            self.run_img(os.path.join(folder, img_name), dest)

        expired = time.time() - start
                
        print('\n ?> Done in %.2f seconds' % expired)
        if len(img_names) > 0:
            print(' ?> Elapsed time per image: %.2f seconds' % (expired / len(img_names)))
            print('\n ?> Output saved in ' + dest)
        else:
            print(' ?> No images found')



    def run_img(self, img_path, dest):

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img / 255., dtype=np.float32)

        mask  = self.get_mask(img)
        
        img_name = os.path.basename(img_path)
        fpath = os.path.join(dest, img_name)

        fname, _ = os.path.splitext(fpath)
        mask_name = fname + '_mask.png'
        cv2.imwrite(mask_name, mask)
        
        return mask_name



    def run(self, source, dest):
      
        if os.path.exists(dest) is False:
            os.mkdir(dest)

        # check if source is array
        if isinstance(source, np.ndarray):

            for idx in range(len(source)):
                itm = source[idx]
                self.run(itm, dest)
            
        else:

            # Check if source is a directory or a file
            if os.path.isdir(source):
                self.run_folder(source, dest)
            else:
                print(' -> Processing: %s' % source)
                start = time.time()
                self.run_img(source, dest)
                print(" -> Done in %.2f seconds" % (time.time() - start))
                print(' ?> Output saved in ' + dest)


    def run_img_array(self, img_array):
        """
        Process an image array directly without needing a file path.
        """
        # Ensure input is within valid range for processing
        img_array_safe = np.clip(img_array, 0, 255)
        img = np.array(img_array_safe / 255., dtype=np.float32)
        mask = self.get_mask(img)
        mask = cv2.bitwise_not(mask)

        # Apply the mask to remove the sky
        ground_image_no_sky = cv2.bitwise_and(img_array_safe, img_array_safe, mask=mask)
        
        return ground_image_no_sky, mask


