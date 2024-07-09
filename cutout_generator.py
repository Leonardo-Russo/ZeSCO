import cv2
import numpy as np
import math

def equirectangular_to_rectilinear(img, fov, theta, phi, width, height):
    """
    Converts an equirectangular image to a rectilinear image.
    
    Parameters:
    - img: Input equirectangular image
    - fov: Field of view in degrees (e.g., 90 for a wide-angle view)
    - theta: Horizontal angle of the center of the view in degrees (0 is forward, 90 is right, -90 is left)
    - phi: Vertical angle of the center of the view in degrees (0 is straight, 90 is up, -90 is down)
    - width: Output image width
    - height: Output image height
    """
    fov = math.radians(fov)
    theta = math.radians(theta)
    phi = math.radians(phi)
    
    equ_h, equ_w = img.shape[:2]
    
    # Calculate the focal length
    f = width / (2 * math.tan(fov / 2))
    
    # Create a grid of (x, y) coordinates
    x = np.linspace(-width / 2, width / 2, width)
    y = np.linspace(-height / 2, height / 2, height)
    x, y = np.meshgrid(x, y)
    
    # Convert (x, y) to spherical coordinates (theta, phi)
    x_map = np.arctan2(x, f)
    y_map = np.arctan2(y, np.sqrt(x**2 + f**2))
    
    # Add the rotation angles
    x_map += theta
    y_map += phi
    
    # Normalize the angles to be within the valid range
    x_map = x_map % (2 * np.pi)
    y_map = np.clip(y_map, -np.pi / 2, np.pi / 2)
    
    # Convert spherical coordinates back to equirectangular coordinates
    equ_x = (x_map / (2 * np.pi) + 0.5) * equ_w
    equ_y = (y_map / np.pi + 0.5) * equ_h
    
    # Remap the pixels
    dst = cv2.remap(img, equ_x.astype(np.float32), equ_y.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    
    return dst

# Load your 360° equirectangular image
equ_image = cv2.imread('/home/lrusso/cvusa/streetview/panos/27/-81/27.020071_-81.994164.jpg')

# Parameters for the cutout
fov = 90  # Field of view in degrees
theta = 0  # Horizontal angle
phi = 0  # Vertical angle
width = 800  # Output image width
height = 600  # Output image height

# Extract the rectilinear cutout
rectilinear_image = equirectangular_to_rectilinear(equ_image, fov, theta, phi, width, height)

# Save or display the result
cv2.imwrite('rectilinear_cutout.jpg', rectilinear_image)
cv2.imshow('Rectilinear Cutout', rectilinear_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
