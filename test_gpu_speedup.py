"""
Test script to benchmark GPU vs CPU performance for find_alignment
"""

import numpy as np
import time
import sys
sys.path.insert(0, 'c:\\Users\\russ_le\\Projects\\ZeSCO')

from zesco import utils

# Create synthetic data matching your typical use case
num_layers = 3
grid_size = 16
feature_dim = 768  # CLIP/DINO feature dimension
num_orientations = 64  # Assuming 360/angle_step

print(f"Benchmark Configuration:")
print(f"  Grid size: {grid_size}x{grid_size}")
print(f"  Num layers: {num_layers}")
print(f"  Num orientations: {num_orientations}")
print(f"  Feature dim: {feature_dim}")
print(f"  Total comparisons: {num_orientations * grid_size} = {num_orientations * grid_size}\n")

# Generate random data
vertical_tokens = np.random.randn(num_layers, grid_size, feature_dim).astype(np.float32)
radial_tokens = np.random.randn(num_layers, num_orientations, feature_dim).astype(np.float32)
image_span = 90.0

# Simple cosine distance loss for CPU mode
def cosine_distance(a, b):
    """Cosine distance between two arrays of vectors"""
    # a, b shape: (num_layers, feature_dim)
    cos_sim = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)
    return np.mean(1 - cos_sim)

print(f"Device: {utils.DEVICE}")
print(f"GPU Available: {utils.USE_GPU}\n")

# Warm-up run (compiles GPU kernels if using CUDA)
print("Warming up...")
_ = utils.find_alignment(cosine_distance, vertical_tokens, radial_tokens, grid_size, image_span)

# Benchmark with GPU enabled
print("\n=== GPU-Accelerated Mode ===")
start = time.time()
for i in range(5):
    result = utils.find_alignment(cosine_distance, vertical_tokens, radial_tokens, grid_size, image_span)
    print(f"Run {i+1}: Orientation={result[0]:.2f}°, Min Distance={result[2]:.4f}")
gpu_time = (time.time() - start) / 5
print(f"Average GPU time: {gpu_time:.4f} seconds")

# Benchmark with GPU disabled (CPU fallback)
if utils.USE_GPU:
    print("\n=== CPU Mode (for comparison) ===")
    utils.USE_GPU = False
    start = time.time()
    for i in range(5):
        result = utils.find_alignment(cosine_distance, vertical_tokens, radial_tokens, grid_size, image_span)
        print(f"Run {i+1}: Orientation={result[0]:.2f}°, Min Distance={result[2]:.4f}")
    cpu_time = (time.time() - start) / 5
    print(f"Average CPU time: {cpu_time:.4f} seconds")
    
    print(f"\n{'='*50}")
    print(f"SPEEDUP: {cpu_time/gpu_time:.2f}x faster with GPU!")
    print(f"{'='*50}")
    
    # Re-enable GPU
    utils.USE_GPU = True
else:
    print("\nNo GPU available - CPU-only mode")

print("\nBenchmark complete!")
