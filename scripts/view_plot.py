import pickle
import matplotlib.pyplot as plt
import argparse
import os
import sys
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plots

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='View a pickled matplotlib figure.')
    parser.add_argument('file', type=str, help='Path to the .pkl file')
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File {args.file} not found.")
        sys.exit(1)

    print(f"Opening {args.file}...")
    try:
        with open(args.file, 'rb') as f:
            fig = pickle.load(f)
        plt.show()
    except Exception as e:
        print(f"Error loading plot: {e}")
