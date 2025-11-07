import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import argparse
import json


DIRS = [
    "clip",
    "crop_0.15",
    "crop_0.20",
    "crop_0.20_nlayers_1",
    "crop_0.25",
    "crop_0.25_nlayers_1",
    "crop_0.25_nlayers_2",
    "crop_0.25_nlayers_3",
    "crop_0.25_nlayers_4",
    "crop_0.30",
    "crop_0.30_nlayers_1",
    "crop_0.30_nlayers_2",
    "crop_0.30_nlayers_3",
    "crop_0.30_nlayers_4",
    "crop_0.30_nlayers_5",
    "crop_0.30_nlayers_6",
    "crop_0.35",
    "crop_0.35_nlayers_3",
    "crop_0.35_nlayers_4",
    "crop_0.40_nlayers_3",
    "crop_0.40_nlayers_4",
    "cvglobal_crop_0.15",
    "cvglobal_crop_0.20",
    "cvglobal_crop_0.25",
    "cvglobal_crop_0.30",
    "cvglobal_fov_180",
    "cvglobal_fov_360",
    "cvglobal_fov_70",
    "cvglobal_fov_90",
    "cvusa_fov_180",
    "cvusa_fov_360",
    "cvusa_fov_70",
    "cvusa_fov_90",
    "dinov2",
    "dinov3_sat"
]


individual_regions = True

INDIVIDUAL_REGIONS = ['AFR', 'AFU', 'ASR', 'ASU', 'EUR', 'EUU', 'NAR', 'NAU', 'SAR', 'SAU']


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Redo the histogram and statistics for delta_yaws.')
    parser.add_argument('--main_dir', type=str, default=None, help='Directory where output files are saved')
    parser.add_argument('--threshold_k', type=int, default=5, help='Threshold in degrees for Recall@K calculation')
    args = parser.parse_args()

    if args.main_dir is None:
        if individual_regions:
            dirs = INDIVIDUAL_REGIONS
        else:
            dirs = DIRS
    else:
        dirs = [args.main_dir]

    for dir in dirs:
        
        if individual_regions:
            main_dir = os.path.join(r'..\results\final_results\cvglobal_fov_90', dir)
        else:
            main_dir = os.path.join(r'..\results', dir)

        try:
            with open(os.path.join(main_dir, 'delta_yaws.pkl'), 'rb') as f:
                global_delta_yaws = pickle.load(f)
        except FileNotFoundError:
            print(f"File not found: {os.path.join(main_dir, 'delta_yaws.pkl')}. Skipping...")
            continue

        # Plot combined histogram
        global_delta_yaws = np.array(global_delta_yaws)
        
        # Convert to directional error (0-180° maps to 0-90°)
        # A line at 179° is only 1° different from a line at 0°
        directional_errors = np.minimum(global_delta_yaws, 180 - global_delta_yaws)
        
        # Calculate metrics for directional errors
        dir_error_mean = np.mean(directional_errors)
        dir_error_std = np.std(directional_errors)
        dir_error_median = np.median(directional_errors)
        
        # Calculate metrics for original delta yaws
        error_mean = np.mean(global_delta_yaws)
        error_std = np.std(global_delta_yaws)
        error_median = np.median(global_delta_yaws)
        recall_at_k = np.mean(global_delta_yaws <= args.threshold_k) * 100.0
        
        # Calculate recall for directional errors
        tau_recall_at_k = np.mean(directional_errors <= args.threshold_k) * 100.0
        
        # Save all metrics to JSON
        metrics = {
            "dir": args.main_dir,
            "mean": float(error_mean),
            "median": float(error_median),
            "tau_mean": float(dir_error_mean),
            "tau_median": float(dir_error_median),
            f"recall_at_{args.threshold_k}": float(recall_at_k),
            f"tau_recall_at_{args.threshold_k}": float(tau_recall_at_k)
        }
        
        if individual_regions:
            with open(os.path.join(main_dir, 'ZeSCO.json'), 'w') as f:
                json.dump(metrics, f, indent=4)
        else:
            with open(os.path.join(main_dir, 'ZeSCO.json'), 'w') as f:
                json.dump(metrics, f, indent=4)
        
        print(f"\nOverall Delta Yaw Mean Error: {error_mean:.2f}°")
        print(f"Overall Delta Yaw Median Error: {error_median:.2f}°")
        print(f"Overall Recall@{args.threshold_k}: {recall_at_k:.2f}%")
        print(f"\nDirectional Error Mean (τMean): {dir_error_mean:.2f}°")
        print(f"Directional Error Median (τMedian): {dir_error_median:.2f}°")
        print(f"Directional Error Recall@{args.threshold_k} (τRecall): {tau_recall_at_k:.2f}%")

        # Plot histogram for delta yaws
        plt.figure(figsize=(10, 6))
        plt.hist(global_delta_yaws, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Delta Yaw (degrees)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Delta Yaw Distribution - CVGlobal\n' +
                r'$\tau$Mean: ' + f'{dir_error_mean:.2f}°, Median: {error_median:.2f}°, r@{args.threshold_k}: {recall_at_k:.2f}%',
                fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(main_dir, 'delta_yaws_hist.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Plot histogram for directional errors
        plt.figure(figsize=(10, 6))
        plt.hist(directional_errors, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Directional Error (degrees)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Directional Error Distribution - CVGlobal\n' +
                r'$\tau$Mean: ' + f'{dir_error_mean:.2f}°, Median: {dir_error_median:.2f}°, r@{args.threshold_k}: {recall_at_k:.2f}%',
                fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(main_dir, 'directional_error_hist.png'), dpi=300, bbox_inches='tight')
        plt.close()