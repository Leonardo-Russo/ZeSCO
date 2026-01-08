import os
import json
import argparse


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Collect all metrics.json files from subdirectories into one combined JSON file.')
    parser.add_argument('--main_dir', type=str, default=r'..\results', help='Main directory containing result subdirectories')
    parser.add_argument('--path', type=str, default=r'cvglobal_dinov3_imsize448_fov90_nl1_cp30_sp20', help='Subdirectory path to focus ons')
    parser.add_argument('--output_name', type=str, default='regional_metrics.json', help='Name of the output combined JSON file')
    args = parser.parse_args()
    
    main_dir = os.path.join(args.main_dir, args.path)
    combined_metrics = {}
    
    # Walk through all subdirectories in the main directory
    for root, dirs, files in os.walk(main_dir):
        if 'metrics.json' in files:
            metrics_path = os.path.join(root, 'metrics.json')

            # Get the relative path from main_dir to use as key
            relative_path = os.path.relpath(root, main_dir)
            if relative_path == '.':
                continue  # Skip the main directory itself
            try:
                with open(metrics_path, 'r') as f:
                    metrics_data = json.load(f)
                    combined_metrics[relative_path] = metrics_data
                    print(f"Loaded metrics from: {relative_path}")
            except Exception as e:
                print(f"Error loading {metrics_path}: {e}")
    
    # Save combined metrics to output file
    output_path = os.path.join(main_dir, args.output_name)
    with open(output_path, 'w') as f:
        json.dump(combined_metrics, f, indent=4)
    
    print(f"\nCombined {len(combined_metrics)} metrics files into: {output_path}")
