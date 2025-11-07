import numpy as np
import pickle
import os
import argparse


def load_results(results_dir):
    """Load delta_yaws from pickle file and compute statistics."""
    delta_yaws_path = os.path.join(results_dir, 'delta_yaws.pkl')
    
    if not os.path.exists(delta_yaws_path):
        print(f"Warning: {delta_yaws_path} not found")
        return None
    
    with open(delta_yaws_path, 'rb') as f:
        delta_yaws = pickle.load(f)
    
    delta_yaws = np.array(delta_yaws)
    
    # Compute statistics
    mean_error = np.mean(delta_yaws)
    median_error = np.median(delta_yaws)
    recall_at_5 = np.mean(delta_yaws <= 5.0) * 100.0  # percentage
    
    return {
        'mean': mean_error,
        'median': median_error,
        'recall_at_5': recall_at_5
    }


def generate_latex_table(zesco_dirs, crop_values):
    """Generate LaTeX table from results directories."""
    
    print("Loading results...")
    print("-" * 60)
    
    # Load results for all directories
    zesco_results = []
    
    for i, (zesco_dir, crop) in enumerate(zip(zesco_dirs, crop_values)):
        print(f"\nCrop {crop}:")
        print(f"  ZeSCO dir: {zesco_dir}")
        
        zesco_stats = load_results(zesco_dir)
        
        if zesco_stats:
            print(f"  ZeSCO: Mean={zesco_stats['mean']:.2f}°, Median={zesco_stats['median']:.2f}°, r@5={zesco_stats['recall_at_5']:.2f}%")
        
        zesco_results.append(zesco_stats)
    
    # Generate LaTeX table
    print("\n" + "=" * 60)
    print("LaTeX Table:")
    print("=" * 60)
    print()
    
    latex_table = r"""\begin{table}[t]
\centering
\caption{ZeSCO performance across different cropping threshold values. Metrics reported: mean, median, and recall at 5 (r@5$^\circ$).}
\label{tab:threshold_results}
\setlength{\tabcolsep}{3.2pt} % tighter column spacing
\begin{tabular}{l|ccc}
\toprule
\textbf{Cropping} & \textbf{Mean} & \textbf{Median} & \textbf{r@5$^\circ$} \\ 
\midrule
"""
    
    # Add data rows
    for crop, zesco in zip(crop_values, zesco_results):
        if zesco:
            line = f"{crop} & {zesco['mean']:.2f}$^\circ$ & {zesco['median']:.2f}$^\circ$ & {zesco['recall_at_5']:.2f} \\\\\n"
        else:
            line = f"{crop} & -- & -- & -- \\\\\n"
        
        latex_table += line
    
    latex_table += r"""\bottomrule
\end{tabular}
\end{table}"""
    
    print(latex_table)
    print()
    
    return latex_table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate LaTeX table from results directories.')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path for LaTeX table (optional)')
    args = parser.parse_args()
    
    # Convert relative paths to absolute paths
    results_base = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')

    zesco_dirs = ['crop_0.15', 'crop_0.20', 'crop_0.25', 'crop_0.30_nlayers_4', 'crop_0.35_nlayers_4', 'crop_0.40_nlayers_4']
    crop_values = ['0.15', '0.20', '0.25', '0.30', '0.35', '0.40']
    zesco_dirs_full = [os.path.join(results_base, d) for d in zesco_dirs]

    # Generate table
    latex_table = generate_latex_table(zesco_dirs_full, crop_values)
    
    # Save to file if specified
    if args.output:
        with open(args.output, 'w') as f:
            f.write(latex_table)
        print(f"LaTeX table saved to: {args.output}")
