import os

DATASETS = ['AFR', 'AFU', 'ASR', 'ASU', 'EUR', 'EUU', 'NAR', 'NAU', 'SAR', 'SAU']
DATASETS_FULL_NAMES = {
    'AFR': 'Africa Rural',
    'AFU': 'Africa Urban',
    'ASR': 'Asia Rural',
    'ASU': 'Asia Urban',
    'EUR': 'Europe Rural',
    'EUU': 'Europe Urban',
    'NAR': 'North America Rural',
    'NAU': 'North America Urban',
    'SAR': 'South America Rural',
    'SAU': 'South America Urban'
}

if __name__ == '__main__':

    results_dir = 'results'

    print("Median Delta Yaw Errors:")
    print("-----------------------")
    for dataset_name in DATASETS:
        dataset_dir = os.path.join(results_dir, dataset_name)

        # Read the results file
        results_file = os.path.join(dataset_dir, 'info.txt')
        if not os.path.isfile(results_file):
            print(f"Results file for {dataset_name} not found.")
            continue

        with open(results_file, 'r') as f:
            lines = f.readlines()

        # Process each line in the results file
        for line in lines:
            if 'Median Delta Yaw Error:' in line:
                # Extract the value after the colon
                value = line.split(':')[-1].strip()
                print(f"{DATASETS_FULL_NAMES[dataset_name]}:\t{value}")
