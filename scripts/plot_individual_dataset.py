import matplotlib.pyplot as plt
import numpy as np

datasets = ['AFR', 'AFU', 'ASR', 'ASU', 'EUR', 'EUU', 'NAR', 'NAU', 'SAR', 'SAU']

# Median Delta Yaw Errors for each experiment
errors = np.array([
    [31.36, 23.21, 23.86, 27.00, 18.57, 16.71, 20.50, 20.93, 39.93, 25.50],
    [21.93, 23.86, 21.86, 20.43, 18.57, 16.36, 18.57, 20.57, 37.57, 23.93],
    [19.86, 20.50, 20.79, 20.50, 17.43, 14.14, 17.93, 17.00, 30.50, 23.79],
    [20.36, 18.36, 20.36, 21.07, 16.57, 15.07, 15.86, 15.71, 36.86, 24.71],
    [22.57, 18.79, 20.71, 19.29, 17.29, 17.43, 15.36, 15.43, 38.21, 24.29]
])

# ===================== Line Plot =====================
plt.figure(figsize=(12,6))
for i in range(errors.shape[0]):
    plt.plot(datasets, errors[i], marker='o', label=f'Run {i+1}')

plt.title('Median Delta Yaw Errors by Dataset (Line Plot)')
plt.xlabel('Dataset')
plt.ylabel('Median Delta Yaw Error (°)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(r'..\median_delta_yaw_errors_line_plot.png', dpi=300)
plt.show()

# ===================== Grouped Bar Plot =====================
x = np.arange(len(datasets))  # dataset positions
width = 0.15  # width of each bar

plt.figure(figsize=(14,6))
for i in range(errors.shape[0]):
    plt.bar(x + i*width, errors[i], width=width, label=f'Run {i+1}')

plt.xticks(x + width*2, datasets)  # center tick labels
plt.title('Median Delta Yaw Errors by Dataset (Bar Plot)')
plt.xlabel('Dataset')
plt.ylabel('Median Delta Yaw Error (°)')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(r'..\median_delta_yaw_errors_bar_plot.png', dpi=300)
plt.show()
