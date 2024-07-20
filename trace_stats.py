import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# Directory containing the trace datasets
trace_directory = 'traces'
figure_directory = './traces/figures/'

def plot_trace(Dataset_name, filename, timestamps, bandwidths):
    # Plot the trace data
    save_path = figure_directory + Dataset_name + '/' + 'plots/' + filename
    if not os.path.exists(figure_directory  + Dataset_name + '/' + 'plots/'):
        os.makedirs(figure_directory + Dataset_name + '/' + 'plots/')
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, bandwidths, label='Trace Data')
    plt.xlabel('Time (s)')
    plt.ylabel('Bandwidth (Mbps)')
    plt.title(f'Trace Dataset: {Dataset_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_path}.png')
    plt.savefig(f'{save_path}.pdf')
    plt.show()
    plt.close()

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# def plot_box_plot(dataset_data):
#     plt.figure(figsize=(15, 10))
#     data_to_plot = []
#     labels = []
#     for dataset, data in dataset_data.items():
#         if data:
#             bandwidths = [bandwidth for trace in data for bandwidth in trace[1]]
#             data_to_plot.append(bandwidths)
#             labels.append(dataset)
#     colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FFD700', '#FF6F61', '#009688']
#     bplot = plt.boxplot(data_to_plot,patch_artist=True, labels=labels)
#     # Coloring each box
#     for patch, color in zip(bplot['boxes'], colors):
#         patch.set_facecolor(color)
    
#     plt.xticks(fontsize=26)
#     plt.yticks(fontsize=24)

#     plt.xlabel('Dataset',fontsize=25)
#     plt.ylabel('Bandwidth (Mbps)',fontsize=25)
#     plt.title('Box Plot of Bandwidth for Different Network Traces',fontsize=20,fontweight='bold')
#     plt.grid(True, which="both", ls="--", alpha=0.3)
#     plt.savefig(figure_directory+'box_plot_bandwidth.png')
#     plt.savefig(figure_directory+'box_plot_bandwidth.pdf')
#     plt.show()
#     plt.close()
def plot_box_plot(dataset_data):
    plt.figure(figsize=(15, 10))
    data_to_plot = []
    labels = []
    for dataset, data in dataset_data.items():
        if data:
            bandwidths = [bandwidth for trace in data for bandwidth in trace[1]]
            data_to_plot.append(bandwidths)
            labels.append(dataset)
    
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FFD700', '#FF6F61', '#009688']
    
    # 增加箱体宽度和中位数线宽度
    bplot = plt.boxplot(data_to_plot, patch_artist=True, labels=labels, 
                        widths=0.4,  # 增加箱体宽度
                        medianprops={'linewidth': 3})  # 增加中位数线宽度
    
    # 为每个箱体上色并增加边框宽度
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')  # 设置边框颜色
        patch.set_linewidth(3)  # 增加边框宽度
    
    # 增加须的宽度
    for whisker in bplot['whiskers']:
        whisker.set_linewidth(1.5)
    
    # 增加帽的宽度
    for cap in bplot['caps']:
        cap.set_linewidth(2)
    
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=24)
    
    plt.xlabel('Dataset', fontsize=25)
    plt.ylabel('Bandwidth (Mbps)', fontsize=25)
    plt.title('Box Plot of Bandwidth for Different Network Traces', fontsize=20, fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    plt.tight_layout()  # 确保所有元素都能完整显示
    
    plt.savefig(figure_directory+'box_plot_bandwidth.png', dpi=300, bbox_inches='tight')
    plt.savefig(figure_directory+'box_plot_bandwidth.pdf', bbox_inches='tight')
    
    plt.show()
    plt.close()
def plot_moving_average(dataset_data, window_size=50):
    for dataset, data in dataset_data.items():
        if not data:
            continue
        plt.figure(figsize=(15, 10))
        # Limiting the number of traces plotted to 5 for clarity
        for i, (timestamps, bandwidths) in enumerate(data):
            if i >= 5:
                break

            mov_avg = moving_average(bandwidths, window_size)
            plt.plot(timestamps[:len(mov_avg)], mov_avg, label=f'Moving Average {i+1}')
        plt.xlabel('Time (s)')
        plt.ylabel('Bandwidth (Mbps)')
        plt.title(f'Moving Average of Bandwidth for {dataset} (Window Size={window_size})')
        plt.legend(loc='upper right', fontsize='small')
        plt.grid(True)
        plt.savefig(f'{figure_directory}{dataset}_moving_average.png')
        plt.savefig(f'{figure_directory}{dataset}_moving_average.pdf')
        plt.show()
        plt.close()

def plot_histogram(dataset_data):
    plt.figure(figsize=(15, 10))
    for dataset, data in dataset_data.items():
        if data:
            bandwidths = [bandwidth for trace in data for bandwidth in trace[1]]

            plt.hist(bandwidths, bins=50, alpha=0.5, label=dataset)
    plt.xlabel('Bandwidth (Mbps)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Bandwidth for Different Network Traces')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(figure_directory+'histogram_bandwidth.png')
    plt.savefig(figure_directory+'histogram_bandwidth.pdf')
    plt.show()
    plt.close()

def plot_spectral_analysis(dataset_data):
    for dataset, data in dataset_data.items():
        if not data:
            continue
        plt.figure(figsize=(15, 10))
        # Limiting the number of traces plotted to 5 for clarity
        for i, (timestamps, bandwidths) in enumerate(data):
            if i >= 3:
                break
            fft_vals = np.fft.fft(bandwidths)
            fft_freq = np.fft.fftfreq(len(fft_vals), d=(timestamps[1] - timestamps[0]))
            plt.plot(fft_freq, np.abs(fft_vals), label=f'Spectral Analysis {i+1}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.title(f'Spectral Analysis of Bandwidth for {dataset}')
        plt.legend(loc='upper right', fontsize='small')
        plt.grid(True)
        plt.savefig(f'{figure_directory}{dataset}_spectral_analysis.png')
        plt.savefig(f'{figure_directory}{dataset}_spectral_analysis.pdf')
        plt.show()
        plt.close()
def plot_time_series_variability(dataset_data):
    for dataset, data in dataset_data.items():
        if not data:
            continue
        plt.figure(figsize=(15, 10))
        # Limiting the number of traces plotted to 5 for clarity
        for i, (timestamps, bandwidths) in enumerate(data):
            if i >= 5:
                break
            variability = np.diff(bandwidths)
            plt.plot(timestamps[1:], variability, label=f'Variability {i+1}')
        plt.xlabel('Time (s)')
        plt.ylabel('Bandwidth Change (Mbps)')
        plt.title(f'Time Series Variability of Bandwidth for {dataset}')
        plt.legend(loc='upper right', fontsize='small')
        plt.grid(True)
        plt.savefig(f'{figure_directory}{dataset}_variability.png')
        plt.savefig(f'{figure_directory}{dataset}_variability.pdf')
        plt.show()
        plt.close()


# Dictionary to store the data for each dataset
dataset_data = {
    'fcc18': [],
    'ghent': [],
    'hsr': [],
    'lab': [],
    'oboe': [],
    'HSDPA': [],
    # 'lumos5G': []
}

# Iterate over each dataset directory
for dataset in dataset_data.keys():
    dataset_dir = os.path.join(trace_directory, dataset)
    
    # Check if the dataset directory exists
    if not os.path.exists(dataset_dir):
        print(f"Directory not found: {dataset_dir}")
        continue
    
    # Iterate over each trace file in the dataset directory
    trace_files = os.listdir(dataset_dir)
    
    if not trace_files:
        print(f"No trace files found in directory: {dataset_dir}")
        continue
    
    for filename in trace_files:
        timestamps = []
        bandwidths = []
        
        # Read the data from the file and skip directory 
        with open(os.path.join(dataset_dir, filename), 'r') as file:
            for line in file:
                try:
                    timestamp, bandwidth = line.strip().split()
                    timestamps.append(float(timestamp))
                    bandwidths.append(float(bandwidth))
                except ValueError:
                    print(f"Invalid data format in file: {filename}")
                    continue
        
        # plot the trace data
        plot_trace(dataset, filename, timestamps, bandwidths)
        # Append the data to the respective dataset list
        dataset_data[dataset].append((timestamps, bandwidths))

# Check if any data was loaded
if not any(dataset_data.values()):
    print("No data found in the trace files.")
    exit(1)

# Plot the CDF for average bandwidth and standard deviation for each dataset
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
Max_avg_bandwidth = 0
Max_std_dev = 0
markers_map = {
    'fcc18': 'o',
    'ghent': 's',
    'hsr': 'D',
    'lab': '^',
    'oboe': 'v',
    'HSDPA': 'X',
    # 'lumos5G': 'P'
}
line_styles_map = {
    'fcc18': '-',
    'ghent': '--',
    'hsr': '-.',
    'lab': ':',
    'oboe': '-',
    'HSDPA': '--',
    # 'lumos5G': '-.'
}
for dataset, data in dataset_data.items():
    if not data:
        continue

    # Calculate average bandwidth and standard deviation for each trace in the dataset
    avg_bandwidths = []
    std_devs = []
    for _, bandwidths in data:

        avg_bandwidths.append(sum(bandwidths) / len(bandwidths))
        variance = sum((x - avg_bandwidths[-1]) ** 2 for x in bandwidths) / len(bandwidths)
        std_devs.append(variance ** 0.5)
    
    # Plot CDF for average bandwidth
    sorted_avg_bandwidths = np.sort(avg_bandwidths)
    if Max_avg_bandwidth < max(sorted_avg_bandwidths):
        Max_avg_bandwidth = max(sorted_avg_bandwidths)
    cdf_avg = np.arange(len(sorted_avg_bandwidths)) / float(len(sorted_avg_bandwidths))
    ax1.plot(sorted_avg_bandwidths, cdf_avg * 100, linestyle=line_styles_map[dataset], marker=markers_map[dataset], markersize=1.5, label=dataset)
    
    # Plot CDF for standard deviation
    sorted_std_devs = np.sort(std_devs)
    if Max_std_dev < max(sorted_std_devs):
        Max_std_dev = max(sorted_std_devs)
    cdf_std = np.arange(len(sorted_std_devs)) / float(len(sorted_std_devs))
    
    ax2.plot(sorted_std_devs, cdf_std * 100, linestyle=line_styles_map[dataset], marker=markers_map[dataset], markersize=1.5, label=dataset)

# Set labels and limits for the average bandwidth CDF plot
ax1.set_xlabel('Average Bandwidth (Mbps)', fontsize=18)
ax1.set_ylabel('CDF (%)', fontsize=18)
ax1.set_xlim(0, Max_avg_bandwidth)
ax1.set_ylim(0, 100)
ax1.grid(True)
ax1.legend(fontsize=14)

# Set labels and limits for the standard deviation CDF plot
ax2.set_xlabel('Standard Deviation (Mbps)', fontsize=14)
ax2.set_ylabel('CDF (%)', fontsize=14)
ax2.set_xlim(0, Max_std_dev)
ax2.set_ylim(0, 100)
ax2.grid(True)
ax2.legend(fontsize=14)

plt.tight_layout()
plt.savefig(figure_directory+'cdf_plots.pdf')
plt.show()

# Generate additional representative figures
plot_box_plot(dataset_data)
# plot_moving_average(dataset_data)
# plot_histogram(dataset_data)
# plot_spectral_analysis(dataset_data)
# plot_time_series_variability(dataset_data)
