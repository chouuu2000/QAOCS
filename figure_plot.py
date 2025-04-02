import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Load the provided CSV files
file_paths = [
    'BBB_360p24_performance_comparison.csv',
    # 'LOL_3D_performance_comparison.csv',
    # 'sport_highlight_performance_comparison.csv',
    # 'underwater_performance_comparison.csv',
    # 'video_game_performance_comparison.csv',
    # 'sport_long_take_performance_comparison.csv'
]

dataframes = [pd.read_csv(file) for file in file_paths]

# Combine all dataframes into one for easier analysis
combined_df = pd.concat(dataframes, ignore_index=True)


# Categorize traces into Slow, Medium, and Fast based on CDF plots
def correct_categorize_trace(file_name):
    if 'oboe' in file_name or 'fcc18' in file_name:
        return 'Slow'
    elif 'lab' in file_name or 'hsr' in file_name:
        return 'Medium'
    elif 'ghent' in file_name:
        return 'Fast'
    return 'Unknown'

combined_df['Network Type'] = combined_df['File'].apply(correct_categorize_trace)

# Calculate VMAF Change and QoE for each entry
combined_df['VMAF Change'] = combined_df['Average VMAF Smoothness']
combined_df['QoE'] = combined_df['Average VMAF Score'] - (combined_df['Stall Time(s)']* 0.1) - (combined_df['Switch Ratio(%)'] * 100)

# Aggregate data by video category for plotting
video_categories = combined_df['File'].str.extract(r'([A-Za-z]+)').squeeze().unique()
# print video_categories 
print("Video ->>",video_categories)
combined_df['Video Category'] = combined_df['File'].str.extract(r'([A-Za-z]+)').squeeze()

# Box Plots for key metrics across different network types and methods
fig, axes = plt.subplots(2, 2, figsize=(18, 15))
metrics_to_boxplot = [
    ('Average VMAF Score', 'Score'),
    ('Stall Ratio(%)', 'Ratio'),
    ('Switch Ratio(%)', 'Ratio'),
    ('Average Bitrate(bps)', 'bps')
]

for i, (metric, unit) in enumerate(metrics_to_boxplot):
    row, col = divmod(i, 2)
    sns.boxplot(data=combined_df, x='Method', y=metric, hue='Network Type', ax=axes[row, col])
    axes[row, col].set_title(f'{metric} by Method and Network Type')
    axes[row, col].set_xlabel('')
    axes[row, col].set_ylabel(unit)
# for i, (metric, unit) in enumerate(metrics_to_boxplot):
#     row, col = divmod(i, 2)
#     grouped_data = [combined_df[combined_df['Network Type'] == network_type][metric] for network_type in combined_df['Network Type'].unique()]
#     labels = combined_df['Network Type'].unique()
#     axes[row, col].boxplot(grouped_data, labels=labels)
#     axes[row, col].set_title(f'{metric} by Network Type')
#     axes[row, col].set_xlabel('Network Type')
#     axes[row, col].set_ylabel(unit)
plt.tight_layout()
plt.legend(title='Network Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
plt.savefig('box_plots.png')
# CDF plots for key metrics
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
metrics_to_cdf = [
    ('Average VMAF Score', 'Score'),
    ('Stall Ratio(%)', '%')
]

def plot_cdf(data, metric, ax, label):
    sorted_data = np.sort(data)
    yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1) * 100
    ax.plot(sorted_data, yvals, label=label)

for i, (metric, unit) in enumerate(metrics_to_cdf):
    ax = axes[i]
    for network_type in combined_df['Network Type'].unique():
        for method in combined_df['Method'].unique():
            data = combined_df[(combined_df['Network Type'] == network_type) & (combined_df['Method'] == method)][metric]
            plot_cdf(data, metric, ax, f'{method} - {network_type}')
    ax.set_title(f'CDF of {metric}')
    ax.set_xlabel(metric)
    ax.set_ylabel('CDF (%)')
    ax.legend()

plt.tight_layout()
plt.show()
plt.savefig('cdf_plots.png')
# Scatter Plot for Average Bitrate vs Average VMAF Score
plt.figure(figsize=(12, 8))
sns.scatterplot(data=combined_df, x='Average Bitrate(bps)', y='Average VMAF Score', hue='Method', style='Network Type')
plt.title('Average Bitrate vs Average VMAF Score by Method and Network Type')
plt.xlabel('Average Bitrate (bps)')
plt.ylabel('Average VMAF Score')
plt.legend(title='Method and Network Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Heatmap for performance metrics
heatmap_data = combined_df.groupby(['Network Type', 'Method']).agg(
    {
        'Average VMAF Score': 'mean',
        'Stall Time(s)': 'mean',
        'Switch Ratio(%)': 'mean',
        'Average Bitrate(bps)': 'mean',
        'Total Size(MB)': 'mean'
    }
).unstack().T

plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title('Performance Metrics Heatmap by Method and Network Type')
plt.xlabel('Method and Network Type')
plt.ylabel('Performance Metrics')
plt.show()
plt.savefig('heatmap.png')

# Radar Chart for overall performance comparison
from math import pi

def create_radar_chart_data(df, method):
    metrics = ['Average VMAF Score', 'Stall Time(s)', 'Switch Ratio(%)', 'Average Bitrate(bps)', 'Total Size(MB)']
    values = df[df['Method'] == method][metrics].mean().tolist()
    values += values[:1]  # Repeat the first value to close the circle
    return values

categories = ['Average VMAF Score', 'Stall Time(s)', 'Switch Ratio(%)', 'Average Bitrate(bps)', 'Total Size(MB)']
num_vars = len(categories)
angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
methods = combined_df['Method'].unique()
colors = sns.color_palette("husl", len(methods))

for i, method in enumerate(methods):
    values = create_radar_chart_data(combined_df, method)
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=method, color=colors[i])
    ax.fill(angles, values, color=colors[i], alpha=0.25)

plt.xticks(angles[:-1], categories, color='grey', size=12)
ax.yaxis.set_visible(False)
plt.title('Comparison of Methods on Multiple Metrics', size=15, color='black', y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
plt.show()
plt.savefig('radar_chart.png')

# Detailed Analysis Plots similar to reference figure
# categories = combined_df['Video Category'].unique()
# stall_ratio = combined_df.groupby('Video Category')['Stall Ratio(%)'].mean().to_dict()
# vmaf_scores = combined_df.groupby('Video Category')['Average VMAF Score'].mean().to_dict()
# stall_ratio_errors = combined_df.groupby('Video Category')['Stall Ratio(%)'].std().to_dict()
# vmaf_errors = combined_df.groupby('Video Category')['Average VMAF Score'].std().to_dict()
# vmaf_vs_vmaf_change = combined_df.groupby('Video Category')['VMAF Change'].mean().to_dict()
# vmaf_vs_vmaf_change_err = combined_df.groupby('Video Category')['VMAF Change'].std().to_dict()
# qoe_vs_buffer = combined_df.groupby('Video Category')['QoE'].mean().to_dict()
# qoe_vs_buffer_err = combined_df.groupby('Video Category')['QoE'].std().to_dict()
# buffer_sizes = combined_df.groupby('Video Category')['Average Buffer State(s)'].mean().to_dict()
# qoe_data = {category: combined_df[combined_df['Video Category'] == category]['QoE'].values for category in categories}
stall_ratio = combined_df.groupby('Method')['Stall Ratio(%)'].mean().to_dict()
vmaf_scores = combined_df.groupby('Method')['Average VMAF Score'].mean().to_dict()
stall_ratio_errors = combined_df.groupby('Method')['Stall Ratio(%)'].std().to_dict()
vmaf_errors = combined_df.groupby('Method')['Average VMAF Score'].std().to_dict()
vmaf_vs_vmaf_change = combined_df.groupby('Method')['VMAF Change'].mean().to_dict()
vmaf_vs_vmaf_change_err = combined_df.groupby('Method')['VMAF Change'].std().to_dict()
qoe_vs_buffer = combined_df.groupby('Method')['QoE'].mean().to_dict()
qoe_vs_buffer_err = combined_df.groupby('Method')['QoE'].std().to_dict()
buffer_sizes = combined_df.groupby('Method')['Average Buffer State(s)'].mean().to_dict()
qoe_data = {method: combined_df[combined_df['Method'] == method]['QoE'].values for method in methods}

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
#markers = ['o', 's', 'D', '^', 'v', '>', '<', 'p']
markers = ['o', 's']

# Plot (a) VMAF vs. Stall Ratio
for name, marker in zip(stall_ratio.keys(), markers):
    x = stall_ratio[name]
    y = vmaf_scores[name]
    x_err = stall_ratio_errors[name]
    y_err = vmaf_errors[name]
    axs[0,0].errorbar(x, y, xerr=x_err, yerr=y_err, label=name, marker=marker, capsize=5)

axs[0, 0].set_xlabel('Time Spent on Stall (%)')
axs[0, 0].set_ylabel('Video Quality (VMAF)')
axs[0, 0].set_title('(a) VMAF vs. Stall Ratio')
axs[0, 0].legend()
axs[0, 0].grid(True)
axs[0, 0].invert_xaxis()

# Plot (b) VMAF vs. VMAF Change
for name, marker in zip(vmaf_vs_vmaf_change.keys(), markers):
    x = combined_df[combined_df['Method'] == name]['VMAF Change']
    y = combined_df[combined_df['Method'] == name]['Average VMAF Score']
    x_err = vmaf_vs_vmaf_change_err[name]
    y_err = vmaf_errors[name]
    axs[0,1].errorbar(x.mean(), y.mean(), xerr=x_err, yerr=y_err, label=name, marker=marker, capsize=5)
axs[0, 1].set_xlabel('Quality Smoothness (VMAF Change)')
axs[0, 1].set_ylabel('Video Quality (VMAF)')
axs[0, 1].set_title('(b) VMAF vs. VMAF Change')
axs[0, 1].legend()

# Plot (c) QoE_DNN vs. Buffer
for name, marker in zip(qoe_vs_buffer.keys(), markers):
    x = buffer_sizes[name]
    y = qoe_vs_buffer[name]
    x_err = combined_df[combined_df['Video Category'] == name]['Average Buffer State(s)'].std()
    y_err = qoe_vs_buffer_err[name]
    axs[1,0].errorbar(x, y, xerr=x_err, yerr=y_err, label=name, marker=marker, capsize=5)

axs[1, 0].set_xlabel('Buffer (s)')
axs[1, 0].set_ylabel('QoE')
axs[1, 0].set_title('(c) QoE vs. Buffer')
axs[1, 0].legend()

# Plot (d) CDF of QoE
def plot_cdf(data, ax, label, linestyle):
    sorted_data = np.sort(data)
    yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
    ax.plot(sorted_data, yvals, label=label, linestyle=linestyle)

linestyles = ['--', '-', ':', '-.', '--', '-']
for (category, vals), ls in zip(qoe_data.items(), linestyles):
    plot_cdf(vals, axs[1,1], category, ls)

axs[1, 1].set_xlabel('QoE')
axs[1, 1].set_ylabel('CDF')
axs[1, 1].set_title('(d) CDF of QoE')
axs[1, 1].legend()

# Adjust spacing between subplots
plt.tight_layout()
plt.show()
# Save the figure
plt.savefig('detailed_analysis_plots.png')
