import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Load the provided CSV files
file_paths = [
    'BBB_360p24_performance_comparison.csv',
    'TOS_360p24_performance_comparison.csv',
    'LOL_3D_performance_comparison.csv',
    'underwater_performance_comparison.csv',
    'video_game_performance_comparison.csv',
    'sport_long_take_performance_comparison.csv',
    'sport_highlight_performance_comparison.csv',
]
vmaf_weights = 0.13
switch_weights = 0.75
rebuffer_weights = 0.2

dataframes = [pd.read_csv(file) for file in file_paths]

# Combine all dataframes into one for easier analysis
combined_df = pd.concat(dataframes, ignore_index=True)

# Function to filter data to 5-95 percentile
def filter_percentile(df, lower=5, upper=90):
    return df[(df >= df.quantile(lower / 100)) & (df <= df.quantile(upper / 100))]

# Filter data to 5-95 percentile

# for column in combined_df.select_dtypes(include='number').columns:
#     combined_df[column] = filter_percentile(combined_df[column])
combined_df["Stall Ratio(%)"] = filter_percentile(combined_df["Stall Ratio(%)"])
combined_df=(combined_df.dropna())


# Categorize traces into Slow, Medium, and Fast based on CDF plots
def correct_categorize_trace(file_name):
    if 'oboe' in file_name :
        return 'Slow'
    elif 'fcc18' in file_name or 'hsr' in file_name:
        return 'Medium'
    elif 'ghent' in file_name or 'lab' in file_name:
        return 'Fast'
    elif 'lumos' in file_name:
        return 'Lumos 5G'
    elif 'HSDPA' in file_name:
        return 'HSDPA'
    return 'Unknown'

combined_df['Network Type'] = combined_df['File'].apply(correct_categorize_trace)
#print(combined_df['Network Type'] )

# # Extract Slow, Medium, and Fast traces
slow_df = combined_df[combined_df['Network Type'] == 'Slow']
medium_df = combined_df[combined_df['Network Type'] == 'Medium']
fast_df = combined_df[combined_df['Network Type'] == 'Fast']
#combined_df = pd.concat([slow_df, medium_df, fast_df], ignore_index=True)

# Extract HSDPA traces and Lumos 5G traces dataframe
hsdpa_df = combined_df[combined_df['Network Type'] == 'HSDPA']
lumos_df = combined_df[combined_df['Network Type'] == 'Lumos 5G']
# combined_df = pd.concat([hsdpa_df, lumos_df], ignore_index=True)

# ALL
combined_df = pd.concat([slow_df, medium_df, fast_df, hsdpa_df, lumos_df], ignore_index=True)


# Calculate VMAF Change and QoE for each entry
combined_df['VMAF Change'] = combined_df['Average VMAF Smoothness']
combined_df['QoE'] = combined_df['Average VMAF Score']*vmaf_weights - (combined_df['Stall Ratio(%)']* rebuffer_weights) - (combined_df['Switch Ratio(%)'] * switch_weights)



def remove_outliers(df, columns):

    return df

def print_video_metrics(df, video_name):
    columns_to_check = ['QoE', 'Average VMAF Score', 'Stall Ratio(%)']
    network_types = ['Slow', 'Medium', 'Fast']
    
    all_metrics = []

    for network_type in network_types:
        df_filtered = df[df['Network Type'] == network_type]
        
        if df_filtered.empty:
            print(f"\nNo {network_type} traces found for {video_name}")
            continue

        df_clean = df_filtered #

        average_metrics = df_clean.groupby('Method').agg({
            'QoE': 'mean',
            'Average VMAF Score': 'mean',
            'Stall Ratio(%)': 'mean'
        }).round(2)

        print(f"\nAverage Metrics for {video_name} ({network_type}):")
        print(average_metrics)

        overall_averages = df_clean.agg({
            'QoE': 'mean',
            'Average VMAF Score': 'mean',
            'Stall Ratio(%)': 'mean'
        }).round(2)

        print(f"\nOverall Averages for {video_name} ({network_type}):")
        print(overall_averages)
        print("\n" + "="*50)

        all_metrics.append(average_metrics)

    # Calculate Overall metrics (average of Slow, Medium, Fast)
    if all_metrics:
        overall_metrics = pd.concat(all_metrics).groupby(level=0).mean().round(2)
        print(f"\nOverall Metrics for {video_name} (Average of Slow, Medium, Fast):")
        print(overall_metrics)

        overall_averages = overall_metrics.mean().round(2)
        print(f"\nOverall Averages for {video_name}:")
        print(overall_averages)
        print("\n" + "="*50)

    return all_metrics, overall_metrics

# Initialize dictionaries to store metrics for all networks
all_networks_metrics = {
    'Slow': [],
    'Medium': [],
    'Fast': [],
    'Overall': []
}

# Process each video separately
for file_path in file_paths:
    video_name = file_path.split('_performance_comparison.csv')[0]
    df = pd.read_csv(file_path)
    
    # 添加 Network Type 列
    df['Network Type'] = df['File'].apply(correct_categorize_trace)
    df["Stall Ratio(%)"] = filter_percentile(df["Stall Ratio(%)"])
    df=(df.dropna())
    # Calculate VMAF Change and QoE for each entry
    df['VMAF Change'] = df['Average VMAF Smoothness']
    df['QoE'] = df['Average VMAF Score']*vmaf_weights - (df['Stall Ratio(%)']* rebuffer_weights) - (df['Switch Ratio(%)'] * switch_weights)
    
    video_metrics, video_overall = print_video_metrics(df, video_name)
    
    # Add metrics to all_networks_metrics
    for i, network_type in enumerate(['Slow', 'Medium', 'Fast']):
        if i < len(video_metrics):
            all_networks_metrics[network_type].append(video_metrics[i])
    all_networks_metrics['Overall'].append(video_overall)

# Calculate and print overall metrics for all videos
print("\nOverall Metrics for All Videos:")
for network_type in ['Slow', 'Medium', 'Fast', 'Overall']:
    if all_networks_metrics[network_type]:
        overall = pd.concat(all_networks_metrics[network_type]).groupby(level=0).mean().round(2)
        print(f"\n{network_type} Network Condition:")
        print(overall)
        print("\nAverage across all methods:")
        print(overall.mean().round(2))
        print("="*50)











# Aggregate data by Network Category for plotting
trace_categories = combined_df['File'].str.extract(r'([A-Za-z]+)').squeeze().unique()
# print trace_categories 
print("Traces ->>",trace_categories)
combined_df['Network Category'] = combined_df['File'].str.extract(r'([A-Za-z]+)').squeeze()


# Box Plots for key metrics across different network types and methods
fig, axes = plt.subplots(2, 2, figsize=(24, 20))
metrics_to_boxplot = [
    ('Average VMAF Score', 'Score'),
    ('Stall Ratio(%)', 'Ratio'),
    ('Switch Ratio(%)', 'Ratio'),
    ('Average Bitrate(bps)', 'bps')
]
plt.rcParams.update({'font.size': 18})

for i, (metric, unit) in enumerate(metrics_to_boxplot):
    row, col = divmod(i, 2)
    sns.boxplot(data=combined_df, x='Method', y=metric, hue='Network Type',linewidth=3, ax=axes[row, col])
    axes[row, col].set_title(f'{metric} by Method and Network Type', fontsize=24, fontweight='bold')
    axes[row, col].set_xlabel('', fontsize=28)
    axes[row, col].set_ylabel(unit, fontsize=30)

                # 增大 x 軸刻度標籤的字體大小
    axes[row, col].tick_params(axis='x', which='major', labelsize=26)
    axes[row, col].tick_params(axis='y', which='major', labelsize=24)
plt.tight_layout()
plt.legend(title='Network Type', bbox_to_anchor=(1, 1),fontsize=20)
plt.show()
plt.savefig('box_plots.pdf')
plt.close()



# CDF plots for key metrics
metrics_to_cdf = [
    ('Average VMAF Score', 'Score'),
    ('Stall Ratio(%)', '%'),
    ('QoE','Score')
]


# 为每种方法定义不同的线型和颜色
method_styles = {
    'QAOCS': ('-', 'red'),
    'Constant-4': ('--', 'blue'),
    'GOP-4': ('-.', 'green'),
    'Segue': (':', 'purple')
}

metrics_to_cdf = [
    ('Average VMAF Score', 'Score'),
    ('Stall Ratio(%)', '%'),
    ('QoE', 'Score')
]

for network_type in combined_df['Network Type'].unique():
    fig, axes = plt.subplots(1, 3, figsize=(24, 12))
    
    for i, (metric, unit) in enumerate(metrics_to_cdf):
        ax = axes[i]
        
        for method in combined_df['Method'].unique():
            data = combined_df[(combined_df['Network Type'] == network_type) & (combined_df['Method'] == method)][metric]
            
            # 直接计算并绘制 CDF
            sorted_data = np.sort(data)
            yvals = np.arange(1, len(sorted_data) + 1) / float(len(sorted_data)) * 100
            
            linestyle, color = method_styles[method]
            ax.plot(sorted_data, yvals, label=method, linestyle=linestyle, color=color, linewidth=2)
        
        ax.set_title(f'CDF of {metric}', fontsize=24, fontweight='bold')
        ax.set_xlabel(metric, fontsize=24)
        ax.set_ylabel('CDF (%)', fontsize=24)
        ax.legend(fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # # 设置图例
    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05),
    #            ncol=4, fontsize=20, title='Methods', title_fontsize=22)

    plt.tight_layout()
    plt.suptitle(f'CDF Plots for {network_type} Network', fontsize=28, y=1.05)
    plt.savefig(f'cdf_plots_{network_type}.pdf', bbox_inches='tight')
    plt.close()

# # Scatter Plot for Average Bitrate vs Average VMAF Score
# plt.figure(figsize=(14, 12))
# sns.scatterplot(data=combined_df, x='Average Bitrate(bps)', y='Average VMAF Score', hue='Method', style='Network Type')
# plt.title('Average Bitrate vs Average VMAF Score by Method and Network Type',fontweight='bold',fontsize=18)
# plt.xlabel('Average Bitrate (bps)',fontsize=20)
# plt.ylabel('Average VMAF Score',fontsize=20)
# plt.legend(title='Method and Network Type', bbox_to_anchor=(1, 1))
# plt.tight_layout()
# plt.show()
# plt.savefig('scatter_plot.pdf')
# Scatter Plot for Average Bitrate vs Average VMAF Score

plt.figure(figsize=(14, 10))
# 定義每種網絡類型的標記
markers = {"Slow": "o", "Medium": "s", "Fast": "^", "HSDPA": "D", "Lumos 5G": "p"}

# 創建散點圖
sns.scatterplot(data=combined_df, 
                x='Average Bitrate(bps)', 
                y='Average VMAF Score', 
                hue='Method', 
                style='Network Type',
                markers=markers,
                size='Network Type',  # 使用 Network Type 來區分大小
                sizes=(89, 90),     # 設置點的大小範圍
                palette='deep')       # 使用 seaborn 的深色調色板

plt.title('Average Bitrate vs Average VMAF Score by Method and Network Type', fontweight='bold', fontsize=18)
plt.xlabel('Average Bitrate (bps)', fontsize=20)
plt.ylabel('Average VMAF Score', fontsize=20)

# 調整圖例
plt.legend(title='Method and Network Type', bbox_to_anchor=(.65, .8), loc='upper left', fontsize=24,markerscale=1.8)

# 調整刻度標籤大小
plt.tick_params(axis='both', which='major', labelsize=20)

plt.tight_layout()
plt.show()
plt.savefig('scatter_plot.pdf', bbox_inches='tight')

# Detailed Analysis Plots similar to reference figure
##categories = combined_df['Network Category'].unique()
categories = combined_df['Method'].unique()
stall_ratio = combined_df.groupby('Method')['Stall Ratio(%)'].mean().to_dict()
vmaf_scores = combined_df.groupby('Method')['Average VMAF Score'].mean().to_dict()
stall_ratio_errors = combined_df.groupby('Method')['Stall Ratio(%)'].std().to_dict()
vmaf_errors = combined_df.groupby('Method')['Average VMAF Score'].std().to_dict()
vmaf_vs_vmaf_change = combined_df.groupby('Method')['VMAF Change'].mean().to_dict()
vmaf_vs_vmaf_change_err = combined_df.groupby('Method')['VMAF Change'].std().to_dict()
qoe_dnn_vs_buffer = combined_df.groupby('Method')['QoE'].mean().to_dict()
qoe_dnn_vs_buffer_err = combined_df.groupby('Method')['QoE'].std().to_dict()
buffer_sizes = combined_df.groupby('Method')['Average Buffer State(s)'].mean().to_dict()
qoe_data = {category: combined_df[combined_df['Method'] == category]['QoE'].values for category in categories}

fig, axs = plt.subplots(2, 2, figsize=(18, 12))
markers = ['o', 's', 'D', '^', 'v', '>', '<', 'p']

# Plot (a) VMAF vs. Stall Ratio
for name, marker in zip(stall_ratio.keys(), markers):
    x = stall_ratio[name]
    y = vmaf_scores[name]
    x_err = stall_ratio_errors[name]
    y_err = vmaf_errors[name]
    axs[0,0].errorbar(x, y, xerr=x_err, yerr=y_err, label=name, marker=marker, capsize=5)

axs[0, 0].set_xlabel('Time Spent on Stall (%)')
axs[0, 0].set_ylabel('Video Quality (VMAF)',fontsize=20)
axs[0, 0].set_title('(a) VMAF vs. Stall Ratio',fontsize=20)
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
axs[0, 1].set_xlabel('Quality Smoothness (VMAF Change)',fontsize=20)
axs[0, 1].set_ylabel('Video Quality (VMAF)',fontsize=20)
axs[0, 1].set_title('(b) VMAF vs. VMAF Change')
axs[0, 1].legend()

# Plot (c) QoE_DNN vs. Buffer
for name, marker in zip(qoe_dnn_vs_buffer.keys(), markers):
    x = buffer_sizes[name]
    y = qoe_dnn_vs_buffer[name]
    x_err = combined_df[combined_df['Method'] == name]['Average Buffer State(s)'].std()
    y_err = qoe_dnn_vs_buffer_err[name]
    axs[1,0].errorbar(x, y, xerr=x_err, yerr=y_err, label=name, marker=marker, capsize=5)

axs[1, 0].set_xlabel('Buffer (s)',fontsize=20)
axs[1, 0].set_ylabel('QoE',fontsize=20)
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

axs[1, 1].set_xlabel('QoE',fontsize=20)
axs[1, 1].set_ylabel('CDF',fontsize=20)
axs[1, 1].set_title('(d) CDF of QoE')
axs[1, 1].legend()

# Adjust spacing between subplots
plt.tight_layout()
plt.show()
# Save the figure
plt.savefig('detailed_analysis_plots.pdf')


######################################## Improvement Bar Charts ########################################
# 计算改进幅度
def calculate_improvement(qaocs_value, baseline_value, metric):
    if metric == 'Stall Ratio(%)':
        return (baseline_value - qaocs_value) / baseline_value * 100
    else:
        return (qaocs_value - baseline_value) / baseline_value * 100

# 为每种网络类型计算改进幅度，包括整体改进
improvement_data = {network_type: {} for network_type in ['Slow', 'Medium', 'Fast', 'HSDPA', 'Lumos 5G', 'Overall']}
metrics = ['QoE', 'Average VMAF Score', 'Stall Ratio(%)']

for network_type in improvement_data.keys():
    if network_type == 'Overall':
        network_df = combined_df
    else:
        network_df = combined_df[combined_df['Network Type'] == network_type]
    
    qaocs_values = network_df[network_df['Method'] == 'QAOCS'][metrics].mean()
    
    for baseline in ['Constant-4', 'GOP-4', 'Segue']:
        baseline_values = network_df[network_df['Method'] == baseline][metrics].mean()
        improvement_data[network_type][baseline] = {
            metric: calculate_improvement(qaocs_values[metric], baseline_values[metric], metric)
            for metric in metrics
        }

# 绘制条形图
fig, axs = plt.subplots(3, 1, figsize=(18, 24))
metrics = ['QoE', 'Average VMAF Score', 'Stall Ratio(%)']
baselines = ['Constant-4', 'GOP-4', 'Segue']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for i, metric in enumerate(metrics):
    data = []
    for network_type in ['Slow', 'Medium', 'Fast', 'HSDPA', 'Lumos 5G', 'Overall']:
        data.append([improvement_data[network_type][baseline][metric] for baseline in baselines])
    
    x = np.arange(len(baselines))
    width = 0.13
    
    for j, d in enumerate(data):
        axs[i].bar(x + j*width, d, width, label=list(improvement_data.keys())[j], color=colors[j])
    
    axs[i].set_ylabel(f'Improvement in {metric} (%)', fontsize=14)
    axs[i].set_title(f'QAOCS Improvement in {metric}', fontsize=18,fontweight='bold')
    axs[i].set_xticks(x + width * 2.5)
    axs[i].set_xticklabels(baselines, fontsize=18)
    axs[i].legend(fontsize=12)
    axs[i].grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加数值标签
    for j, d in enumerate(data):
        for k, v in enumerate(d):
            axs[i].text(x[k] + j*width, v, f'{v:.1f}%', ha='center', va='bottom', fontsize=18)

plt.tight_layout()
plt.savefig('improvement_bar_charts_with_overall.pdf')
plt.show()


def plot_improvement_by_method_and_video_type(file_paths, video_type_mapping):
    metrics = ['QoE', 'Average VMAF Score', 'Stall Ratio(%)']
    baselines = ['Constant-4', 'GOP-4', 'Segue']
    network_types = ['Slow', 'Medium', 'Fast', 'Overall']

    # 提取所有视频的名称和类型
    video_names = [file_path.split('_performance_comparison.csv')[0] for file_path in file_paths]
    video_types = [video_type_mapping.get(name, 'Unknown') for name in video_names]

    # 存储所有视频的改进数据
    all_improvement_data = {}

    for file_path in file_paths:
        video_name = file_path.split('_performance_comparison.csv')[0]
        print(f"\nProcessing video: {video_name}")
        
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {df.columns}")
        
        # 添加 Network Type 列
        df['Network Type'] = df['File'].apply(correct_categorize_trace)
        df["Stall Ratio(%)"] = pd.to_numeric(df["Stall Ratio(%)"], errors='coerce')
        df = df.dropna()
        
        # Calculate VMAF Change and QoE for each entry
        df['VMAF Change'] = df['Average VMAF Smoothness']
        df['QoE'] = df['Average VMAF Score']*vmaf_weights - (df['Stall Ratio(%)']* rebuffer_weights) - (df['Switch Ratio(%)'] * switch_weights)
        
        # 计算每种网络类型的平均值
        video_data = {}
        for nt in network_types:
            if nt != 'Overall':
                video_data[nt] = df[df['Network Type'] == nt].groupby('Method')[metrics].mean()
            else:
                video_data[nt] = df.groupby('Method')[metrics].mean()
        
        # 计算改进幅度
        improvement_data = {nt: {} for nt in network_types}
        for nt in network_types:
            if not video_data[nt].empty:
                if 'QAOCS' in video_data[nt].index:
                    qaocs_values = video_data[nt].loc['QAOCS']
                    for baseline in baselines:
                        if baseline in video_data[nt].index:
                            baseline_values = video_data[nt].loc[baseline]
                            improvement_data[nt][baseline] = {
                                metric: calculate_improvement(qaocs_values[metric], baseline_values[metric], metric)
                                for metric in metrics if metric in qaocs_values and metric in baseline_values
                            }
                else:
                    print(f"QAOCS not found in {nt} network type")
        
        all_improvement_data[video_name] = improvement_data

    # 计算每个视频类型的平均改进幅度
    avg_improvement_by_type = {vtype: {baseline: {metric: [] for metric in metrics} for baseline in baselines} for vtype in set(video_types)}
    
    for video, vtype in zip(video_names, video_types):
        for nt in network_types:
            for baseline in baselines:
                for metric in metrics:
                    if nt in all_improvement_data[video] and baseline in all_improvement_data[video][nt]:
                        improvement = all_improvement_data[video][nt][baseline].get(metric, 0)
                        avg_improvement_by_type[vtype][baseline][metric].append(improvement)

    # 绘图
    fig, axs = plt.subplots(len(metrics), 1, figsize=(15, 5*len(metrics)))
    
    for idx, metric in enumerate(metrics):
        x = np.arange(len(set(video_types)))
        width = 0.25
        
        for i, baseline in enumerate(baselines):
            data = [np.mean(avg_improvement_by_type[vtype][baseline][metric]) for vtype in set(video_types)]
            axs[idx].bar(x + i*width, data, width, label=baseline)
        
        axs[idx].set_ylabel('Improvement (%)', fontsize=16)
        axs[idx].set_title(f'QAOCS Improvement in {metric}', fontsize=18, fontweight='bold')
        axs[idx].set_xticks(x + width)
        axs[idx].set_xticklabels(list(set(video_types)), fontsize=18)
        axs[idx].legend(fontsize=16)
        axs[idx].grid(axis='y', linestyle='--', alpha=0.7)
        
        # 添加数值标签
        for i, baseline in enumerate(baselines):
            data = [np.mean(avg_improvement_by_type[vtype][baseline][metric]) for vtype in set(video_types)]
            for j, v in enumerate(data):
                if v >= 0:
                    va = 'bottom'
                    y = v
                else:
                    va = 'top'
                    y = v
                axs[idx].text(x[j] + i*width, y, f'{v:.1f}%', ha='center', va=va, fontsize=15)

    plt.tight_layout()
    plt.savefig('improvement_by_method_and_video_type.pdf')
    plt.show()

# 调用函数
video_type_mapping = {
    'BBB_360p24': 'Animation',
    'TOS_360p24': 'Movies',
    'LOL_3D': 'Movies',
    'underwater': 'Documentary',
    'video_game': 'Animation',
    'sport_long_take': 'Sports',
    'sport_highlight': 'Sports'
}
plot_improvement_by_method_and_video_type(file_paths, video_type_mapping)



######################################## Improvement Bar Charts Video  ########################################

def plot_improvement_for_video(file_path, video_type_mapping):
    video_name = file_path.split('_performance_comparison.csv')[0]
    video_type = video_type_mapping.get(video_name, 'Unknown')
    
    metrics = ['QoE', 'Average VMAF Score', 'Stall Ratio(%)']
    baselines = ['Constant-4', 'GOP-4', 'Segue']
    network_types = ['Slow', 'Medium', 'Fast', 'Overall']

    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    print(f"Processing video: {video_name}")
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {df.columns}")
    print(df.dtypes)
    
    # 添加 Network Type 列
    df['Network Type'] = df['File'].apply(correct_categorize_trace)
    df["Stall Ratio(%)"] = pd.to_numeric(df["Stall Ratio(%)"], errors='coerce')
    df = df.dropna()
    
    # Calculate VMAF Change and QoE for each entry
    df['VMAF Change'] = df['Average VMAF Smoothness']
    df['QoE'] = df['Average VMAF Score']*vmaf_weights - (df['Stall Ratio(%)']* rebuffer_weights) - (df['Switch Ratio(%)'] * switch_weights)
    
    # 计算每种网络类型的平均值
    video_data = {}
    for nt in network_types:
        if nt != 'Overall':
            video_data[nt] = df[df['Network Type'] == nt].groupby('Method')[metrics].mean()
        else:
            video_data[nt] = df.groupby('Method')[metrics].mean()
        
        print(f"\nNetwork type: {nt}")
        print(video_data[nt])
    
    # 计算改进幅度
    improvement_data = {nt: {} for nt in network_types}
    for nt in network_types:
        if not video_data[nt].empty:
            if 'QAOCS' in video_data[nt].index:
                qaocs_values = video_data[nt].loc['QAOCS']
                for baseline in baselines:
                    if baseline in video_data[nt].index:
                        baseline_values = video_data[nt].loc[baseline]
                        improvement_data[nt][baseline] = {
                            metric: calculate_improvement(qaocs_values[metric], baseline_values[metric], metric)
                            for metric in metrics if metric in qaocs_values and metric in baseline_values
                        }
            else:
                print(f"QAOCS not found in {nt} network type")
    # 绘图
    fig, axs = plt.subplots(2, 2, figsize=(20, 15))
    axs = axs.flatten()
    
    for idx, nt in enumerate(network_types):
        data = []
        for baseline in baselines:
            if nt in improvement_data and baseline in improvement_data[nt]:
                data.append([improvement_data[nt][baseline][metric] for metric in metrics])
            else:
                data.append([0, 0, 0])  # 如果没有数据，用0填充
        
        x = np.arange(len(metrics))
        width = 0.25
        
        for i, baseline in enumerate(baselines):
            axs[idx].bar(x + i*width, data[i], width, label=baseline)
        
        axs[idx].set_ylabel('Improvement (%)', fontsize=16)
        axs[idx].set_title(f'{nt} Network Condition', fontsize=18, fontweight='bold')
        axs[idx].set_xticks(x + width)
        axs[idx].set_xticklabels(metrics, fontsize=18)
        axs[idx].legend(fontsize=16)
        axs[idx].grid(axis='y', linestyle='--', alpha=0.7)
        
        # 添加数值标签
        for i, d in enumerate(data):
            for j, v in enumerate(d):
                if v >= 0:
                    va = 'bottom'
                    y = v
                else:
                    va = 'top'
                    y = v
                axs[idx].text(x[j] + i*width, y, f'{v:.1f}%', ha='center', va=va, fontsize=15)
    
    fig.suptitle(f'QAOCS Improvement for {video_name} ({video_type})', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'improvement_bar_chart_{video_name}.pdf')
    plt.show()

# 定义视频类型映射
video_type_mapping = {
    'BBB_360p24': 'Animation',
    'TOS_360p24': 'Movies',
    'LOL_3D': 'Movies',
    'underwater': 'Documentary',
    'video_game': 'Animation',
    'sport_long_take': 'Sports',
    'sport_highlight': 'Sports'
}

# 文件路径
file_paths = [
    'BBB_360p24_performance_comparison.csv',
    'TOS_360p24_performance_comparison.csv',
    'LOL_3D_performance_comparison.csv',
    'underwater_performance_comparison.csv',
    'video_game_performance_comparison.csv',
    'sport_long_take_performance_comparison.csv',
    'sport_highlight_performance_comparison.csv',
]

# 为每个视频生成图表
for file_path in file_paths:
    plot_improvement_for_video(file_path, video_type_mapping)

######################################## Improvement Bar Charts Network  ########################################

def plot_improvement_for_network(network_type='Slow'):
    metrics = ['QoE', 'Average VMAF Score', 'Stall Ratio(%)']
    baselines = ['Constant-4', 'GOP-4', 'Segue']
    
    data = []
    for baseline in baselines:
        data.append([improvement_data[network_type][baseline][metric] for metric in metrics])
    
    x = np.arange(len(metrics))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, baseline in enumerate(baselines):
        ax.bar(x + i*width, data[i], width, label=baseline)
    
    ax.set_ylabel('Improvement (%)', fontsize=14)
    ax.set_title(f'QAOCS Improvement for {network_type} Network Condition', fontsize=18, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics, fontsize=16)
    ax.legend(fontsize=16)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加数值标签
    for i, d in enumerate(data):
        for j, v in enumerate(d):
            ax.text(x[j] + i*width, v, f'{v:.1f}%', ha='center', va='bottom', fontsize=15)
    
    plt.tight_layout()
    plt.savefig(f'improvement_bar_chart_{network_type}.pdf')
    plt.show()

for network_type in ['Slow', 'Medium', 'Fast', 'HSDPA', 'Lumos 5G', 'Overall']:
    plot_improvement_for_network(network_type)