import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os 
from pathlib import Path
# Constants
VMAF_WEIGHT = 0.13
SWITCH_WEIGHT = 0.75
#REBUFFER_WEIGHT = 0.2
REBUFFER_WEIGHT = 0.2

# File paths
FILE_PATHS = [
    #'./thesis_csv_result/BBB_360p24_performance_comparison.csv',
    #'./thesis_csv_result/TOS_360p24_performance_comparison.csv',
    './Training_output_abr_without_qaocs/LOL_3D_performance_comparison.csv',
    #'./thesis_csv_result/underwater_performance_comparison.csv',
    './Training_output_abr_without_qaocs/video_game_performance_comparison.csv',
    #'./thesis_csv_result/sport_long_take_performance_comparison.csv',
    './Training_output_abr_without_qaocs/sport_highlight_performance_comparison.csv',
]

VIDEO_TYPE_MAPPING = {
    'BBB_360p24': 'Animation',
    'TOS_360p24': 'Movies',
    'LOL_3D': 'Movies',
    'underwater': 'Documentary',
    'video_game': 'Animation',
    'sport_long_take': 'Sports',
    'sport_highlight': 'Sports'
}

# Helper functions
def filter_percentile(df, lower=5, upper=90):
    return df[(df >= df.quantile(lower / 100)) & (df <= df.quantile(upper / 100))]

def categorize_trace(file_name):
    file_name = file_name.lower() 
    if 'oboe' in file_name:
        return 'Slow'
    elif 'fcc18' in file_name or 'hsr' in file_name:
        return 'Medium'
    elif 'ghent' in file_name or 'lab' in file_name:
        return 'Fast'
    #elif 'lumos' in file_name:
        #return 'Lumos 5G'
    elif 'lumos' in file_name:
        return 'Fast'
    elif 'HSDPA' in file_name:
        return 'HSDPA'
    return 'Unknown'

def calculate_improvement(qaocs_value, baseline_value, metric):
    if metric == 'Stall Ratio(%)':
        return (baseline_value - qaocs_value) / baseline_value * 100
    elif metric == 'Switch Ratio(%)':
         return (baseline_value - qaocs_value) / baseline_value * 100
    else:
        return (qaocs_value - baseline_value) / baseline_value * 100

# Data loading and preprocessing
def load_and_preprocess_data(file_paths):
    dataframes = [pd.read_csv(file) for file in file_paths]
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    combined_df["Stall Ratio(%)"] = filter_percentile(combined_df["Stall Ratio(%)"])
    #combined_df["Switch Ratio(%)"] = filter_percentile(combined_df["Switch Ratio(%)"])
    combined_df = combined_df.dropna()
    
    combined_df['Network Type'] = combined_df['File'].apply(categorize_trace)
    combined_df['VMAF Change'] = combined_df['Average VMAF Smoothness']
    combined_df['QoE'] = (combined_df['Average VMAF Score'] * VMAF_WEIGHT - 
                          (combined_df['Stall Ratio(%)'] * REBUFFER_WEIGHT) - 
                          (combined_df['Switch Ratio(%)'] * SWITCH_WEIGHT))
    
    return combined_df

# Plotting functions
def plot_box_plots(combined_df):
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
        sns.boxplot(data=combined_df, x='Method', y=metric, hue='Network Type', linewidth=3, ax=axes[row, col])
        axes[row, col].set_title(f'{metric} by Method and Network Type', fontsize=24, fontweight='bold')
        axes[row, col].set_xlabel('', fontsize=28)
        axes[row, col].set_ylabel(unit, fontsize=30)
        axes[row, col].tick_params(axis='x', which='major', labelsize=26)
        axes[row, col].tick_params(axis='y', which='major', labelsize=24)

    plt.tight_layout()
    plt.legend(title='Network Type', bbox_to_anchor=(1, 1), fontsize=20)
    plt.savefig('./thesis_csv_result/box_plots.pdf')
    plt.close()

def plot_cdf_plots(combined_df):
    metrics_to_cdf = [
        ('Average VMAF Score', 'Score'),
        ('Stall Ratio(%)', '%'),
        ('QoE', 'Score')
    ]
    method_styles = {
        'QAOCS': ('-', 'red'),
        'Constant-4': ('--', 'blue'),
        'GOP-4': ('-.', 'green'),
        'Segue': (':', 'purple')
    }

    for network_type in combined_df['Network Type'].unique():
        fig, axes = plt.subplots(1, 3, figsize=(24, 12))
        
        for i, (metric, unit) in enumerate(metrics_to_cdf):
            ax = axes[i]
            
            for method in combined_df['Method'].unique():
                data = combined_df[(combined_df['Network Type'] == network_type) & (combined_df['Method'] == method)][metric]
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
        
        plt.tight_layout()
        plt.suptitle(f'CDF Plots for {network_type} Network', fontsize=28, y=1.05)
        plt.savefig(f'./thesis_csv_result/cdf_plots_{network_type}.pdf', bbox_inches='tight')
        plt.close()

def plot_scatter_plot(combined_df):
    plt.figure(figsize=(14, 10))
    markers = {"Slow": "o", "Medium": "s", "Fast": "^", "HSDPA": "D", "Lumos 5G": "p"}

    sns.scatterplot(data=combined_df, 
                    x='Average Bitrate(bps)', 
                    y='Average VMAF Score', 
                    hue='Method', 
                    style='Network Type',
                    markers=markers,
                    size='Network Type',
                    sizes=(89, 90),
                    palette='deep')

    plt.title('Average Bitrate vs Average VMAF Score by Method and Network Type', fontweight='bold', fontsize=18)
    plt.xlabel('Average Bitrate (bps)', fontsize=20)
    plt.ylabel('Average VMAF Score', fontsize=20)
    plt.legend(title='Method and Network Type', bbox_to_anchor=(.65, .8), loc='upper left', fontsize=24, markerscale=1.8)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tight_layout()
    plt.savefig('./thesis_csv_result/scatter_plot.pdf', bbox_inches='tight')
    plt.close()

# def plot_improvement_bar_charts(combined_df):
#     improvement_data = calculate_improvement_data(combined_df)
#     metrics = ['QoE', 'Average VMAF Score', 'Stall Ratio(%)']
#     baselines = ['Constant-4', 'GOP-4', 'Segue']
#     colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

#     fig, axs = plt.subploAe in ['Slow', 'Medium', 'Fast']:    
#             data.append([improvement_data[network_type][baseline][metric] for baseline in baselines])
        
#         x = np.arange(len(baselines))
#         width = 0.13
        
#         for j, d in enumerate(data):
#             axs[i].bar(x + j*width, d, width, label=list(improvement_data.keys())[j], color=colors[j])
        
#         axs[i].set_ylabel(f'Improvement in {metric} (%)', fontsize=14)
#         axs[i].set_title(f'QAOCS Improvement in {metric}', fontsize=18, fontweight='bold')
#         axs[i].set_xticks(x + width * 2.5)
#         axs[i].set_xticklabels(baselines, fontsize=18)
#         axs[i].legend(fontsize=12)
#         axs[i].grid(axis='y', linestyle='--', alpha=0.7)
        
#         for j, d in enumerate(data):
#             for k, v in enumerate(d):
#                 axs[i].text(x[k] + j*width, v, f'{v:.1f}%', ha='center', va='bottom', fontsize=18)

#     plt.tight_layout()
#     plt.savefig('./thesis_csv_result/improvement_bar_charts_with_overall.pdf')
#     plt.close()

def calculate_improvement_data(combined_df):
    improvement_data = {network_type: {} for network_type in ['Slow', 'Medium', 'Fast', 'HSDPA', 'Lumos 5G', 'Overall']}
    metrics = ['QoE', 'Average VMAF Score', 'Stall Ratio(%)']

    method_pairs = [
    ('QAOCS-BBA', 'SEGUE-BBA'),
    ('QAOCS-MPC', 'SEGUE-MPC'),
    ('QAOCS-BOLA', 'SEGUE-BOLA'),
    ('QAOCS-RBA', 'SEGUE-RBA'),
    ('QAOCS-PENSIEVE', 'SEGUE-PENSIEVE')
    ]

    for network_type in improvement_data.keys():
        if network_type == 'Overall':
            network_df = combined_df
        else:
            network_df = combined_df[combined_df['Network Type'] == network_type]
        
        # qaocs_values = network_df[network_df['Method'] == 'QAOCS'][metrics].mean()
        
        # for baseline in ['Constant-4', 'GOP-4', 'Segue']:
        #     baseline_values = network_df[network_df['Method'] == baseline][metrics].mean()
        #     improvement_data[network_type][baseline] = {
        #         metric: calculate_improvement(qaocs_values[metric], baseline_values[metric], metric)
        #         for metric in metrics
        #     }
        for qaocs_method, baseline_method in method_pairs:
            qaocs_df = network_df[network_df['Method'] == qaocs_method]
            baseline_df = network_df[network_df['Method'] == baseline_method]

            if not qaocs_df.empty and not baseline_df.empty:
                qaocs_values = qaocs_df[metrics].mean()
                baseline_values = baseline_df[metrics].mean()

                improvement_data[network_type][baseline_method] = {
                    metric: calculate_improvement(qaocs_values[metric], baseline_values[metric], metric)
                    for metric in metrics
                }


    return improvement_data

# def plot_improvement_by_method_and_video_type(combined_df, video_type_mapping):
#     metrics = ['QoE', 'Average VMAF Score', 'Stall Ratio(%)']
#     baselines = ['Constant-4', 'GOP-4', 'Segue']
#     network_types = ['Slow', 'Medium', 'Fast', 'Overall']

#     video_names = [file_path.split('_performance_comparison.csv')[0] for file_path in FILE_PATHS]
#     video_types = [video_type_mapping.get(name, 'Unknown') for name in video_names]

#     all_improvement_data = {}

#     for file_path in FILE_PATHS:
#         video_name = file_path.split('_performance_comparison.csv')[0]
#         df = pd.read_csv(file_path)
        
#         df['Network Type'] = df['File'].apply(categorize_trace)
#         df["Stall Ratio(%)"] = pd.to_numeric(df["Stall Ratio(%)"], errors='coerce')
#         df = df.dropna()
        
#         df['VMAF Change'] = df['Average VMAF Smoothness']
#         df['QoE'] = df['Average VMAF Score']*VMAF_WEIGHT - (df['Stall Ratio(%)']*REBUFFER_WEIGHT) - (df['Switch Ratio(%)']*SWITCH_WEIGHT)
        
#         video_data = {}
#         for nt in network_types:
#             if nt != 'Overall':
#                 video_data[nt] = df[df['Network Type'] == nt].groupby('Method')[metrics].mean()
#             else:
#                 video_data[nt] = df.groupby('Method')[metrics].mean()
        
#         improvement_data = {nt: {} for nt in network_types}
#         for nt in network_types:
#             if not video_data[nt].empty and 'QAOCS' in video_data[nt].index:
#                 qaocs_values = video_data[nt].loc['QAOCS']
#                 for baseline in baselines:
#                     if baseline in video_data[nt].index:
#                         baseline_values = video_data[nt].loc[baseline]
#                         improvement_data[nt][baseline] = {
#                             metric: calculate_improvement(qaocs_values[metric], baseline_values[metric], metric)
#                             for metric in metrics if metric in qaocs_values and metric in baseline_values
#                         }
        
#         all_improvement_data[video_name] = improvement_data

#     avg_improvement_by_type = {vtype: {baseline: {metric: [] for metric in metrics} for baseline in baselines} for vtype in set(video_types)}
    
#     for video, vtype in zip(video_names, video_types):
#         for nt in network_types:
#             for baseline in baselines:
#                 for metric in metrics:
#                     if nt in all_improvement_data[video] and baseline in all_improvement_data[video][nt]:
#                         improvement = all_improvement_data[video][nt][baseline].get(metric, 0)
#                         avg_improvement_by_type[vtype][baseline][metric].append(improvement)

#     fig, axs = plt.subplots(len(metrics), 1, figsize=(15, 5*len(metrics)))
    
#     for idx, metric in enumerate(metrics):
#         x = np.arange(len(set(video_types)))
#         width = 0.25
        
#         for i, baseline in enumerate(baselines):
#             data = [np.mean(avg_improvement_by_type[vtype][baseline][metric]) for vtype in set(video_types)]
#             axs[idx].bar(x + i*width, data, width, label=baseline)
        
#         axs[idx].set_ylabel('Improvement (%)', fontsize=16)
#         axs[idx].set_title(f'QAOCS Improvement in {metric}', fontsize=18, fontweight='bold')
#         axs[idx].set_xticks(x + width)
#         axs[idx].set_xticklabels(list(set(video_types)), fontsize=18)
#         axs[idx].legend(fontsize=16)
#         axs[idx].grid(axis='y', linestyle='--', alpha=0.7)
        
#         for i, baseline in enumerate(baselines):
#             data = [np.mean(avg_improvement_by_type[vtype][baseline][metric]) for vtype in set(video_types)]
#             for j, v in enumerate(data):
#                 y = v if v >= 0 else v
#                 axs[idx].text(x[j] + i*width, y, f'{v:.1f}%', ha='center', va='bottom' if v >= 0 else 'top', fontsize=15)

#     plt.tight_layout()
#     plt.savefig('./thesis_csv_result/improvement_by_method_and_video_type.pdf')
#     plt.close()
# def plot_improvement_bar_charts(combined_df):
#     improvement_data = calculate_improvement_data(combined_df)
#     metrics = ['QoE', 'Average VMAF Score', 'Stall Ratio(%)']

#     fig, axs = plt.subplots(3, 1, figsize=(14, 18))
#     methods = list(improvement_data.keys())
#     x = np.arange(len(methods))

#     for i, metric in enumerate(metrics):
#         values = [improvement_data[m][metric] for m in methods]
#         axs[i].bar(x, values, color='skyblue')
#         axs[i].set_xticks(x)
#         axs[i].set_xticklabels(methods, rotation=45, fontsize=12)
#         axs[i].set_ylabel(f'{metric} Improvement (%)', fontsize=14)
#         axs[i].set_title(f'QAOCS Improvement over Corresponding Baseline: {metric}', fontsize=16, fontweight='bold')
#         for j, val in enumerate(values):
#             axs[i].text(j, val, f'{val:.1f}%', ha='center', va='bottom' if val > 0 else 'top', fontsize=12)
#         axs[i].grid(axis='y', linestyle='--', alpha=0.7)

#     plt.tight_layout()
#     plt.savefig('./Training_output_abr_without_qaocs/qaocs_pairwise_improvement.pdf')
#     plt.close()
def plot_improvement_bar_charts(combined_df):
    improvement_data = calculate_improvement_data(combined_df)
    metrics = ['QoE', 'Average VMAF Score', 'Stall Ratio(%)']
    method_pairs = [
        ('QAOCS-BBA', 'SEGUE-BBA'),
        ('QAOCS-MPC', 'SEGUE-MPC'),
        ('QAOCS-BOLA', 'SEGUE-BOLA'),
        ('QAOCS-RBA', 'SEGUE-RBA'),
        ('QAOCS-PENSIEVE', 'SEGUE-PENSIEVE')
    ]
    network_types = ['Slow', 'Medium', 'Fast', 'HSDPA', 'Lumos 5G', 'Overall']
    
    baseline_colors = {
    'SEGUE-BBA': '#1f77b4',       # 藍色
    'SEGUE-MPC': '#ff7f0e',       # 橘色
    'SEGUE-BOLA': '#2ca02c',      # 綠色
    'SEGUE-RBA': '#d62728',       # 紅色
    'SEGUE-PENSIEVE': '#9467bd'   # 紫色
    }
    label_name = {
        'SEGUE-BBA': 'BBA',
        'SEGUE-MPC': 'RMPC',
        'SEGUE-BOLA': 'BOLA',
        'SEGUE-RBA': 'RBA',
        'SEGUE-PENSIEVE': 'PENSIEVE'
    }
    for net in network_types:
        if net not in improvement_data:
            continue  # skip missing entries
        data = improvement_data[net]
        # fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        # for i, metric in enumerate(metrics):
        #     values = []
        #     labels = []
        #     for qaocs_method, baseline in method_pairs:
        #         if baseline in data and metric in data[baseline]:
        #             values.append(data[baseline][metric])
        #             labels.append(baseline)
        #     colors = [baseline_colors.get(label, 'gray') for label in labels]
        #     axs[i].bar(labels, values, color= colors)
        #     axs[i].set_title(f'{net} - {metric}', fontsize=14, fontweight='bold')
        #     axs[i].set_ylabel('Improvement (%)', fontsize=12)
        #     axs[i].tick_params(axis='x', rotation=30)
        #     for j, val in enumerate(values):
        #         axs[i].text(j, val, f'{val:.1f}%', ha='center', va='bottom' if val > 0 else 'top', fontsize=10)
        #     axs[i].grid(axis='y', linestyle='--', alpha=0.7)
        # #print("Slow methods:", combined_df[combined_df['Network Type'] == 'Slow']['Method'].value_counts())
        # #print("medium:", combined_df[combined_df['Network Type'] == 'Medium']['Method'].value_counts())
        # plt.tight_layout()
        # plt.savefig(f'./Training_output_abr_without_qaocs/pairwise_improvement_{net}.png')
        # plt.close()
        
        # 只挑該網路條件下真的有數值的 baseline，並保持既定順序
        baselines = [b for _, b in method_pairs if (b in data and all(m in data[b] for m in metrics))]
        if not baselines:
            continue

        x = np.arange(len(metrics))  # 三個群組
        width = min(0.18, 0.8 / max(1, len(baselines)))  # 動態寬度，避免重疊
        offsets = (np.arange(len(baselines)) - (len(baselines)-1)/2) * (width + 0.02)

        plt.figure(figsize=(9, 5))

        for bi, b in enumerate(baselines):
            y = [float(data[b][m]) for m in metrics]
            bars = plt.bar(x + offsets[bi], y, width=width,
                           label=label_name.get(b, b),
                           color=baseline_colors.get(b, 'gray'))
            # 標上百分比
            for rect, val in zip(bars, y):
                plt.text(rect.get_x() + rect.get_width()/2,
                         val + (0.8 if val >= 0 else -0.8),
                         f'{val:.1f}%', ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)

        plt.xticks(x, metrics,fontsize=20)
        plt.ylabel('Improvement (%)',fontsize=20)
        plt.title(f'QAOCS Improvement for {net} Network Condition', fontweight='bold', fontsize=20)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'./Training_output_abr_without_qaocs/pairwise_improvement_{net}.pdf', dpi=150)
        plt.close()


def plot_improvement_bar_charts_3x1(combined_df, networks=('Slow','Medium','Fast')):
    improvement_data = calculate_improvement_data(combined_df)
    metrics = ['QoE', 'Average VMAF Score', 'Stall Ratio(%)']
    method_pairs = [
        ('QAOCS-BBA', 'SEGUE-BBA'),
        ('QAOCS-MPC', 'SEGUE-MPC'),
        ('QAOCS-BOLA', 'SEGUE-BOLA'),
        ('QAOCS-RBA', 'SEGUE-RBA'),
        ('QAOCS-PENSIEVE', 'SEGUE-PENSIEVE')
    ]
    network_types = ['Slow', 'Medium', 'Fast', 'HSDPA', 'Lumos 5G', 'Overall']
    
    baseline_colors = {
        'SEGUE-BBA': '#1f77b4',       # 藍色
        'SEGUE-MPC': '#ff7f0e',       # 橘色
        'SEGUE-BOLA': '#2ca02c',      # 綠色
        'SEGUE-RBA': '#d62728',       # 紅色
        'SEGUE-PENSIEVE': '#9467bd'   # 紫色
    }
    label_name = {
        'SEGUE-BBA': 'BBA',
        'SEGUE-MPC': 'RMPC',
        'SEGUE-BOLA': 'BOLA',
        'SEGUE-RBA': 'RBA',
        'SEGUE-PENSIEVE': 'PENSIEVE'
    }

    # 只取你要的 3 個網路條件（預設 Slow/Medium/Fast），且要在你的 network_types 範圍內
    nets = [n for n in networks if n in network_types and n in improvement_data]
    if len(nets) == 0:
        print('[plot] no valid networks to plot.')
        return

    # 建 3x1 圖
    fig, axs = plt.subplots(len(nets), 1, figsize=(10, 12), sharex=True)
    if len(nets) == 1:
        axs = [axs]  # 統一成可迭代

    # 為全域 legend 收集出現過的 baseline
    baselines_in_any = set()

    x = np.arange(len(metrics))  # 三個群組
    for idx, net in enumerate(nets):
        data = improvement_data[net]
        ax = axs[idx]

        # 只挑該網路條件下真的有數值的 baseline，並保持既定順序
        baselines = [b for _, b in method_pairs if (b in data and all(m in data[b] for m in metrics))]
        if not baselines:
            ax.text(0.5, 0.5, f'No valid baselines for {net}', ha='center', va='center', fontsize=10)
            ax.set_axis_off()
            continue

        for b in baselines:
            baselines_in_any.add(b)

        width = min(0.18, 0.8 / max(1, len(baselines)))  # 動態寬度，避免重疊
        offsets = (np.arange(len(baselines)) - (len(baselines)-1)/2) * (width + 0.02)

        for bi, b in enumerate(baselines):
            y = [float(data[b][m]) for m in metrics]
            bars = ax.bar(
                x + offsets[bi], y, width=width,
                label=label_name.get(b, b),
                color=baseline_colors.get(b, 'gray')
            )
            # 標上百分比
            for rect, val in zip(bars, y):
                ax.text(rect.get_x() + rect.get_width()/2,
                        val + (0.8 if val >= 0 else -0.8),
                        f'{val:.1f}%', ha='center',
                        va='bottom' if val >= 0 else 'top', fontsize=9)
                
        ax.legend(loc='upper left', fontsize=10, frameon=False)
        ax.set_ylim(0, 100)
        ax.set_ylabel('Improvement (%)', fontsize=12)
        ax.set_title(f'QAOCS Improvement for {net} Network Condition', fontweight='bold', fontsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.tick_params(axis='y', labelsize=11)

    # 共用 X 軸（只在最下排畫）
    axs[-1].set_xticks(x)
    axs[-1].set_xticklabels(metrics, fontsize=13)



    net_combo = "_".join([n.replace(' ', '') for n in nets])
    out_path = f'./Training_output_abr_without_qaocs/pairwise_improvement_3x1_{net_combo}.pdf'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot] Saved -> {out_path}")

def plot_improvement_for_video(file_path, video_type_mapping):
    #video_name = file_path.split('_performance_comparison.csv')[0]
    video_name = os.path.basename(file_path).replace('_performance_comparison.csv', '')
    video_type = video_type_mapping.get(video_name, 'Unknown')
    
    metrics = ['QoE', 'Average VMAF Score', 'Stall Ratio(%)']
    baselines = ['Constant-4', 'GOP-4', 'Segue']
    network_types = ['Slow', 'Medium', 'Fast', 'Overall']

    df = pd.read_csv(file_path)
    
    df['Network Type'] = df['File'].apply(categorize_trace)
    df["Stall Ratio(%)"] = pd.to_numeric(df["Stall Ratio(%)"], errors='coerce')
    df = df.dropna()
    
    df['VMAF Change'] = df['Average VMAF Smoothness']
    df['QoE'] = df['Average VMAF Score']*VMAF_WEIGHT - (df['Stall Ratio(%)']*REBUFFER_WEIGHT) - (df['Switch Ratio(%)']*SWITCH_WEIGHT)
    
    video_data = {}
    for nt in network_types:
        if nt != 'Overall':
            video_data[nt] = df[df['Network Type'] == nt].groupby('Method')[metrics].mean()
        else:
            video_data[nt] = df.groupby('Method')[metrics].mean()
    
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
    
    fig, axs = plt.subplots(2, 2, figsize=(20, 15))
    axs = axs.flatten()
    
    for idx, nt in enumerate(network_types):
        data = []
        for baseline in baselines:
            if nt in improvement_data and baseline in improvement_data[nt]:
                data.append([improvement_data[nt][baseline][metric] for metric in metrics])
            else:
                data.append([0, 0, 0])  # If no data, fill with zeros
        
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
        
        for i, d in enumerate(data):
            for j, v in enumerate(d):
                y = v if v >= 0 else v
                axs[idx].text(x[j] + i*width, y, f'{v:.1f}%', ha='center', va='bottom' if v >= 0 else 'top', fontsize=15)
    
    fig.suptitle(f'QAOCS Improvement for {video_name} ({video_type})', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'./Training_output_abr_without_qaocs/improvement_bar_chart_{video_name}.pdf')
    plt.close()

def plot_improvement_for_network(improvement_data, network_type='Slow'):
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
    
    for i, d in enumerate(data):
        for j, v in enumerate(d):
            ax.text(x[j] + i*width, v, f'{v:.1f}%', ha='center', va='bottom', fontsize=15)
    plt.tight_layout()
    plt.savefig(f'./Training_output_abr_without_qaocs/improvement_bar_chart_{network_type}.pdf')
    plt.close()


def plot_detailed_analysis(combined_df):
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
    markers = ['o', 's', 'D', '^']

    # Plot (a) VMAF vs. Stall Ratio
    for name, marker in zip(stall_ratio.keys(), markers):
        x = stall_ratio[name]
        y = vmaf_scores[name]
        x_err = stall_ratio_errors[name]
        y_err = vmaf_errors[name]
        axs[0,0].errorbar(x, y, xerr=x_err, yerr=y_err, label=name, marker=marker, capsize=5)

    axs[0, 0].set_xlabel('Time Spent on Stall (%)')
    axs[0, 0].set_ylabel('Video Quality (VMAF)', fontsize=20)
    axs[0, 0].set_title('(a) VMAF vs. Stall Ratio', fontsize=20)
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
    axs[0, 1].set_xlabel('Quality Smoothness (VMAF Change)', fontsize=20)
    axs[0, 1].set_ylabel('Video Quality (VMAF)', fontsize=20)
    axs[0, 1].set_title('(b) VMAF vs. VMAF Change')
    axs[0, 1].legend()

    # Plot (c) QoE_DNN vs. Buffer
    for name, marker in zip(qoe_dnn_vs_buffer.keys(), markers):
        x = buffer_sizes[name]
        y = qoe_dnn_vs_buffer[name]
        x_err = combined_df[combined_df['Method'] == name]['Average Buffer State(s)'].std()
        y_err = qoe_dnn_vs_buffer_err[name]
        axs[1,0].errorbar(x, y, xerr=x_err, yerr=y_err, label=name, marker=marker, capsize=5)

    axs[1, 0].set_xlabel('Buffer (s)', fontsize=20)
    axs[1, 0].set_ylabel('QoE', fontsize=20)
    axs[1, 0].set_title('(c) QoE vs. Buffer')
    axs[1, 0].legend()

    # Plot (d) CDF of QoE
    def plot_cdf(data, ax, label, linestyle):
        sorted_data = np.sort(data)
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
        ax.plot(sorted_data, yvals, label=label, linestyle=linestyle)

    linestyles = ['--', '-', ':', '-.']
    for (category, vals), ls in zip(qoe_data.items(), linestyles):
        plot_cdf(vals, axs[1,1], category, ls)

    axs[1, 1].set_xlabel('QoE', fontsize=20)
    axs[1, 1].set_ylabel('CDF', fontsize=20)
    axs[1, 1].set_title('(d) CDF of QoE')
    axs[1, 1].legend()

    # Adjust spacing between subplots
    plt.tight_layout()
    plt.savefig('./thesis_csv_result/detailed_analysis_plots.pdf')
    plt.close()


# Main execution
if __name__ == "__main__":
    combined_df = load_and_preprocess_data(FILE_PATHS)
    
    #plot_box_plots(combined_df)
    #plot_cdf_plots(combined_df)
    #plot_scatter_plot(combined_df)
    #plot_improvement_bar_charts(combined_df)
    #plot_improvement_by_method_and_video_type(combined_df, VIDEO_TYPE_MAPPING)
    #plot_detailed_analysis(combined_df) 
    #for file_path in FILE_PATHS:
        #plot_improvement_for_video(file_path, VIDEO_TYPE_MAPPING)
    plot_improvement_bar_charts_3x1(combined_df)
    
    improvement_data = calculate_improvement_data(combined_df)
    #for network_type in ['Slow', 'Medium', 'Fast', 'HSDPA', 'Lumos 5G', 'Overall']:
        #plot_improvement_for_network(improvement_data, network_type)

    print("Analysis completed. Check the generated PDF files for results.")