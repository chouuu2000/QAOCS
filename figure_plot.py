import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.patches as patches

# Load the provided CSV files
# file_paths = [
#     #'BBB_360p24_performance_comparison.csv',
#      './Training_output_abr_without_qaocs/LOL_3D_performance_comparison.csv',
#      './Training_output_abr_without_qaocs/sport_highlight_performance_comparison.csv',
#     # 'underwater_performance_comparison.csv',
#      './Training_output_abr_without_qaocs/video_game_performance_comparison.csv',
#     # 'sport_long_take_performance_comparison.csv'
# ]

file_paths = [
    #'BBB_360p24_performance_comparison.csv',
     './pq_test/LOL_3D_performance_comparison.csv',
     #'sport_highlight_performance_comparison.csv',
    # 'underwater_performance_comparison.csv',
     #'video_game_performance_comparison.csv',
    # 'sport_long_take_performance_comparison.csv'
]

save_path = './pq_test/compare'

dataframes = [pd.read_csv(file) for file in file_paths]

# Combine all dataframes into one for easier analysis
combined_df = pd.concat(dataframes, ignore_index=True)


# Categorize traces into Slow, Medium, and Fast based on CDF plots
def correct_categorize_trace(file_name):
    if 'oboe' in file_name or 'fcc18' in file_name:
        return 'Slow'
    elif 'lab' in file_name or 'hsr' in file_name:
        return 'Medium'
    elif 'ghent' in file_name or 'lumos5g' in file_name:
        return 'Fast'
    #return 'Unknown'

combined_df['Network Type'] = combined_df['File'].apply(correct_categorize_trace)

# Calculate VMAF Change and QoE for each entry
combined_df['VMAF Change'] = combined_df['Average VMAF Smoothness']
combined_df['QoE'] = 0.13*combined_df['Average VMAF Score'] - (combined_df['Stall Time(s)']* 0.75) - (combined_df['Switch Ratio(%)']*0.2 )

# Aggregate data by video category for plotting
video_categories = combined_df['File'].str.extract(r'([A-Za-z]+)').squeeze().unique()
# print video_categories 
print("Video ->>",video_categories)
combined_df['Video Category'] = combined_df['File'].str.extract(r'([A-Za-z]+)').squeeze()

# Box Plots for key metrics across different network types and methods
#fig, axes = plt.subplots(2, 2, figsize=(18, 15))
fig, axes = plt.subplots(4, 1, figsize=(10, 20))
metrics_to_boxplot = [
    ('Average VMAF Score', 'Score'),
    ('Stall Ratio(%)', 'Ratio'),
    ('Switch Ratio(%)', 'Ratio'),
    ('Average Bitrate(bps)', 'bps')
]

#for i, (metric, unit) in enumerate(metrics_to_boxplot):
#    row, col = divmod(i, 2)
#    sns.boxplot(data=combined_df, x='Method', y=metric, hue='Network Type', ax=axes[row, col])
#    axes[row, col].set_title(f'{metric} by Method and Network Type')
#    axes[row, col].set_xlabel('')
#    axes[row, col].set_ylabel(unit)
plot_df = combined_df.copy()
# plot_df['Method'] = plot_df['Method'].replace({
#     'QAOCS-MPC': 'QAOCS-RMPC'
# })

for i, (metric, unit) in enumerate(metrics_to_boxplot):
    ax = axes[i]
    #sns.boxplot(data=combined_df, x='Method', y=metric, hue='Network Type', ax=ax)
    sns.boxplot(
    data=plot_df,
    x='Method',
    y=metric,
    hue='Network Type',
    hue_order=['Slow', 'Medium', 'Fast'],  
    #order=['QAOCS-RBA', 'QAOCS-RMPC', 'QAOCS-BOLA', 'QAOCS-BBA','QAOCS-PENSIEVE', 'Segue', 'GOP-4', 'Constant-4'],
    ax=ax
    )
    ax.set_title(f'{metric} by Method and Network Type', fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel(unit)

# for i, (metric, unit) in enumerate(metrics_to_boxplot):
#     row, col = divmod(i, 2)
#     grouped_data = [combined_df[combined_df['Network Type'] == network_type][metric] for network_type in combined_df['Network Type'].unique()]
#     labels = combined_df['Network Type'].unique()
#     axes[row, col].boxplot(grouped_data, labels=labels)
#     axes[row, col].set_title(f'{metric} by Network Type')
#     axes[row, col].set_xlabel('Network Type')
#     axes[row, col].set_ylabel(unit)
for ax in axes:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        break
plt.tight_layout()
#plt.legend(title='Network Type', bbox_to_anchor=(1.05, 1), loc='upper left')
fig.legend(handles, labels, title='Network Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
plt.savefig(f'{save_path}/box_plots.pdf')
#debug
print('box done')

# CDF plots for key metrics
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
metrics_to_cdf = [
    ('Average VMAF Score', 'Score'),
    ('Stall Ratio(%)', '%'),
    ('QoE', ''),
    ('Stall Time(s)', 'time')
]

# def plot_cdf(data, metric, ax, label):
#     sorted_data = np.sort(data)
#     yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1) * 100
#     ax.plot(sorted_data, yvals, label=label)

# for i, (metric, unit) in enumerate(metrics_to_cdf):
#     ax = axes[i]
#     for network_type in combined_df['Network Type'].unique():
#         for method in combined_df['Method'].unique():
#             data = combined_df[(combined_df['Network Type'] == network_type) & (combined_df['Method'] == method)][metric]
#             plot_cdf(data, metric, ax, f'{method} - {network_type}')
#     ax.set_title(f'CDF of {metric}')
#     ax.set_xlabel(metric)
#     ax.set_ylabel('CDF (%)')
#     ax.legend()
policy_groups = {
    'pq-policy': ['pq(20, 30)', 'pq(30, 40)', 'pq(40, 50)'],
    'D-policy':  ['D-policy-5', 'D-policy-10', 'D-policy-15'],        # 範例：請換成你的實際名稱
    'N-policy':  ['N-policy-3', 'N-policy-5', 'N-policy-7'],   # 範例：請換成你的實際名稱
}
net_order = ['Slow', 'Medium', 'Fast']
linestyles = ['--', '-', ':', '-.', (0, (1, 1)), (0, (5, 1)), (0, (3, 1, 1, 1)), '--', '-',  ':','-.']

def plot_ecdf_percent(values, ax, label, q_cut_low=0.01, q_cut_high=0.99, **plot_kwargs):
    """
    繪製 ECDF，並自動過濾掉資料中最小 q_cut quantile 以下、
    以及最大 q_cut_high quantile 以上的值
    q_cut:    要裁掉的低端比例 (0.01=1%)
    q_cut_high: 要裁掉的高端比例 (0.99=99%)
    """
    data = np.asarray(values, dtype=float)
    data = data[~np.isnan(data)]

    if data.size == 0:
        return

    if 0 < q_cut_low < 1:
        low_th = np.quantile(data, q_cut_low)
        data = data[data >= low_th]

    if 0 < q_cut_high < 1:
        high_th = np.quantile(data, q_cut_high)
        data = data[data <= high_th]

    if data.size == 0:
        return

    data = np.sort(data)
    if data.size == 1:
        y = np.array([100.0])
    else:
        y = np.arange(data.size) / float(data.size - 1) * 100.0

    ax.plot(data, y, label=label, **plot_kwargs)
for metric, unit in metrics_to_cdf:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for i, net in enumerate(net_order):
        ax = axes[i]
        sub = combined_df[combined_df['Network Type'] == net]
        if sub.empty:
            ax.set_title(f"{net} (no data)")
            ax.set_xlabel(metric)
            if i == 0:
                ax.set_ylabel("CDF (%)")
            ax.grid(True, linestyle = '--', alpha=0.5)
            continue

        methods = sorted(sub['Method'].dropna().unique())
        for (method, ls) in zip(methods, linestyles):
            vals = sub[sub['Method'] == method][metric].values
            plot_ecdf_percent(vals, ax, label=method, linestyle=ls, linewidth=1.8, q_cut_low=0.05, q_cut_high=0.99)

        ax.set_title(f"{net} network")
        ax.set_xlabel(metric)
        if i == 0:
            ax.set_ylabel("CDF (%)")
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle = '--', alpha=0.5)
        ax.legend(title="Method", fontsize=9)
    
    plt.tight_layout()
    safe_metric = metric.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    fig.savefig(f"{save_path}/cdf_{safe_metric}_by_network.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- overall (合併所有 network) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    methods = sorted(combined_df['Method'].dropna().unique())
    for (method, ls) in zip(methods, linestyles):
        vals = combined_df.loc[combined_df['Method'] == method, metric].values
        plot_ecdf_percent(vals, ax, label=method, linestyle=ls, linewidth=1.8, q_cut_low=0.05, q_cut_high=0.99)

    ax.set_title(f"CDF of {metric} (overall)", fontweight='bold')
    ax.set_xlabel(f"{metric}" + (f" ({unit})" if unit else ""))
    ax.set_ylabel("CDF (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="Method", fontsize=9)
    plt.tight_layout()
    fig.savefig(f"{save_path}/cdf_{safe_metric}_overall.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)

# for metric, unit in metrics_to_cdf:
#     # 逐一處理三種 policy
#     for policy_name, policy_methods in policy_groups.items():
#         # --- 1×3（按網路） ---
#         fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
#         for i, net in enumerate(net_order):
#             ax = axes[i]
#             # 只取該網路 + 該 policy 的方法
#             sub = combined_df[(combined_df['Network Type'] == net) &
#                               (combined_df['Method'].isin(policy_methods))]
#             if sub.empty:
#                 ax.set_title(f"{net} (no data)")
#                 ax.set_xlabel(metric)
#                 if i == 0:
#                     ax.set_ylabel("CDF (%)")
#                 ax.set_ylim(0, 100)
#                 ax.grid(True, linestyle='--', alpha=0.5)
#                 continue

#             # 用該 policy 底下實際出現的方法來畫
#             methods = sorted(sub['Method'].dropna().unique())
#             for (method, ls) in zip(methods, linestyles):
#                 vals = sub.loc[sub['Method'] == method, metric].values
#                 plot_ecdf_percent(vals, ax,
#                                   label=method,
#                                   linestyle=ls, linewidth=1.8,
#                                   q_cut_low=0.05, q_cut_high=0.99)

#             ax.set_title(f"{policy_name} – {net}")
#             ax.set_xlabel(metric + (f" ({unit})" if unit else ""))
#             if i == 0:
#                 ax.set_ylabel("CDF (%)")
#             ax.set_ylim(0, 100)
#             ax.grid(True, linestyle='--', alpha=0.5)
#             ax.legend(title="Method", fontsize=9)

#         plt.tight_layout()
#         safe_metric = metric.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
#         safe_policy = policy_name.lower().replace(" ", "_").replace("-", "")
#         fig.savefig(f"{save_path}/cdf_{safe_metric}_{safe_policy}_by_network.pdf",
#                    dpi=200, bbox_inches="tight")
#         plt.close(fig)

#         # --- overall（合併所有網路，但僅該 policy 的方法） ---
#         fig, ax = plt.subplots(figsize=(10, 5))
#         sub_all = combined_df[combined_df['Method'].isin(policy_methods)]
#         if sub_all.empty:
#             ax.set_title(f"CDF of {metric} (overall) – {policy_name} (no data)")
#             ax.set_xlabel(metric + (f" ({unit})" if unit else ""))
#             ax.set_ylabel("CDF (%)")
#             ax.set_ylim(0, 100)
#             ax.grid(True, linestyle="--", alpha=0.5)
#             fig.savefig(f"{save_path}/cdf_{safe_metric}_{safe_policy}_overall.pdf",
#                        dpi=200, bbox_inches="tight")
#             plt.close(fig)
#         else:
#             methods_all = sorted(sub_all['Method'].dropna().unique())
#             for (method, ls) in zip(methods_all, linestyles):
#                 vals = sub_all.loc[sub_all['Method'] == method, metric].values
#                 plot_ecdf_percent(vals, ax,
#                                   label=method,
#                                   linestyle=ls, linewidth=1.8,
#                                   q_cut_low=0.05, q_cut_high=0.99)

#             ax.set_title(f"CDF of {metric} (overall) – {policy_name}")
#             ax.set_xlabel(metric + (f" ({unit})" if unit else ""))
#             ax.set_ylabel("CDF (%)")
#             ax.set_ylim(0, 100)
#             ax.grid(True, linestyle="--", alpha=0.5)
#             ax.legend(title="Method", fontsize=9)
#             plt.tight_layout()
#             fig.savefig(f"{save_path}/cdf_{safe_metric}_{safe_policy}_overall.pdf",
#                        dpi=200, bbox_inches="tight")
#             plt.close(fig)

# # --- overall (合併所有 network) — All Methods in One Figure ---
# from itertools import cycle
# fig, ax = plt.subplots(figsize=(10, 5))

# methods_all = sorted(combined_df['Method'].dropna().unique())
# style_cycle = cycle(linestyles)
# for method in methods_all:
#     ls = next(style_cycle)
#     vals = combined_df.loc[combined_df['Method'] == method, metric].values
#     if vals.size == 0:
#         continue
#     plot_ecdf_percent(vals, ax,
#                       label=method,
#                       linestyle=ls, linewidth=1.8,
#                       q_cut_low=0.05, q_cut_high=0.99)

# ax.set_title(f"CDF of {metric} (Overall, All Methods)")
# ax.set_xlabel(f"{metric}" + (f" ({unit})" if unit else ""))
# ax.set_ylabel("CDF (%)")
# ax.set_ylim(0, 100)
# ax.grid(True, linestyle="--", alpha=0.5)
# ax.legend(title="Method", fontsize=9, ncol=2)  # 方法多時較不擠
# plt.tight_layout()

# safe_metric = metric.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
# fig.savefig(f"{save_path}/cdf_{safe_metric}_overall_all_methods.pdf",
#            dpi=200, bbox_inches="tight")
# plt.close(fig)

plt.tight_layout()
plt.show()
plt.savefig(f'{save_path}/cdf_plots.pdf')
# Scatter Plot for Average Bitrate vs Average VMAF Score
plt.figure(figsize=(12, 8))
sns.scatterplot(data=combined_df, x='Average Bitrate(bps)', y='Average VMAF Score', hue='Method', style='Network Type')
plt.title('Average Bitrate vs Average VMAF Score by Method and Network Type')
plt.xlabel('Average Bitrate (bps)')
plt.ylabel('Average VMAF Score')
plt.legend(title='Method and Network Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

#debug
print('cdf done')

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
plt.savefig(f'{save_path}/heatmap.pdf')

#debug
print('heatmap done')

# Radar Chart for overall performance comparison
from math import pi
from matplotlib.patches import Circle
metric_config = {
    'Average VMAF Score': True,
    'Stall Ratio(%)': False,
    'Switch Ratio(%)': False,
    'Average Bitrate(bps)': True
}

# 用來顯示在雷達圖上的指標標籤（保持方向語意一致）
display_labels = {
    'Stall Ratio(%)': 'Stall Avoidance',
    'Average VMAF Score': 'Average VMAF Score',
    'Switch Ratio(%)': 'Switch Avoidance',
    'Average Bitrate(bps)': 'Average Bitrate (bps)'
}



# def create_radar_chart_data(df, method, min_max_dict):
#     metrics = ['Average VMAF Score', 'Stall Ratio(%)', 'Switch Ratio(%)', 'Average Bitrate(bps)']
#     #normalized
#     raw_values = df[df['Method'] == method][metrics].mean()
#     normalized_values = [
#         (raw_values[m] - min_max_dict[m][0]) / (min_max_dict[m][1] - min_max_dict[m][0] + 1e-8)
#         for m in metrics
#     ]
#     normalized_values += normalized_values[:1]  # close the loop
#     # values = df[df['Method'] == method][metrics].mean().tolist()
#     # values += values[:1]  # Repeat the first value to close the circle
#     return normalized_values

# Radar data 正規化 + 反轉處理
# def create_radar_chart_data(df, method, min_max_dict):
#     raw_values = df[df['Method'] == method][metrics].mean()
#     normalized_values = []
#     for m in metrics:
#         min_val, max_val = min_max_dict[m]
#         val = raw_values[m]
#         if metric_config[m]:  # 越大越好
#             norm_val = (val - min_val) / (max_val - min_val + 1e-8)
#         else:  # 越小越好（需反轉）
#             norm_val = (max_val - val) / (max_val - min_val + 1e-8)
#         normalized_values.append(norm_val)
#     normalized_values += normalized_values[:1]  # 關閉環狀
#     return normalized_values

# metrics = ['Stall Ratio(%)', 'Average VMAF Score', 'Switch Ratio(%)', 'Average Bitrate(bps)']
# categories = [display_labels[m] for m in metrics]
# num_vars = len(categories)
# angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
# angles += angles[:1]

# fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
#methods = combined_df['Method'].unique()
# #colors = sns.color_palette("husl", len(methods))
# colors = [
#     (0.894, 0.102, 0.110),  # 紅
#     (0.216, 0.494, 0.722),  # 藍
#     (0.302, 0.686, 0.290),  # 綠
#     (0.596, 0.306, 0.639),  # 紫
#     (1.000, 0.498, 0.000),  # 橙
#     (1.000, 1.000, 0.200),  # 黃
#     (0.651, 0.337, 0.157),  # 棕
#     (0.969, 0.506, 0.749)   # 粉
# ]

# min_max_dict = {metric: (combined_df[metric].min(), combined_df[metric].max()) for metric in metrics}
# for i, method in enumerate(methods):
#     values = create_radar_chart_data(combined_df, method, min_max_dict)
#     ax.plot(angles, values, linewidth=2, linestyle='solid', label=method, color=colors[i])
#     #ax.fill(angles, values, color=colors[i], alpha=0.25)

# for r in np.arange(0.1, 1.0, 0.1): 
#     circle = Circle((0, 0), radius=r, transform=ax.transData._b,
#                     color='black', lw=0.6, fill=False, linestyle='solid')
#     ax.add_patch(circle)

# # 右側圈內標籤，往左偏移 6pt
# ax.set_ylim(0, 1.02)
# for r in np.arange(0.1, 1.01, 0.1):
#     ax.annotate(f'{r:.1f}', xy=(0, r), xytext=(-1, 0),  # θ=0，向左偏移
#                 textcoords='offset points', ha='right', va='center', fontsize=9)

# #plt.xticks(angles[:-1], categories, color='black', size=10)

# # 關掉預設 xtick 標籤
# ax.set_xticklabels([])

# # 在半徑外一點的位置手動畫標籤
# r_label = ax.get_ylim()[1] * 1.06

# for cat, ang in zip(categories, angles[:-1]):
#     ang_deg = (np.degrees(ang) + 360) % 360  # 0~360
#     rot = 0
#     ha, va = 'center', 'center'
#     rr = r_label

#     if 350 <= ang_deg or ang_deg <= 10:          # 右側 (0°)
#         rot = 90
#         ha, va = 'center', 'center'   # ← 置中
#     elif 170 <= ang_deg <= 190:                   # 左側 (180°)
#         rot = 90
#         ha, va = 'center', 'center'   # ← 置中
#     elif 80 <= ang_deg <= 100:                    # 上方 (90°)
#         rot = 0
#         ha, va = 'center', 'bottom'
#     elif 260 <= ang_deg <= 280:                   # 下方 (270°)
#         rot = 0
#         ha, va = 'center', 'top'

#     ax.text(ang, rr, cat,
#             rotation=rot,
#             rotation_mode='default',  # ← 不要用 'anchor'
#             ha=ha, va=va, fontsize=10, clip_on=False)
    
# ax.yaxis.set_visible(False)
# plt.title('Comparison on Multiple Metrics After Min-Max Normalization', size=15, color='black', y=1.1, fontweight='bold')
# plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
# plt.show()
# plt.savefig(f'{save_path}/radar_chart.pdf')

#debug
print('radar done')

# Detailed Analysis Plots similar to reference figure
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

# Compute stall time (平均重緩衝時長)
stall_time = combined_df.groupby('Method')['Stall Time(s)'].mean().to_dict()
stall_time_err = combined_df.groupby('Method')['Stall Time(s)'].std().to_dict()
stall_ratio = combined_df.groupby('Method')['Stall Ratio(%)'].mean().to_dict()
stall_ratio_err = combined_df.groupby('Method')['Stall Ratio(%)'].std().to_dict()

#fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig, ax = plt.subplots(figsize=(6, 4))
markers = ['o', 's', 'D', '^', 'v', '>', '<', 'p','o', 's', 'D', '^', 'v', '>', '<', 'p']
#markers = ['o', 's', 'D', '^', 'v', '>', '<']

# Plot (a) VMAF vs. Stall Ratio
# for name, marker in zip(stall_ratio.keys(), markers):
#     x = stall_ratio[name]
#     y = vmaf_scores[name]
#     x_err = stall_ratio_errors[name]
#     y_err = vmaf_errors[name]
#     axs[0,0].errorbar(x, y, xerr=x_err, yerr=y_err, label=name, marker=marker, capsize=5)

# axs[0, 0].set_xlabel('Time Spent on Stall (%)')
# axs[0, 0].set_ylabel('Video Quality (VMAF)')
# axs[0, 0].set_title('(a) VMAF vs. Stall Ratio')
# axs[0, 0].legend()
# axs[0, 0].grid(True)
# axs[0, 0].invert_xaxis()
# print('a done')

# Plot (b) VMAF vs. VMAF Change
# for name, marker in zip(vmaf_vs_vmaf_change.keys(), markers):
#     x = combined_df[combined_df['Method'] == name]['VMAF Change']
#     y = combined_df[combined_df['Method'] == name]['Average VMAF Score']
#     x_err = vmaf_vs_vmaf_change_err[name]
#     y_err = vmaf_errors[name]
#     axs[0,1].errorbar(x.mean(), y.mean(), xerr=x_err, yerr=y_err, label=name, marker=marker, capsize=5)
# axs[0, 1].set_xlabel('Quality Smoothness (VMAF Change)')
# axs[0, 1].set_ylabel('Video Quality (VMAF)')
# axs[0, 1].set_title('(b) VMAF vs. VMAF Change')
# axs[0, 1].legend()
# print('b done')
# Plot (c) QoE_DNN vs. Buffer
# for name, marker in zip(qoe_vs_buffer.keys(), markers):
#     x = buffer_sizes[name]
#     y = qoe_vs_buffer[name]
#     x_err = combined_df[combined_df['Method'] == name]['Average Buffer State(s)'].std()
#     y_err = qoe_vs_buffer_err[name]
#     axs[1,0].errorbar(x, y, xerr=x_err, yerr=y_err, label=name, marker=marker, capsize=5)

# axs[1, 0].set_xlabel('Buffer (s)')
# axs[1, 0].set_ylabel('QoE')
# axs[1, 0].set_title('(c) QoE vs. Buffer')
# axs[1, 0].legend()


for name, marker in zip(qoe_vs_buffer.keys(), markers):
    x = buffer_sizes[name]
    y = qoe_vs_buffer[name]
    x_err = combined_df[combined_df['Method'] == name]['Average Buffer State(s)'].std()
    y_err = qoe_vs_buffer_err[name]
    ax.errorbar(x, y, xerr=x_err, yerr=y_err, label=name, marker=marker, capsize=5)

ax.set_xlabel('Buffer (s)')
ax.set_ylabel('QoE')
ax.set_title('(c) QoE vs. Buffer')
ax.legend()
print('c done')

# Plot (d) CDF of QoE
# def plot_cdf(data, ax, label, linestyle):
#     sorted_data = np.sort(data)
#     yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
#     ax.plot(sorted_data, yvals, label=label, linestyle=linestyle)

# linestyles = ['--', '-', ':', '-.', '--', '-']
# #for (category, vals), ls in zip(qoe_data.items(), linestyles):
# for (category, vals), ls in zip(qoe_data.items(), linestyles):
#     plot_cdf(vals, axs[1,1], category, ls)

# axs[1, 1].set_xlabel('QoE')
# axs[1, 1].set_ylabel('CDF')
# axs[1, 1].set_title('(d) CDF of QoE')
# axs[1, 1].legend()
# print('d done')
# # Adjust spacing between subplots
# plt.tight_layout()
# plt.show()

# fig, ax = plt.subplots(figsize=(6, 4))

# markers = ['o', 's', 'D', '^', 'v', '>', '<', 'p','h','x','*','+','1','2','3','4']

# for name, marker in zip(stall_time.keys(), markers):
#     x = stall_time[name]
#     y = stall_ratio[name]
#     x_err = stall_time_err[name]
#     y_err = stall_ratio_err[name]
    
#     ax.errorbar(x, y, 
#                 xerr=x_err, yerr=y_err, 
#                 label=name, marker=marker, 
#                 capsize=5, linestyle='')

# ax.set_xlabel('Stall Time (s)')
# ax.set_ylabel('Stall Ratio (%)')
# ax.set_title('Stall Ratio vs. Stall Time')
# ax.legend()
# plt.tight_layout()
# plt.show()
# plt.savefig(f"{save_path}/stall_time_vs_stall_ratio_2Dbox.pdf")


methods = combined_df['Method'].unique()
colors = plt.cm.Set1.colors  # 給每個方法不同顏色

fig, ax = plt.subplots(figsize=(8,6))

for i, method in enumerate(methods):
    data = combined_df[combined_df['Method'] == method].copy()
    x = data['Stall Time(s)'].to_numpy()
    y = data['Stall Ratio(%)'].to_numpy()

    if method.startswith(("N-policy", "D-policy")):
        lower = np.percentile(y, 5)   # 小於 5% 的丟掉
        upper = np.percentile(y, 90)  # 大於 95% 的丟掉
        mask = y <= upper
        x, y = x[mask], y[mask]

    if len(x) == 0 or len(y) == 0:
        continue  # 避免空集合

    # 四分位數
    x_q1, x_q2, x_q3 = np.percentile(x, [25, 50, 75])
    y_q1, y_q2, y_q3 = np.percentile(y, [25, 50, 75])

    # 畫 2D 箱型矩形
    rect = patches.Rectangle(
        (x_q1, y_q1),
        x_q3 - x_q1,
        y_q3 - y_q1,
        linewidth=2,
        edgecolor=colors[i % len(colors)],
        facecolor=colors[i % len(colors)],
        alpha=0.3,
        label=method
    )
    ax.add_patch(rect)

    # 畫中位數十字
    ax.plot([x_q2, x_q2], [y_q1, y_q3], color=colors[i % len(colors)], lw=2)
    ax.plot([x_q1, x_q3], [y_q2, y_q2], color=colors[i % len(colors)], lw=2)

ax.set_xlabel("Stall Time (s)")
ax.set_ylabel("Stall Ratio (%)")
ax.set_title("2D Boxplot of Stall Time vs Stall Ratio by Method")
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig(f"{save_path}/stall_time_vs_stall_ratio_2Dbox.pdf")
plt.show()
print("2D boxplot done")

# Save the figure
plt.savefig(f'{save_path}/detailed_analysis_plots.pdf')

# #debug
# print('detailed done')



def calculate_improvement(qaocs_value, baseline_value, metric):
    if metric == 'Stall Ratio(%)':
        return (baseline_value - qaocs_value) / baseline_value * 100
    elif metric == 'Switch Ratio(%)':
         return (baseline_value - qaocs_value) / baseline_value * 100
    elif metric == 'Stall Time(s)':
         return (baseline_value - qaocs_value) / baseline_value * 100
    else:
        return (qaocs_value - baseline_value) / baseline_value * 100

def calculate_improvement_data(combined_df):
    improvement_data = {network_type: {} for network_type in ['Slow', 'Medium', 'Fast', 'HSDPA', 'Lumos 5G', 'Overall']}
    metrics = ['QoE', 'Stall Time(s)']

    method_pairs = [
        ('pq(30, 40)', 'Single(40)'),
        ('pq(40, 50)', 'Single(50)'),
    ]

    for network_type in improvement_data.keys():
        if network_type == 'Overall':
            network_df = combined_df
        else:
            network_df = combined_df[combined_df['Network Type'] == network_type]
        
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


def plot_improvement_bar_charts(combined_df):
    improvement_data = calculate_improvement_data(combined_df)
    metrics = ['QoE', 'Stall Time(s)']
    metric_labels = {
    'QoE': 'QoE',
    'Stall Time(s)': 'Average Stall Time (s)',
    }
    method_pairs = [
        ('pq(30, 40)', 'Single(40)'),
        ('pq(40, 50)', 'Single(50)'),
    ]
   #network_types = ['Slow', 'Medium', 'Fast', 'Overall']
    network_types = ['Slow', 'Overall']
    
    baseline_colors = {
        'Single(40)': '#1f77b4',  # 藍色
        'Single(50)': '#ff7f0e',  # 橘色
    }
    label_name = {
        'Single(40)': 'Single(40)',
        'Single(50)': 'Single(50)',
    }

    # 建立 2x2 subplot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes = axes.flatten()  # 攤平成一維陣列，方便迴圈存取

    for idx, net in enumerate(network_types):
        if net not in improvement_data:
            continue
        data = improvement_data[net]
        baselines = [b for _, b in method_pairs if (b in data and all(m in data[b] for m in metrics))]
        if not baselines:
            continue

        ax = axes[idx]  # 選對應子圖
        x = np.linspace(0, len(metrics)-1, len(metrics)) * 0.6 
        width = min(0.18, 0.8 / max(1, len(baselines)))
        offsets = (np.arange(len(baselines)) - (len(baselines)-1)/2) * (width + 0.02)

        for bi, b in enumerate(baselines):
            y = [float(data[b][m]) for m in metrics]
            bars = ax.bar(x + offsets[bi], y, width=width,
                          label=label_name.get(b, b),
                          color=baseline_colors.get(b, 'gray'))
            # 標上百分比
            for rect, val in zip(bars, y):
                ax.text(rect.get_x() + rect.get_width()/2,
                        val + (0.1 if val >= 0 else -0.1),
                        f'{val:.1f}%', ha='center',
                        va='bottom' if val >= 0 else 'top',
                        fontsize=9)

        ax.set_xticks(x)
        #ax.set_xticklabels(metrics)
        ax.set_xticklabels([metric_labels[m] for m in metrics])
        ax.set_ylabel('Improvement (%)')
        ax.set_title(f'PQ-Policy Improvement - {net}', fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_ylim(-10, 50)
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(f'{save_path}/buffer_policy_improvement_all.pdf', bbox_inches='tight')


plot_improvement_bar_charts(combined_df)