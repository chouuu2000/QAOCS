import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.patches as patches
from math import pi
from matplotlib.patches import Circle

file_paths = [
    './abr_result/sport_highlight_performance_comparison_new.csv',
    './abr_result/video_game_performance_comparison_new.csv',
    './abr_result/LOL_3D_performance_comparison_new.csv',
]

save_path = './abr_result'
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
plot_df['Method'] = plot_df['Method'].replace({
    'QAOCS-MPC': 'QAOCS-RMPC'
     })

for i, (metric, unit) in enumerate(metrics_to_boxplot):
    ax = axes[i]
    #sns.boxplot(data=combined_df, x='Method', y=metric, hue='Network Type', ax=ax)
    sns.boxplot(
    data=plot_df,
    x='Method',
    y=metric,
    hue='Network Type',
    hue_order=['Slow', 'Medium', 'Fast'],  
    order=['QAOCS-BBA', 'QAOCS-RBA', 'QAOCS-RMPC', 'QAOCS-BOLA','QAOCS-PENSIEVE', 'Segue', 'GOP-4', 'Constant-4'],
    ax=ax
    )
    ax.set_title(f'{metric} by Method and Network Type', fontweight='bold',fontsize=14)
    ax.set_xlabel('')
    ax.set_ylabel(unit, fontsize=18)

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
plt.savefig(f'{save_path}/box_plots_abr.pdf')
#debug
print('box done')





# Radar Chart for overall performance comparison

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



def create_radar_chart_data(df, method, min_max_dict):
    #metrics = ['Average VMAF Score', 'Stall Ratio(%)', 'Switch Ratio(%)', 'Average Bitrate(bps)']
    metrics = ['Average VMAF Score', 'Stall Ratio(%)', 'Average Bitrate(bps)']
    #normalized
    raw_values = df[df['Method'] == method][metrics].mean()
    normalized_values = [
        (raw_values[m] - min_max_dict[m][0]) / (min_max_dict[m][1] - min_max_dict[m][0] + 1e-8)
        for m in metrics
    ]
    normalized_values += normalized_values[:1]  # close the loop
    # values = df[df['Method'] == method][metrics].mean().tolist()
    # values += values[:1]  # Repeat the first value to close the circle
    return normalized_values

#Radar data 正規化 + 反轉處理
def create_radar_chart_data(df, method, min_max_dict):
    raw_values = df[df['Method'] == method][metrics].mean()
    normalized_values = []
    for m in metrics:
        min_val, max_val = min_max_dict[m]
        val = raw_values[m]
        if metric_config[m]:  # 越大越好
            norm_val = (val - min_val) / (max_val - min_val + 1e-8)
        else:  # 越小越好（需反轉）
            norm_val = (max_val - val) / (max_val - min_val + 1e-8)
        normalized_values.append(norm_val)
    normalized_values += normalized_values[:1]  # 關閉環狀
    return normalized_values

#metrics = ['Stall Ratio(%)', 'Average VMAF Score', 'Switch Ratio(%)', 'Average Bitrate(bps)']
metrics = ['Average VMAF Score', 'Stall Ratio(%)', 'Average Bitrate(bps)']
categories = [display_labels[m] for m in metrics]
num_vars = len(categories)
angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
# 指定繪線與圖例顯示順序
radar_df = combined_df.copy()
radar_df['Method'] = radar_df['Method'].str.strip().replace({'QAOCS-MPC': 'QAOCS-RMPC'})

desired_order = [
    'QAOCS-BBA',
    'QAOCS-RBA',
    'QAOCS-RMPC',
    'QAOCS-BOLA',
    'QAOCS-PENSIEVE',
    'Segue',
    'Constant-4',
    'GOP-4'
]

# 過濾出實際有在資料中的 method，依指定順序繪製
methods = [m for m in desired_order if m in radar_df['Method'].unique()]

#colors = sns.color_palette("husl", len(methods))
colors = [
    (0.894, 0.102, 0.110),  # 紅
    (0.216, 0.494, 0.722),  # 藍
    (0.302, 0.686, 0.290),  # 綠
    (0.596, 0.306, 0.639),  # 紫
    (1.000, 0.498, 0.000),  # 橙
    (1.000, 1.000, 0.200),  # 黃
    (0.651, 0.337, 0.157),  # 棕
    (0.969, 0.506, 0.749)   # 粉
]

min_max_dict = {metric: (radar_df[metric].min(), radar_df[metric].max()) for metric in metrics}
linestyles = ['--', (0, (3, 1, 1, 1)), '-.', ':', (0, (1, 1)), (0, (5, 1)), '--', '--']
for i, method in enumerate(methods):
    values = create_radar_chart_data(radar_df, method, min_max_dict)
    ax.plot(angles, values, linewidth=2, linestyle=linestyles[i % len(linestyles)], label=method, color=colors[i])
    #ax.fill(angles, values, color=colors[i], alpha=0.25)

ax.set_thetagrids([0, 90, 180, 270])
for r in np.arange(0.1, 1.0, 0.1): 
    circle = Circle((0, 0), radius=r, transform=ax.transData._b,
                    color='black', lw=0.6, fill=False, linestyle='solid')
    ax.add_patch(circle)

# 右側圈內標籤，往左偏移 6pt
ax.set_ylim(0, 1.02)
for r in np.arange(0.1, 1.01, 0.1):
    ax.annotate(f'{r:.1f}', xy=(0, r), xytext=(-1, 0),  # θ=0，向左偏移
                textcoords='offset points', ha='right', va='center', fontsize=9)

#plt.xticks(angles[:-1], categories, color='black', size=10)

# 關掉預設 xtick 標籤
ax.set_xticklabels([])

# 在半徑外一點的位置手動畫標籤
r_label = ax.get_ylim()[1] * 1.06

for cat, ang in zip(categories, angles[:-1]):
    ang_deg = (np.degrees(ang) + 360) % 360  # 0~360
    rot = 0
    ha, va = 'center', 'center'
    rr = r_label

    if 350 <= ang_deg or ang_deg <= 10:          # 右側 (0°)
        rot = 90
        ha, va = 'center', 'center'   # ← 置中
    elif 170 <= ang_deg <= 190:                   # 左側 (180°)
        rot = 90
        ha, va = 'center', 'center'   # ← 置中
    elif 80 <= ang_deg <= 100:                    # 上方 (90°)
        rot = 0
        ha, va = 'center', 'bottom'
    elif 260 <= ang_deg <= 280:                   # 下方 (270°)
        rot = 0
        ha, va = 'center', 'top'

    ax.text(ang, rr, cat,
            fontsize=18,
            rotation=rot,
            rotation_mode='default',  # ← 不要用 'anchor'
            ha=ha, va=va, clip_on=False)
    
ax.yaxis.set_visible(False)
plt.title('Comparison on Multiple Metrics After Min-Max Normalization', size=16, color='black', y=1.1, fontweight='bold')
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
plt.savefig(f'{save_path}/radar_chart.pdf', bbox_inches='tight')

#debug
print('radar done')


# ===== 3x1 Table figure: mean ± std for each Network Type =====


# 與上游一致：把 MPC → RMPC
tbl_df = combined_df.copy()
tbl_df['Method'] = tbl_df['Method'].replace({'QAOCS-MPC': 'QAOCS-RMPC'})

# 你想在表裡顯示的指標（3 欄）
metrics_for_table = [
    ('Average VMAF Score', 'Score'),
    ('Stall Ratio(%)', 'Ratio'),
    ('Switch Ratio(%)', 'Ratio'),
    ('Average Bitrate(bps)', 'bps')
]

# 指標方向（True=越大越好；False=越小越好）用來決定哪個值要粗體
metric_better_is_higher = {
    'QoE': True,
    'Average VMAF Score': True,
    'Stall Ratio(%)': False,
    'Switch Ratio(%)':False
}
# 顯示小數位
decimals = {'QoE': 2, 'Average VMAF Score': 2, 'Stall Ratio(%)': 2}

# Method 顯示順序
method_order = ['QAOCS-BBA', 'QAOCS-RBA', 'QAOCS-RMPC', 'QAOCS-BOLA',
                'QAOCS-PENSIEVE', 'Segue', 'GOP-4', 'Constant-4']

def _fmt(mean, std, d=2):
    if np.isnan(mean) or np.isnan(std):
        return '–'
    return f"{mean:.{d}f} ± {std:.{d}f}"

def _build_table_matrix(df):
    # 聚合
    agg = df.groupby('Method')[[m for m, _ in metrics_for_table]].agg(['mean', 'std'])
    rows = [m for m in method_order if m in agg.index]
    if not rows:
        return [], [], []

    # 取得要加粗的位置（每欄最佳值 index）
    bold_idx_per_col = []
    for col_name, _label in metrics_for_table:
        series = agg[(col_name, 'mean')].loc[rows]
        if metric_better_is_higher[col_name]:
            best_idx = series.idxmax()
        else:
            best_idx = series.idxmin()
        bold_idx_per_col.append(best_idx)

    # 組成字串表格和加粗 mask
    cell_text = []
    bold_mask = []  # 形狀與 cell_text 相同，True=要加粗
    for r in rows:
        row_text, row_bold = [], []
        for (col_name, _label) in metrics_for_table:
            mean_val = agg.loc[r, (col_name, 'mean')]
            std_val  = agg.loc[r, (col_name, 'std')]
            row_text.append(_fmt(mean_val, std_val, decimals[col_name]))
            row_bold.append(r == bold_idx_per_col[[c for c,(n,_) in enumerate(metrics_for_table) if n==col_name][0]])
        cell_text.append(row_text)
        bold_mask.append(row_bold)
    col_labels = [lbl for _, lbl in metrics_for_table]
    return rows, col_labels, cell_text, bold_mask

nets = ['Slow', 'Medium', 'Fast']
fig, axs = plt.subplots(3, 1, figsize=(10, 6.5))  # 高度可依需要調
if not isinstance(axs, (list, np.ndarray)): axs = [axs]

for i, net in enumerate(nets):
    ax = axs[i]
    ax.axis('off')
    sub = tbl_df[tbl_df['Network Type'] == net]
    rows, col_labels, cell_text, bold_mask = _build_table_matrix(sub)
    if not rows:
        ax.text(0.5, 0.5, f'No data for {net}', ha='center', va='center', fontsize=12)
        continue

    tbl = ax.table(cellText=cell_text,
                   rowLabels=rows,
                   colLabels=['Method'] + col_labels,
                   cellLoc='center',
                   rowLoc='center',
                   loc='center')

    # 美化：隱藏格線、調字體、標頭粗體、行高
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.05, 1.25)
    # 表頭粗體
    for key, cell in tbl.get_celld().items():
        r, c = key
        # 表頭列 r==0
        if r == 0:
            cell.set_text_props(fontweight='bold')
        # 第一欄是 rowLabels 的標題 "Method"（由 table 自動放在 (0,0)）
        # 其餘格線淡化
        cell.set_edgecolor((0,0,0,0))  # 透明邊框

    # 把「Method」那一欄填入 rowLabels
    # Matplotlib 的 table 會把 rowLabels 放在第一欄；已由 colLabels=['Method', ...] 對齊

    # ✅ 對最佳值加粗（逐欄位，安全版）
    for r_idx in range(len(rows)):
        for c_idx in range(len(col_labels)):
            key = (r_idx + 1, c_idx)  # Matplotlib Table 資料格從 col=0 開始
            if key in tbl.get_celld() and bold_mask[r_idx][c_idx]:
                tbl[key].get_text().set_fontweight('bold')


    # 小標題（子圖標題）
    ax.set_title(f'{net}', fontsize=12, fontweight='bold', pad=6)

# 全圖標題
fig.suptitle('Mean ± Std by Method (per Network Type)', fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f'{save_path}/table_mean_std_3x1.pdf', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"[table] Saved -> {save_path}/table_mean_std_3x1.pdf")
