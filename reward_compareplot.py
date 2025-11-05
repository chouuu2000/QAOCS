import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# === 1) 三份 CSV 檔案 ===
csv_files = [
    "./reward_compare/LOL_3D_performance_comparison.csv",
    "./reward_compare/sport_highlight_performance_comparison.csv",
    "./reward_compare/video_game_performance_comparison.csv",
]

# === 2) 依 File 名稱歸類網路型態 ===
def categorize_trace(file_name: str) -> str:
    name = str(file_name).lower()
    if ('oboe' in name) or ('fcc18' in name):
        return 'Slow'
    elif ('lab' in name) or ('hsr' in name):
        return 'Medium'
    elif ('ghent' in name) or ('lumos5g' in name) or ('lumos' in name):
        return 'Fast'
    return 'Unknown'

# === 3) 讀取與合併 ===
df = pd.concat([pd.read_csv(p) for p in csv_files], ignore_index=True)

# 必備欄位檢查（依你的 CSV 標頭截圖）
required = ['File', 'Method', 'Stall Time(s)', 'Stall Ratio(%)']
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

# 轉數值、丟缺失
for col in ['Stall Time(s)', 'Stall Ratio(%)']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=['Stall Time(s)', 'Stall Ratio(%)']).copy()

# 分類網路型態
df['NetType'] = df['File'].map(categorize_trace)

# （可選）Method 顯示名稱映射：若不需要就保留原名
rename_map = {'stallfrequency':'Stall Probability',
              'stalltime':'Stall Duration',
              'binaryindicator':'Binary Stall Indicator'}
df['Method'] = df['Method'].replace(rename_map)

# 不指定順序：依資料出現順序
methods = df['Method'].unique()
net_order = ['Slow', 'Medium', 'Fast']  # 子圖順序

styles = ['-', '--', ':', '-.']  # 夠三種方法用
linestyle_map = {m: styles[i % len(styles)] for i, m in enumerate(methods)}

# （可選）過濾極端大值：把最極端 1% 移除（預設關閉）
def filter_outliers(dfin, metric_col, keep_quantile=1.0):
    if keep_quantile >= 1.0:
        return dfin
    q = dfin[metric_col].quantile(keep_quantile)
    return dfin[dfin[metric_col] <= q]

# ECDF 小工具
def ecdf_vals(arr: np.ndarray):
    x = np.sort(arr)
    y = np.arange(1, x.size + 1) / x.size
    return x, y

# === 4) 畫圖：一張圖有三個子圖（Slow/Medium/Fast）===
outdir = Path("./reward_compare")
outdir.mkdir(parents=True, exist_ok=True)

def plot_three_panel_cdf(df_in, metric_col, outfile, keep_quantile=0.001):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    # 把 2x2 的 axes 變成一維，好用 axes[i]
    try:
        axes = axes.flatten()
    except AttributeError:
        axes = [axes]  # 單一子圖時保險

    for i, net in enumerate(net_order[:3]):
        ax = axes[i]
        sub = df_in[df_in['NetType'] == net]
        if sub.empty:
            ax.set_title(f"{net} (no data)")
            ax.set_xlabel(metric_col)
            ax.set_ylabel("ECDF" if i == 0 else "")
            ax.grid(True, linestyle="--", alpha=0.6)
            continue

        # 過濾 outliers（可選）
        sub = filter_outliers(sub, metric_col, keep_quantile=keep_quantile)

        for m in methods:
            g = sub[sub['Method'] == m][metric_col].to_numpy()
            if g.size == 0:
                continue
            x, y = ecdf_vals(g)
            ax.plot(x, y * 100.0, label=m, linestyle=linestyle_map[m], linewidth=1.8)

        ax.set_title(f"{net} Network", fontsize=18)
        ax.set_xlabel(metric_col, fontsize=18)
        ax.set_ylabel("CDF (%)", fontsize=18)
        ax.grid(True, linestyle="--", alpha=0.6)
        
        if ax.lines:  # 確保有資料才畫 legend
            ax.legend(title="Method", loc="lower right", fontsize=10,  title_fontsize=10)

    ax = axes[3]
    sub_all = df_in.dropna(subset=[metric_col])
    if sub_all.empty:
        ax.set_title("Overall (no data)")
        ax.set_xlabel(metric_col)
        ax.set_ylabel("CDF (%)")
        ax.grid(True, linestyle="--", alpha=0.6)
    else:
        sub_all = filter_outliers(sub_all, metric_col, keep_quantile=keep_quantile)
        for m in methods:
            g = sub_all[sub_all['Method'] == m][metric_col].to_numpy()
            if g.size == 0:
                continue
            x, y = ecdf_vals(g)   # y ∈ [0,1]
            ax.plot(x, y * 100.0, label=m, linestyle=linestyle_map[m], linewidth=1.8)

        ax.set_title("Overall", fontsize=18)
        ax.set_xlabel(metric_col, fontsize=18)
        ax.set_ylabel("CDF (%)", fontsize=18)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle="--", alpha=0.6)
        if ax.lines:
            ax.legend(title="Method", loc="lower right", fontsize=10, title_fontsize=10)


    #fig.suptitle(f"CDFs of {metric_col}", y=0.98, fontweight='bold', fontsize=18)
    plt.tight_layout()
    plt.savefig(outdir / outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)

# === 5) 產生兩張圖 ===
# 圖1：Stall Time(s)
plot_three_panel_cdf(df, metric_col="Stall Time(s)", outfile="cdf_stall_time_by_net.pdf", keep_quantile=0.95)

# 圖2：Stall Ratio(%)
plot_three_panel_cdf(df, metric_col="Stall Ratio(%)", outfile="cdf_stall_ratio_by_net.pdf", keep_quantile=1.0)

print("Saved")