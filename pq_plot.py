# -*- coding: utf-8 -*-
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
# -----------------------------
# Config
# -----------------------------
FILE_PATHS = [
    "./pq_test/LOL_3D_performance_comparison_DN.csv",
    #"./pq_test/LOL_3D_performance_comparison_single.csv",
]
SAVE_DIR = Path("./pq_test/compare")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

HUE_ORDER = ["Slow", "Medium", "Fast"]
COLORS = ["tab:blue", "tab:orange", "tab:green"]
LINESTYLES = ["--", "-", ":", "-.", (0, (1, 1)), (0, (5, 1)), (0, (3, 1, 1, 1)), "--", "-", ":", "-."]

#固定方法顯示順序（左→右／圖例順序）
METHOD_ORDER = [
    "pq(20, 30)",
    "pq(30, 40)",
    "pq(40, 50)",
    "D(5)",
    "D(10)",
    "D(15)",
    "N(3)",
    "N(5)",
    "N(7)",
]
# METHOD_ORDER = [
#     "pq(30, 40)",
#     "pq(40, 50)",
#     "Single(40)",
#     "Single(50)"

# ]
KEEP_ONLY_METHOD_ORDER = False  # 只保留在 METHOD_ORDER 內的方法（如要保留其他方法，改成 False）

BOX_METRICS = [
    ("Stall Ratio(%)", "Ratio"),
    ("Stall Time(s)", "Time"),
]

CDF_METRICS = [
    ("Average VMAF Score", "Score"),
    ("Stall Ratio(%)", "%"),
    ("QoE", ""),
    ("Stall Time(s)", "time"),
]

# PQ vs Single 的對照（依你的命名）
IMPROVE_PAIRS = [
    ("pq(30, 40)", "Single(40)"),
    ("pq(40, 50)", "Single(50)"),
]
IMPROVE_NETWORKS = ["Slow", "Overall"]  # 想看更多可加 "Medium", "Fast"

# -----------------------------
# Helpers
# -----------------------------
def read_concat(paths):
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        df["__source__"] = p
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def normalize_method_names(s: str) -> str:
    """D-policy-5 → D-policy(5), N-policy-3 → N-policy(3), pq(20,30) → pq(20, 30)"""
    if not isinstance(s, str):
        return s
    s = re.sub(r"\bD-policy-(\d+)\b", r"D(\1)", s)
    s = re.sub(r"\bN-policy-(\d+)\b", r"N(\1)", s)
    s = re.sub(r"\bpq\(\s*(\d+)\s*,\s*(\d+)\s*\)", r"pq(\1, \2)", s)
    return s

def categorize_trace(fname: str) -> str:
    f = str(fname).lower()
    if any(k in f for k in ["oboe", "fcc18"]):
        return "Slow"
    if any(k in f for k in ["lab", "hsr"]):
        return "Medium"
    if any(k in f for k in ["ghent", "lumos5g"]):
        return "Fast"
    return "Unknown"

def ensure_episode(df: pd.DataFrame) -> pd.Series:
    return df["episode"] if "episode" in df.columns else pd.Series(np.arange(1, len(df)+1), index=df.index, name="episode")

def ecdf_percent(values, q_low=0.05, q_high=0.99):
    data = np.asarray(values, dtype=float)
    data = data[~np.isnan(data)]
    if data.size == 0:
        return None, None
    if 0 < q_low < 1:
        low_th = np.quantile(data, q_low); data = data[data >= low_th]
    if 0 < q_high < 1:
        high_th = np.quantile(data, q_high); data = data[data <= high_th]
    if data.size == 0:
        return None, None
    data = np.sort(data)
    y = np.array([100.0]) if data.size == 1 else np.arange(data.size) / (data.size - 1) * 100.0
    return data, y

def calc_qoe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Network Type"] = df["File"].apply(categorize_trace)
    df["VMAF Change"] = df.get("Average VMAF Smoothness", np.nan)
    df["QoE"] = (
        0.13 * df["Average VMAF Score"]
        - 0.75 * df["Stall Time(s)"]
        - 0.2 * df["Switch Ratio(%)"]
    )
    return df

def safe_metric_name(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )

def _apply_method_order(df: pd.DataFrame) -> pd.DataFrame:
    """正規化名稱→（可選）只保留指定方法→轉為有序分類"""
    d = df.copy()
    d["Method"] = d["Method"].map(normalize_method_names)
    if KEEP_ONLY_METHOD_ORDER:
        d = d[d["Method"].isin(METHOD_ORDER)].copy()
    d["Method"] = pd.Categorical(d["Method"], categories=METHOD_ORDER, ordered=True)
    return d

# -----------------------------
# Plots
# -----------------------------
def _trim_quantiles(df: pd.DataFrame, col: str, q_low: float = 0.0, q_high: float = 1.0,
                    by=("Method", "Network Type")) -> pd.DataFrame:
    """依群組(by)對欄位 col 做分位數裁切"""
    if (q_low <= 0.0 and q_high >= 1.0) or col not in df.columns:
        return df

    def _cut(g: pd.DataFrame) -> pd.DataFrame:
        lo = g[col].quantile(q_low) if q_low > 0.0 else -np.inf
        hi = g[col].quantile(q_high) if q_high < 1.0 else  np.inf
        return g[(g[col] >= lo) & (g[col] <= hi)]

    return df.groupby(list(by), group_keys=False).apply(_cut)

def plot_boxplots(df: pd.DataFrame, q_keep_high: float = 0.99, q_keep_low: float = 0.0, d10_q_high2: float = 0.95):
    """畫 boxplots 前依 (Method, Network Type) 裁掉上下尾百分比（預設砍上尾 1%）"""
    plot_df = _apply_method_order(df)
    plot_df = plot_df[plot_df["Network Type"] == "Slow"].copy()

    n = len(BOX_METRICS)
    fig, axes = plt.subplots(1, n, figsize=(14, 5), sharex=False)
    if n == 1:
        axes = [axes]

    for i, (metric, unit) in enumerate(BOX_METRICS):
        ax = axes[i]
        trimmed = _trim_quantiles(plot_df, metric, q_low=q_keep_low, q_high=q_keep_high,
                                  by=("Method", "Network Type"))
         # 2) 只對 D(10) 再做一次額外過濾（高尾）
        if 0 < d10_q_high2 < 1.0:
            # 支援多種命名別名
            d10_alias = ["D(10)", "D-policy(10)", "D-policy-10"]
            msk = trimmed["Method"].isin(d10_alias)
            if msk.any():
                hi2 = trimmed.loc[msk, metric].quantile(d10_q_high2)
                trimmed = pd.concat(
                    [trimmed.loc[~msk],
                     trimmed.loc[msk & (trimmed[metric] <= hi2)]],
                    ignore_index=True
                )

        sns.boxplot(
            data=trimmed,
            x="Method",
            y=metric,
            hue="Network Type",
            hue_order=HUE_ORDER,
            order=METHOD_ORDER,
            ax=ax,
            showfliers=False
        )

        ax.set_title(f"{metric} by Method", fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis='x', labelsize=8)
        ax.set_ylabel(unit)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        # if i == 0:
        #     ax.legend(title="Network Type")
        # else:
        ax.get_legend().remove()

    fig.tight_layout()
    fig.savefig(SAVE_DIR / "box_plots.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_cdfs(df: pd.DataFrame):
    d = _apply_method_order(df)
    methods_all = [m for m in METHOD_ORDER if m in d["Method"].unique()]

    for metric, unit in CDF_METRICS:
        # by-network (3 子圖)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        for i, net in enumerate(["Slow", "Medium", "Fast"]):
            ax = axes[i]
            sub = d[d["Network Type"] == net]
            ax.set_title(f"{net} network")
            ax.set_xlabel(metric)
            if i == 0:
                ax.set_ylabel("CDF (%)")
            ax.set_ylim(0, 100)
            ax.grid(True, linestyle="--", alpha=0.5)
            if sub.empty:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
                continue
            for method, ls in zip(methods_all, LINESTYLES):
                v = sub.loc[sub["Method"] == method, metric].values
                x, y = ecdf_percent(v, q_low=0.05, q_high=0.99)
                if x is None:
                    continue
                ax.plot(x, y, linestyle=ls, linewidth=1.8, label=method)
            ax.legend(title="Method", fontsize=9)
        fig.tight_layout()
        fig.savefig(SAVE_DIR / f"cdf_{safe_metric_name(metric)}_by_network.pdf", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # overall
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title(f"CDF of {metric} (overall)", fontweight="bold")
        ax.set_xlabel(f"{metric}" + (f" ({unit})" if unit else ""))
        ax.set_ylabel("CDF (%)")
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle="--", alpha=0.5)
        for method, ls in zip(methods_all, LINESTYLES):
            v = d.loc[d["Method"] == method, metric].values
            x, y = ecdf_percent(v, q_low=0.05, q_high=0.99)
            if x is None:
                continue
            ax.plot(x, y, linestyle=ls, linewidth=1.8, label=method)
        ax.legend(title="Method", fontsize=9)
        fig.tight_layout()
        fig.savefig(SAVE_DIR / f"cdf_{safe_metric_name(metric)}_overall.pdf", dpi=200, bbox_inches="tight")
        plt.close(fig)

def calc_improvement(df: pd.DataFrame):
    d = _apply_method_order(df)
    out = {k: {} for k in ["Slow", "Medium", "Fast", "Overall"]}
    metrics = ["QoE", "Stall Time(s)"]
    for net in out.keys():
        sub = d if net == "Overall" else d[d["Network Type"] == net]
        for qaocs, base in IMPROVE_PAIRS:
            qa = sub[sub["Method"] == qaocs]
            bs = sub[sub["Method"] == base]
            if qa.empty or bs.empty:
                continue
            qa_vals = qa[metrics].mean()
            bs_vals = bs[metrics].mean()
            rec = {}
            for m in metrics:
                if m == "Stall Time(s)":
                    rec[m] = (bs_vals[m] - qa_vals[m]) / max(1e-12, bs_vals[m]) * 100.0
                else:
                    rec[m] = (qa_vals[m] - bs_vals[m]) / max(1e-12, bs_vals[m]) * 100.0
            out[net][base] = rec
    return out

def plot_improvement(df: pd.DataFrame):
    improvement = calc_improvement(df)
    metrics = ["QoE", "Stall Time(s)"]
    metric_labels = {"QoE": "QoE", "Stall Time(s)": "Average Stall Time (s)"}
    baseline_colors = {"Single(40)": "#1f77b4", "Single(50)": "#ff7f0e"}

    fig, axes = plt.subplots(1, len(IMPROVE_NETWORKS), figsize=(14, 5))
    if len(IMPROVE_NETWORKS) == 1:
        axes = [axes]

    for ax, net in zip(axes, IMPROVE_NETWORKS):
        data = improvement.get(net, {})
        baselines = [b for _, b in IMPROVE_PAIRS if b in data and all(m in data[b] for m in metrics)]
        if not baselines:
            ax.set_title(f"PQ-Policy Improvement for {net} Network Condition")
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
            continue

        x = np.arange(len(metrics))
        width = min(0.18, 0.8 / max(1, len(baselines)))
        offsets = (np.arange(len(baselines)) - (len(baselines)-1)/2) * (width + 0.02)

        for bi, b in enumerate(baselines):
            y = [float(data[b][m]) for m in metrics]
            bars = ax.bar(x + offsets[bi], y, width=width,
                          label=b, color=baseline_colors.get(b, "gray"))
            for rect, val in zip(bars, y):
                ax.text(rect.get_x() + rect.get_width()/2,
                        val + (0.1 if val >= 0 else -0.1),
                        f"{val:.1f}%", ha="center",
                        va="bottom" if val >= 0 else "top",
                        fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels([metric_labels[m] for m in metrics])
        ax.set_ylabel("Improvement (%)")
        ax.set_title(f"PQ-Policy Improvement - {net}", fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.set_ylim(-10, 50)
        ax.legend(title="Baseline")

    fig.tight_layout()
    fig.savefig(SAVE_DIR / "buffer_policy_improvement_all.pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)

def plot_cdf_plus_improvement(df: pd.DataFrame,
                              metric: str = "Stall Ratio(%)",
                              unit: str = "",
                              cdf_mode: str = "overall_by_method"):
    """
    上排：一張 CDF（'overall_by_method' or 'by_network'）
    下排：兩張改善圖（依 IMPROVE_NETWORKS 前兩項）
    """
    import matplotlib.gridspec as gridspec

    d = _apply_method_order(df)
    methods_all = [m for m in METHOD_ORDER if m in d["Method"].unique()]
    nets_for_imp = IMPROVE_NETWORKS[:2] if len(IMPROVE_NETWORKS) >= 2 else (IMPROVE_NETWORKS + ["Overall"])[:2]

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 2], figure=fig)

    # ---------- 上：CDF ----------
    ax_cdf = fig.add_subplot(gs[0, :])
    ax_cdf.set_title(f"CDF of {metric}(overall)", fontweight="bold", fontsize = 20)
    ax_cdf.set_xlabel(f"{metric}" + (f" ({unit})" if unit else ""), fontsize = 20)
    ax_cdf.set_ylabel("CDF (%)", fontsize = 20)
    ax_cdf.set_ylim(0, 100)
    ax_cdf.grid(True, linestyle="--", alpha=0.5)

    if cdf_mode == "overall_by_method":
        for method, ls in zip(methods_all, LINESTYLES):
            v = d.loc[d["Method"] == method, metric].values
            x, y = ecdf_percent(v, q_low=0.05, q_high=0.99)
            if x is None:
                continue
            ax_cdf.plot(x, y, linestyle=ls, linewidth=1.8, label=method)
        ax_cdf.legend(title="Method", fontsize=16,title_fontsize=16)
    else:
        for color, net in zip(COLORS, ["Slow", "Medium", "Fast"]):
            v = d.loc[d["Network Type"] == net, metric].values
            x, y = ecdf_percent(v, q_low=0.05, q_high=0.99)
            if x is None:
                continue
            ax_cdf.plot(x, y, linewidth=2, label=net, color=color)
        ax_cdf.legend(title="Network", fontsize=16,title_fontsize=16)

    # ---------- 下排：改善圖 ----------
    improvement = calc_improvement(d)
    metrics = ["QoE", "Stall Time(s)"]
    metric_labels = {"QoE": "QoE", "Stall Time(s)": "Average Stall Time (s)"}
    baseline_colors = {"Single(40)": "#1f77b4", "Single(50)": "#ff7f0e"}

    def draw_improve(ax, net_name: str):
        ax.set_title(f"PQ-Policy Improvement — {net_name}", fontweight="bold",fontsize = 20)
        ax.set_ylabel("Improvement (%)",fontsize = 20)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.set_ylim(-10, 50)

        data = improvement.get(net_name, {})
        baselines = [b for _, b in IMPROVE_PAIRS if b in data and all(m in data[b] for m in metrics)]

        x = np.arange(len(metrics))
        ax.set_xticks(x)
        ax.set_xticklabels([metric_labels[m] for m in metrics], fontsize = 20)

        if not baselines:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
            return

        width = min(0.18, 0.8 / max(1, len(baselines)))
        offsets = (np.arange(len(baselines)) - (len(baselines) - 1) / 2) * (width + 0.02)

        for bi, b in enumerate(baselines):
            y = [float(data[b][m]) for m in metrics]
            bars = ax.bar(x + offsets[bi], y, width=width,
                          label=b, color=baseline_colors.get(b, "gray"))
            for rect, val in zip(bars, y):
                ax.text(rect.get_x() + rect.get_width() / 2,
                        val + (0.1 if val >= 0 else -0.1),
                        f"{val:.1f}%", ha="center",
                        va="bottom" if val >= 0 else "top",
                        fontsize=9)
        ax.legend(title="Baseline", fontsize=12,title_fontsize=12)

    ax_imp_left  = fig.add_subplot(gs[1, 0])
    ax_imp_right = fig.add_subplot(gs[1, 1])
    draw_improve(ax_imp_left,  nets_for_imp[0])
    draw_improve(ax_imp_right, nets_for_imp[1])

    plt.tight_layout()
    fname = f"combo_cdf_improve_{safe_metric_name(metric)}"
    fig.savefig(SAVE_DIR / f"{fname}.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_cdf_plus_two_boxes(
    df,
    cdf_metric,
    box_metrics= None,
    network_for_box = "Slow",
    q_keep_high = 0.99,
    q_keep_low = 0.0,
    d10_q_high2 = 0.95,
    overall_cdf= True,
    filename = None,
):
    """
    產生一張含三個子圖的圖：上面 1 個 CDF（可 overall 或 by-network），下面 2 個 Box（預設取 Slow）。
    - cdf_metric: 要畫 CDF 的指標（需存在於 df 欄位）
    - box_metrics: 兩個要畫 Box 的 (metric1, metric2)；若 None 就取 BOX_METRICS 的前兩個
    - network_for_box: Box 的網路型態過濾（預設 'Slow'）
    - q_keep_low/high: Box 事前依 quantile 修剪
    - d10_q_high2: 僅對 D(10) 額外砍高尾（0~1；<=0 或 >=1 則不啟用）
    - overall_cdf: True→overall CDF；False→分網路 CDF（取 network_for_box）
    - filename: 檔名；若 None，自動依 metric 命名
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from matplotlib.patches import Patch

    d = _apply_method_order(df)
    methods_all = [m for m in METHOD_ORDER if m in d["Method"].unique()]

    # --- 決定下方兩個 box 的 metric ---
    if box_metrics is None:
        # 取 BOX_METRICS 的前兩個
        if len(BOX_METRICS) < 2:
            raise ValueError("BOX_METRICS 少於兩個，請指定 box_metrics")
        box_metrics = (BOX_METRICS[0][0], BOX_METRICS[1][0])

    # --- 版面配置：上 1、下 2 ---
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], hspace=0.35, wspace=0.25)
    ax_cdf = fig.add_subplot(gs[0, :])   # 上面跨兩欄
    ax_b1  = fig.add_subplot(gs[1, 0])   # 左下
    ax_b2  = fig.add_subplot(gs[1, 1])   # 右下

    # ======================
    # 上：CDF
    # ======================
    ax = ax_cdf
    ax.set_title(f"CDF of {cdf_metric} (overall)" + ("" if overall_cdf else f" ({network_for_box})"),
                 fontweight="bold", fontsize=12)
    ax.set_xlabel(f"{cdf_metric}", fontsize=12)
    ax.set_ylabel("CDF (%)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.5)

    if overall_cdf:
        sub = d.dropna(subset=[cdf_metric])
        for method, ls in zip(methods_all, LINESTYLES):
            v = sub.loc[sub["Method"] == method, cdf_metric].values
            x, y = ecdf_percent(v, q_low=0.05, q_high=0.99)
            if x is None:
                continue
            ax.plot(x, y, linestyle=ls, linewidth=1.8, label=method)
    else:
        sub = d[d["Network Type"] == network_for_box]
        for method, ls in zip(methods_all, LINESTYLES):
            v = sub.loc[sub["Method"] == method, cdf_metric].values
            x, y = ecdf_percent(v, q_low=0.05, q_high=0.99)
            if x is None:
                continue
            ax.plot(x, y, linestyle=ls, linewidth=1.8, label=method)

    # 只在上方 CDF 放 legend
    if ax.lines:
        ax.legend(title="Method", fontsize=9, title_fontsize=10, loc="upper left")

    # ======================
    # 下：兩個 Boxplots（同一個 Network Type）
    # ======================
    plot_df = d[d["Network Type"] == network_for_box].copy()
    # 依資料中實際出現的 Method 建色票（固定順序）
    methods_present = [m for m in METHOD_ORDER if m in plot_df["Method"].unique()]
    METHOD_PALETTE = dict(zip(methods_present, sns.color_palette("tab10", len(methods_present))))

    def _one_box(ax_box, metric: str):
        trimmed = _trim_quantiles(plot_df, metric, q_low=q_keep_low, q_high=q_keep_high,
                                by=("Method", "Network Type"))
        if 0 < d10_q_high2 < 1.0:
            d10_alias = ["D(10)", "D-policy(10)", "D-policy-10"]
            msk = trimmed["Method"].isin(d10_alias)
            if msk.any():
                hi2 = trimmed.loc[msk, metric].quantile(d10_q_high2)
                trimmed = pd.concat([trimmed.loc[~msk],
                                    trimmed.loc[msk & (trimmed[metric] <= hi2)]],
                                    ignore_index=True)


        # ★ x=Method + hue=Method，記得 dodge=False（避免重複分箱）
        sns.boxplot(
            data=trimmed,
            x="Method",
            y=metric,
            order=methods_present,        # 用實際出現的方法順序
            hue="Method",
            hue_order=methods_present,
            palette=METHOD_PALETTE,
            dodge=False,                  # ★ 關鍵
            showfliers=False,
            ax=ax_box,
        )

        BOX_METRICS = [
            ("Average VMAF Score", "Score"),
            ("Stall Ratio(%)", "Ratio (%)"),
            ("Stall Time(s)", "Time (s)"),
            ("Average Bitrate(bps)", "bps"),
        ]
        title_metrics = [
            ("Stall Ratio(%)", "Stall Ratio"),
            ("Stall Time(s)", "Stall Time"),    
        ]
        ax_box.set_xticklabels([])   # 不顯示每個方法名
        ax_box.set_xlabel("Method", fontsize=12)
        label = next((u for (m, u) in title_metrics if m == metric), metric)
        ax_box.set_title(f"{label} by Method for {network_for_box} Network Condition", fontweight="bold", fontsize=10)
        ax_box.tick_params(axis='x', labelsize=6, rotation=0)
        ax_box.set_ylabel(next((u for (m, u) in BOX_METRICS if m == metric), ""), fontsize=12)
        ax_box.grid(axis="y", linestyle="--", alpha=0.4)
        if ax_box.get_legend() is not None:
            ax_box.get_legend().remove()


        # # ⭐（可選）加上「Method 的 legend」：只在其中一張子圖加即可
        # if add_method_legend:
        #     from matplotlib.patches import Patch
        #     handles = [Patch(facecolor=METHOD_PALETTE[m], edgecolor='black', label=m)
        #             for m in methods_present]
        #     ax_box.legend(handles=handles, title="Method",
        #                 loc="upper right", fontsize=8, title_fontsize=9)


    _one_box(ax_b1, box_metrics[0])
    _one_box(ax_b2, box_metrics[1])

    # ======================
    # 輸出
    # ======================
    handles = [Patch(facecolor=METHOD_PALETTE[m], edgecolor='black', label=m)
               for m in methods_present]

    if handles:
        ax_b1.legend(handles=handles, title='Method',
                    loc='upper right',    # 右上角
                    fontsize=8, title_fontsize=8,
                    frameon=True) 
        
    if filename is None:
        tag = f"{safe_metric_name(cdf_metric)}_{network_for_box if not overall_cdf else 'overall'}"
        filename = f"combo_cdf_boxes_{tag}.pdf"


    fig.savefig(SAVE_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main():
    df = read_concat(FILE_PATHS)
    df = calc_qoe(df)  

    #plot_boxplots(df, q_keep_high=0.99, q_keep_low=0.0)
    #plot_cdfs(df)
    # plot_improvement(df)
    #plot_cdf_plus_improvement(df, metric="QoE", unit="", cdf_mode="overall_by_method")
    plot_cdf_plus_two_boxes(
    df,
    cdf_metric="QoE",
    box_metrics=("Stall Ratio(%)", "Stall Time(s)"),
    network_for_box="Slow",        # 或 "Medium"/"Fast"
    q_keep_high=0.99, q_keep_low=0.0, d10_q_high2=0.99,
    overall_cdf=True               # False 會畫指定 network 的 CDF
    )

    print("All plots saved to:", SAVE_DIR.resolve())

if __name__ == "__main__":
    main()
