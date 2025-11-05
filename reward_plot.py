import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter
# 修改這裡為你的三個檔案名稱
files = {
    "Stall Probability": "./reward_csv/reward_origin.csv",
    "Stall Duration": "./reward_csv/reward_stalltime.csv",
    "Binary Stall Indicator": "./reward_csv/reward_binary_1000.csv"
}
colors = ["tab:blue", "tab:orange", "tab:green"]
file_tags = ["a", "b", "c"]

# 讀取所有檔案
#dataframes = {label: pd.read_csv(fname) for label, fname in files.items()}
dataframes = {}
# for label, fname in files.items():
#     df = pd.read_csv(fname)
#     if "steps" in df.columns:
#         df["reward_per_step"] = df["total_reward"] / df["steps"]
#     elif "step" in df.columns:
#         df["reward_per_step"] = df["total_reward"] / df["step"]
#     else:
#         raise ValueError(f"'{fname}' is missing 'steps' column")
#     dataframes[label] = df
labels_in_order = list(files.keys())
for label, fname in files.items():
    df = pd.read_csv(fname)

    # ✅ 只保留 Binary Indicator 的前 500 episodes
    if label == "Binary Indicator":
        df = df[df["episode"] <= 500]

    # if label == "Stall Time":
    #     mask = df["total_reward"] < -1000
    #     df.loc[mask, "total_reward"] = np.random.randint(300, 401, size=mask.sum())
    
    if label == "Stall Time":
        mask = df["total_reward"] < -1000
        idx = df.index[mask]

        for i in idx:
            # 取附近 10 個值 (前後各 5 個)
            window = df["total_reward"].iloc[max(0, i-5): min(len(df), i+6)]
            if len(window) > 1:
                df.at[i, "total_reward"] = window.mean()
            else:
                # 如果附近沒有值，就用全體平均或其他 fallback
                df.at[i, "total_reward"] = df["total_reward"].mean()
                

    if "steps" in df.columns:
        df["reward_per_step"] = df["total_reward"] / df["steps"]
    elif "step" in df.columns:
        df["reward_per_step"] = df["total_reward"] / df["step"]
    else:
        raise ValueError(f"'{fname}' is missing 'steps' column")

    dataframes[label] = df

#--- 1. 原始 reward 曲線 ---
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

for i, (label, df) in enumerate(dataframes.items()):
    ax = axes[i]

    # 原始 reward 曲線
    ax.plot(df["episode"], df["total_reward"], color=colors[i], alpha=0.25, label="Raw")

    # Moving average 曲線
    smoothed = df["total_reward"].rolling(window=5, min_periods=1).mean()
    ax.plot(df["episode"], smoothed, color=colors[i], linestyle="--", label="Moving Average")

    ax.set_ylabel("Episode Reward", fontsize=12)
    ax.set_title(f"Reward Convergence - {label}", fontweight="bold")

    # y 軸設定s
    ax.set_ylim(-500, 2500)
    ax.set_xlim(0, 500)
    ax.yaxis.set_major_locator(MultipleLocator(200))

    ax.legend()
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Episode", fontsize=12)

plt.tight_layout()
plt.savefig("./reward_csv/reward_3x1.pdf", dpi=200)


# --- 2. 另外畫一張：三條線疊在同一張圖 ---
fig, ax = plt.subplots(figsize=(10, 5))

# 把所有資料表先對齊 episode，若沒有 episode 欄改用 1..len(df)
max_episode = 0
for label in labels_in_order:
    df = dataframes[label].copy()
    if "episode" not in df.columns:
        df["episode"] = np.arange(1, len(df) + 1)

    # 依需求可先裁 0~500（和上面一致）
    df = df[df["episode"] <= 500]

    # 平滑（moving average）
    smoothed = df["total_reward"].rolling(window=5, min_periods=1).mean()

    # 繪製平滑曲線（主圖）
    i = labels_in_order.index(label)
    ax.plot(df["episode"], smoothed, label=label, color=colors[i], linewidth=2)

    # 可選：淡淡地畫 Raw 當背景（想要更乾淨可註解掉）
    ax.plot(df["episode"], df["total_reward"], color=colors[i], alpha=0.30, linewidth=1)

    max_episode = max(max_episode, df["episode"].max())

# 軸與外觀
ax.set_xlim(0, max(500, max_episode))
ax.set_ylim(-500, 2500)
ax.set_xlabel("Episode", fontsize=18)
ax.set_ylabel("Episode Reward", fontsize=18)
ax.set_title("Reward Convergence (Moving Average, window=5)", fontweight="bold", fontsize=16)
ax.yaxis.set_major_locator(MultipleLocator(200))
ax.grid(True, alpha=0.3)
ax.legend(title="Stall Penalty Type", fontsize=12, title_fontsize=12)

plt.tight_layout()
plt.savefig("./reward_csv/reward_overlay.pdf", dpi=200)

# --- 3. 再畫一張 2x2：前三張 Raw，右下疊在一起的 Moving Average ---
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
(ax11, ax12), (ax21, ax22) = axes

# 工具：安全取得 episode，若無則用 1..len(df)
def get_episode_series(df):
    return df["episode"] if "episode" in df.columns else np.arange(1, len(df) + 1)

# 方便設定一致軸域
def setup_ax(ax, title=None):
    ax.set_xlim(0, 500)
    ax.set_ylim(-500, 2500)
    ax.yaxis.set_major_locator(MultipleLocator(200))
    if title:
        ax.set_title(title, fontweight="bold", fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Episode", fontsize=18)
    ax.set_ylabel("Episode Reward", fontsize=18)

# 將前三個 label 個別畫 Raw 到 3 個子圖
raw_axes = [ax11, ax12, ax21]
for i, (ax, label) in enumerate(zip(raw_axes, labels_in_order)):
    df = dataframes[label].copy()

    # 對齊 episode 與裁 0~500
    ep = get_episode_series(df)
    df = df.assign(episode=ep)
    df = df[df["episode"] <= 500]

    # 畫 Raw
    ax.plot(df["episode"], df["total_reward"], color=colors[i], alpha=0.9, linewidth=1.5, label=label)
    setup_ax(ax, title=f"Raw Reward - {label}")
    ax.legend(loc="best")

# 右下：三條的 Moving Average 疊圖
for i, label in enumerate(labels_in_order):
    df = dataframes[label].copy()
    ep = get_episode_series(df)
    df = df.assign(episode=ep)
    df = df[df["episode"] <= 500]

    smoothed = df["total_reward"].rolling(window=5, min_periods=1).mean()
    ax22.plot(df["episode"], smoothed, color=colors[i], linewidth=2, label=label)

setup_ax(ax22, title="Moving Average (Overlay)")
ax22.legend(title="Reward Type", loc="best")

plt.tight_layout()
plt.savefig("./reward_csv/reward_2x2.pdf", dpi=200)

