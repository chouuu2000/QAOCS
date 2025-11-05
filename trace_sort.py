import os
import random
import json

def get_shuffled_trace_list(trace_folder, seed=42, output_file="trace_order.json"):
    # 讀取所有 trace 檔案
    trace_files = sorted([
        f for f in os.listdir(trace_folder)
        if os.path.isfile(os.path.join(trace_folder, f))
    ])

    # 設定隨機種子，確保每次打亂順序一致
    random.seed(seed)
    random.shuffle(trace_files)

    # 儲存這次的順序（給其他 script 使用）
    with open(output_file, "w") as f:
        json.dump(trace_files, f, indent=2)

    print(f"Saved shuffled trace order to {output_file}")
    return trace_files

# 使用範例
trace_folder = "./train/"
trace_order = get_shuffled_trace_list(trace_folder, seed=999)