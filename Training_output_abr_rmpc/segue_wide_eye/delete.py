import os

# 21 個要保留的 CSV 檔案名稱
keep_files = {
    "fcc18_trace_1.txt.csv",
    "fcc18_trace_2.txt.csv",
    "fcc18_trace_3.txt.csv",
    "ghent_trace_0.txt.csv", 
    "ghent_trace_1.txt.csv",
    "ghent_trace_2.txt.csv",
    "ghent_trace_3.txt.csv",
    "hsr_trace_1.txt.csv",
    "hsr_trace_2.txt.csv",
    "hsr_trace_3.txt.csv",
    "lab_trace_1.txt.csv",
    "lab_trace_2.txt.csv",
    "lab_trace_3.txt.csv",
    "lumos5g_trace_1.txt.csv",
    "lumos5g_trace_2.txt.csv",
    "lumos5g_trace_3.txt.csv",
    "lumos5g_trace_4.txt.csv",
    "oboe_trace_0.txt.csv",
    "oboe_trace_1.txt.csv",
    "oboe_trace_2.txt.csv",
    "oboe_trace_3.txt.csv" 

}
target_folders = [
    "./LOL_3D",
    "./sport_highlight",
    "./video_game"
]

for folder in target_folders:
    for f in os.listdir(folder):
        if f.endswith(".csv") and f not in keep_files:
            file_path = os.path.join(folder, f)
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")